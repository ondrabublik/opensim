"""Konvoluční neuronová síť pro řízení motoru ON/OFF (jen výstup do schodů).

Spouští se v Pythonu s PyTorch (zde 3.11), OpenSim není potřeba:
    py -3.11 cnn_train.py

Vstup sítě: posledních 1 s signálů vzorkovaných 20 Hz (20 vzorků x 3 kanály)
    flexe kolene [deg], úhlová rychlost stehna a holeně vůči zemi [deg/s]
Výstup: pravděpodobnost "motor ON" pro aktuální okamžik (práh 0,5).

Štítky (učitel): moment kolene při normální chůzi bez exo (ID z test1)
    ON  <=> extenzní moment > PRAH_MOMENT  a  pravá noha stojí na schodu
(na rovině se nepomáhá, i kdyby moment práh překročil).

Data: 15 m chůze z test1 (200 Hz) -> 20 Hz s 10 posuny fáze vzorkování a augmentací
(rychlost chůze, amplituda, šum a offset snímačů). Testovací množina = poslední cyklus
roviny a poslední 2 cykly schodů (včetně 1 s ochranného pásma), do tréninku nevstupují.

Výstupy: results/cnn/cnn_vahy.npz (váhy pro numpy inferenci v cnn_rizeni.py), cnn_model.pt,
         cnn_metriky.json, grafy.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(ROOT, "data"), os.path.join(ROOT, "results")
OUT = os.path.join(RES, "cnn")

FS_RAW = 200.0
FS = 20.0                  # vzorkovací frekvence řízení [Hz]
WIN = 20                   # délka okna = 1 s
PRAH_MOMENT = 40.0         # [Nm] extenzní moment kolene, nad kterým má motor pomáhat
SCALE = np.array([100.0, 300.0, 300.0])   # normalizace kanálů (pevná, vhodné pro MCU)
N_AUG = 240                # počet augmentovaných variant trénovací sekvence
SEED = 1


# ------------------------------------------------------------------ data
def read_mot(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[i + 1].split("\t")
    return cols, np.array([[float(x) for x in l.split()] for l in lines[i + 2:] if l.strip()])


def load_sequence():
    """Signály 200 Hz: vstupy (n,3), štítek (n,), moment bez exo, čas, meta."""
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    r = np.load(os.path.join(RES, "exo_vysledky.npz"))
    t = r["t"]
    cols, kd = read_mot(os.path.join(DATA, "walk_stairs_ik.mot"))
    n = len(t)
    q = {c: kd[:n, cols.index(c)] for c in cols}
    stehno = q["pelvis_tilt"] + q["hip_flexion_r"]
    holen = stehno + q["knee_angle_r"]
    X = np.column_stack([-q["knee_angle_r"], np.gradient(stehno, t), np.gradient(holen, t)])
    on_stair = np.zeros(n, bool)
    for c in meta["contacts"]["r"]:
        if c["surface"] > 0.01:
            on_stair |= (t >= c["t_on"]) & (t <= (c["t_off"] if c["t_off"] is not None else t[-1]))
    y = ((r["M_bez"] > PRAH_MOMENT) & on_stair).astype(float)
    return t, X, y, r, meta, on_stair


def test_mask(t, meta):
    """Testovací úseky: poslední cyklus roviny a poslední 2 cykly schodů."""
    Tc = meta["Tc"]
    tk = {int(k): v for k, v in meta["t_k"].items()}
    iv = [(6 * Tc, 7 * Tc), (tk[14], t[-1] + 1)]
    m = np.zeros(len(t), bool)
    guard = np.zeros(len(t), bool)
    for a, b in iv:
        m |= (t >= a) & (t < b)
        guard |= (t >= a) & (t < b + 1.0)      # okna končící do 1 s po testu vidí testovací data
    return m, guard, iv


def to_20hz(t, X, y, phase):
    idx = np.arange(phase, len(t), int(FS_RAW / FS))
    return t[idx], X[idx], y[idx]


def augment(t, X, y, rng):
    """Náhodná varianta: rychlost (časové škálování), amplituda, šum, offset gyroskopů."""
    s = rng.uniform(0.85, 1.15)                   # s > 1 = rychlejší chůze
    tn = np.arange(t[0], t[-1] * 1.0 / s, 1.0 / FS_RAW)
    src = tn * s
    Xa = np.column_stack([np.interp(src, t, X[:, k]) for k in range(3)])
    Xa[:, 1:] *= s
    ya = np.interp(src, t, y) > 0.5
    Xa *= rng.uniform(0.9, 1.1, size=3)
    Xa[:, 0] += rng.normal(0, 2.0)                # offset kalibrace úhlu
    Xa[:, 1:] += rng.normal(0, 5.0, size=2)       # bias gyroskopů
    Xa += rng.normal(0, 1, Xa.shape) * np.array([0.5, 6.0, 6.0])   # šum
    return tn, Xa, ya.astype(float), src


def windows(X20, y20, keep=None):
    """Klouzavá okna 1 s končící v každém vzorku (kauzální)."""
    Xn = (X20 / SCALE).astype(np.float32)
    idx = np.arange(WIN - 1, len(X20))
    if keep is not None:
        idx = idx[keep[idx]]
    W = np.stack([Xn[i - WIN + 1:i + 1].T for i in idx])    # (N, 3, WIN)
    return W, y20[idx].astype(np.float32), idx


# ------------------------------------------------------------------ síť
class Net(nn.Module):
    """3x20 -> Conv(3->12,k5) -> ReLU -> Conv(12->12,k5) -> ReLU -> Conv(12->8,k3) -> ReLU
    -> flatten(8x10) -> Dense(80->16) -> ReLU -> Dense(16->1)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 12, 5)
        self.c2 = nn.Conv1d(12, 12, 5)
        self.c3 = nn.Conv1d(12, 8, 3)
        self.f1 = nn.Linear(8 * (WIN - 10), 16)
        self.f2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = torch.relu(self.f1(x.flatten(1)))
        return self.f2(x).squeeze(1)


def scores(y, p):
    tp = float(((p == 1) & (y == 1)).sum())
    fp = float(((p == 1) & (y == 0)).sum())
    fn = float(((p == 0) & (y == 1)).sum())
    tn = float(((p == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"presnost": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "F1": 2 * prec * rec / max(prec + rec, 1e-9), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    t, X, y, r, meta, on_stair = load_sequence()
    tmask, guard, iv = test_mask(t, meta)

    # ---------------- trénovací okna: augmentované varianty, bez testovacích úseků
    Wtr, ytr = [], []
    for k in range(N_AUG):
        tn, Xa, ya, src = augment(t, X, y, rng)
        g = np.interp(src, t, guard.astype(float)) > 0.5
        ph = rng.integers(0, int(FS_RAW / FS))
        idx = np.arange(ph, len(tn), int(FS_RAW / FS))
        W, yy, _ = windows(Xa[idx], ya[idx], keep=~g[idx])
        Wtr.append(W)
        ytr.append(yy)
    Wtr, ytr = np.concatenate(Wtr), np.concatenate(ytr)
    # validace pro výběr epochy: čistá data, testovací úseky se nepoužijí
    Wva, yva = [], []
    for ph in range(0, 10, 3):
        t20, X20, y20 = to_20hz(t, X, y, ph)
        W, yy, _ = windows(X20, y20, keep=~(np.interp(t20, t, guard.astype(float)) > 0.5))
        Wva.append(W)
        yva.append(yy)
    Wva, yva = np.concatenate(Wva), np.concatenate(yva)
    print(f"Trénovací okna: {len(ytr)} (ON {ytr.mean():.1%}), validační {len(yva)}")

    net = Net()
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    pos_w = torch.tensor((1 - ytr.mean()) / ytr.mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, Yt = torch.from_numpy(Wtr), torch.from_numpy(ytr)
    Xv, Yv = torch.from_numpy(Wva), torch.from_numpy(yva)
    hist, best, best_state = [], 1e9, None
    for ep in range(40):
        net.train()
        perm = torch.randperm(len(Yt))
        tot = 0.0
        for b in range(0, len(perm), 256):
            j = perm[b:b + 256]
            opt.zero_grad()
            loss = loss_fn(net(Xt[j]), Yt[j])
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        net.eval()
        with torch.no_grad():
            lv = loss_fn(net(Xv), Yv).item()
        hist.append((tot / len(Yt), lv))
        if lv < best:
            best, best_state = lv, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 5 == 4:
            print(f"  epocha {ep + 1:2d}: ztráta trénink {hist[-1][0]:.4f}, validace {lv:.4f}")
    net.load_state_dict(best_state)
    net.eval()

    # ---------------- vyhodnocení: test (čistá data, všech 10 fází) + odolnost na šum/rychlost
    def predict_seq(t_, X_, ph):
        idx = np.arange(ph, len(t_), int(FS_RAW / FS))
        Xn = (X_[idx] / SCALE).astype(np.float32)
        p = np.zeros(len(idx))
        W = np.stack([np.pad(Xn, ((WIN - 1, 0), (0, 0)), mode="edge")[i:i + WIN].T for i in range(len(idx))])
        with torch.no_grad():
            p[:] = torch.sigmoid(net(torch.from_numpy(W))).numpy()
        return idx, p

    res = {}
    allp = {}
    for name, fn in (("cista_data", lambda: (t, X)),
                     ("sum_2x", lambda: (t, X + rng.normal(0, 1, X.shape) * np.array([1.0, 12.0, 12.0]))),):
        ys, ps = [], []
        for ph in range(10):
            t_, X_ = fn()
            idx, p = predict_seq(t_, X_, ph)
            sel = tmask[idx]
            ys.append(y[idx][sel])
            ps.append((p[sel] > 0.5).astype(float))
            if name == "cista_data" and ph == 0:
                allp["idx"], allp["p"] = idx, p
        res[name] = scores(np.concatenate(ys), np.concatenate(ps))
    # rychlost +-15 % (časové škálování celé sekvence, test na testovacích úsecích)
    for s in (0.85, 1.15):
        tn = np.arange(t[0], t[-1] / s, 1 / FS_RAW)
        Xs = np.column_stack([np.interp(tn * s, t, X[:, k]) for k in range(3)])
        Xs[:, 1:] *= s
        ys_ = np.interp(tn * s, t, y) > 0.5
        tm = np.interp(tn * s, t, tmask.astype(float)) > 0.5
        idx, p = predict_seq(tn, Xs, 0)
        res[f"rychlost_{s:g}"] = scores(ys_[idx][tm[idx]].astype(float), (p[tm[idx]] > 0.5).astype(float))
    for k, v in res.items():
        print(f"  test {k:12s}: přesnost {v['presnost']:.3f}, precision {v['precision']:.3f}, "
              f"recall {v['recall']:.3f}, F1 {v['F1']:.3f}")

    # ---------------- uložení
    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    np.savez(os.path.join(OUT, "cnn_vahy.npz"), scale=SCALE, fs=FS, win=WIN, **sd)
    torch.save(net.state_dict(), os.path.join(OUT, "cnn_model.pt"))
    np.savez(os.path.join(OUT, "cnn_trenink.npz"), hist=np.array(hist), t=t, y=y,
             on_stair=on_stair, test_mask=tmask, idx20=allp["idx"], p20=allp["p"], X=X)
    json.dump({"parametru": int(n_par), "prah_moment_Nm": PRAH_MOMENT, "fs_Hz": FS, "okno_s": WIN / FS,
               "trenovacich_oken": int(len(ytr)), "podil_ON": float(ytr.mean()),
               "testovaci_useky_s": [[float(a), float(min(b, t[-1]))] for a, b in iv],
               "test": res, "epochy": len(hist)},
              open(os.path.join(OUT, "cnn_metriky.json"), "w"), indent=1)
    print(f"Síť: {n_par} parametrů, uloženo do {OUT}")


if __name__ == "__main__":
    main()
