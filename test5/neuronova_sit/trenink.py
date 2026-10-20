"""Učení konvoluční sítě pro řízení motoru exoskeletu z exportovaných dat chůze.

    py -3.11 trenink.py

Vstupy: data/<osoba>.csv (export test5/scripts/s8_export_nn.py), nastavení nastaveni.json.
Typ řízení ("typ" v nastaveni.json):
  "regrese" - síť odhaduje moment kolene [N m/kg], který má převzít motor:
              cíl = podíl(úsek) * max(moment kolene člověka, 0)  v úsecích z "regrese.useky", jinak 0,
              posunutý o "predstih_s" dopředu; přebytek (motor víc než potřeba) se trestá
              "pokuta_prekompenzace"-krát víc než nedostatek.
              Řízení výstup násobí hmotností uživatele (a omezí max. momentem motoru).
  "onoff"   - ON <=> moment kolene [N m/kg] > prah v úsecích "stitek.useky" (a pravá noha stojí).
Síť: vstup = vybrané signály za poslední okno (20 Hz x 1 s), 3x Conv1D + 2x Dense.
Vyhodnocení: leave-one-person-out po skupinách (osoba = všechny její rychlosti chůze), potom
konečná síť ze všech záznamů.
Výstupy: vahy/cnn_regrese.npz nebo vahy/cnn_vahy.npz (numpy formát pro řízení v test5/s4_exo.py),
         vahy/*.pt, vysledky/metriky_<typ>.json, vysledky/graf_<typ>_*.png
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "nastaveni.json"), encoding="utf-8"))
TYP = CFG.get("typ", "onoff")
FS_RAW = 200.0
FS = CFG["fs_Hz"]
WIN = int(round(CFG["okno_s"] * FS))
SCALE = np.array(CFG["normalizace"], float)
STEP = int(round(FS_RAW / FS))
OUT_NPZ = os.path.join(HERE, (CFG["regrese"] if TYP == "regrese" else CFG["stitek"]).get(
    "vystup_soubor", "vahy/cnn_regrese.npz" if TYP == "regrese" else "vahy/cnn_vahy.npz"))


# ------------------------------------------------------------------ data
def load_person(csv):
    lines = open(csv, encoding="utf-8").read().splitlines()
    head = lines[0].split(";")
    rows = [l.split(";") for l in lines[1:]]
    usek = np.array([r[0] for r in rows])
    num = np.array([[float(x) for x in r[1:]] for r in rows])
    col = {h: num[:, i - 1] for i, h in enumerate(head) if i > 0}
    meta = json.load(open(csv[:-4] + ".json", encoding="utf-8"))
    X = np.column_stack([col[c] for c in CFG["vstupy"]])
    M = col["moment_kolene_Nm_kg"]
    if TYP == "regrese":
        r = CFG["regrese"]
        podil = np.array([r["useky"].get(u, 0.0) for u in usek])
        y = podil * (np.clip(M, 0, None) if r.get("jen_extenze", True) else M)
        # předstih: síť odhaduje moment o predstih_s dopředu (vyrovná zpoždění řízení 20 Hz a náběh motoru)
        y = np.interp(col["cas_s"] + r.get("predstih_s", 0.0), col["cas_s"], y)
        w = np.array([r.get("vaha_useku", {}).get(u, r.get("vaha_ostatni", 1.0)) for u in usek])
    else:
        st = CFG["stitek"]
        y = (M > st["prah_Nm_kg"]) & np.isin(usek, st["useky"])
        if st.get("jen_stoj", True):
            y &= col["stoj_prava"] > 0.5
        y, w = y.astype(float), np.ones(len(y))
    return dict(jmeno=meta["jmeno"], skupina=meta.get("skupina", meta["jmeno"]), t=col["cas_s"], X=X,
                y=y, w=w, M=M, usek=usek, meta=meta)


def augment(p, rng):
    a = CFG["augmentace"]
    s = rng.uniform(*a["rychlost"])
    t = p["t"]
    tn = np.arange(t[0], t[-1] / s, 1.0 / FS_RAW)
    src = tn * s
    X = np.column_stack([np.interp(src, t, p["X"][:, k]) for k in range(p["X"].shape[1])])
    for k, c in enumerate(CFG["vstupy"]):            # rychlosti se škálují s rychlostí chůze
        if "omega" in c or "rychlost" in c:
            X[:, k] *= s
    y = np.interp(src, t, p["y"])
    if TYP != "regrese":
        y = (y > 0.5).astype(float)
    w = np.interp(src, t, p["w"])
    X *= rng.uniform(1 - a["zesileni"], 1 + a["zesileni"], size=X.shape[1])
    for k, c in enumerate(CFG["vstupy"]):
        X[:, k] += rng.normal(0, a["offset_uhlu_deg"] if "deg" in c and "_s" not in c else a["bias_gyra_deg_s"])
    X += rng.normal(0, 1, X.shape) * np.array(a["sum"][:X.shape[1]])
    return X, y, w


def windows(X, y, w, phase=0):
    idx = np.arange(phase, len(X), STEP)
    Xn = (X[idx] / SCALE).astype(np.float32)
    Xp = np.pad(Xn, ((WIN - 1, 0), (0, 0)), mode="edge")         # začátek: okno vyplněné 1. vzorkem
    W = np.stack([Xp[i:i + WIN].T for i in range(len(idx))])
    return W, y[idx].astype(np.float32), w[idx].astype(np.float32), idx


def dataset(persons, rng):
    Ws, ys, ws = [], [], []
    for p in persons:
        for k in range(CFG["augmentace"]["variant_na_osobu"]):
            X, y, w = augment(p, rng)
            W, yy, ww, _ = windows(X, y, w, rng.integers(0, STEP))
            Ws.append(W)
            ys.append(yy)
            ws.append(ww)
    return np.concatenate(Ws), np.concatenate(ys), np.concatenate(ws)


# ------------------------------------------------------------------ síť (stejná jako test2)
class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.c1 = nn.Conv1d(n_in, 12, 5)
        self.c2 = nn.Conv1d(12, 12, 5)
        self.c3 = nn.Conv1d(12, 8, 3)
        self.f1 = nn.Linear(8 * (WIN - 10), 16)
        self.f2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        return self.f2(torch.relu(self.f1(x.flatten(1)))).squeeze(1)


def train(W, y, w, seed):
    torch.manual_seed(seed)
    net = Net(W.shape[1])
    u = CFG["uceni"]
    opt = torch.optim.Adam(net.parameters(), lr=u["krok"], weight_decay=1e-4)
    if TYP == "regrese":
        k_over = float(CFG["regrese"].get("pokuta_prekompenzace", 1.0))

        def loss_fn(out, tgt, wt):                         # vážená kvadratická chyba, přebytek k_over x dražší
            e = out - tgt
            return torch.sum(wt * torch.where(e > 0, k_over, 1.0) * e ** 2) / torch.sum(wt)
    else:
        pos = max(float(y.mean()), 1e-3)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos))

        def loss_fn(out, tgt, wt):
            return bce(out, tgt)
    Xt, Yt, Wt = torch.from_numpy(W), torch.from_numpy(y), torch.from_numpy(w)
    hist = []
    for ep in range(u["epochy"]):
        net.train()
        perm = torch.randperm(len(Yt))
        tot = 0.0
        for b in range(0, len(perm), u["davka"]):
            j = perm[b:b + u["davka"]]
            opt.zero_grad()
            loss = loss_fn(net(Xt[j]), Yt[j], Wt[j])
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        hist.append(tot / len(Yt))
    net.eval()
    return net, hist


def predict(net, p):
    """Kauzální výpočet 20 Hz na čistých datech osoby, výstup držený do dalšího vzorku (200 Hz)."""
    W, _, _, idx = windows(p["X"], p["y"], p["w"])
    with torch.no_grad():
        out = net(torch.from_numpy(W)).numpy()
    if TYP == "regrese":
        if CFG["regrese"].get("jen_extenze", True):
            out = np.clip(out, 0, None)
    else:
        out = 1 / (1 + np.exp(-out))
    k = np.clip(np.searchsorted(idx, np.arange(len(p["t"])), side="right") - 1, 0, len(idx) - 1)
    return out[k]


def scores(p, out):
    usek = p["usek"]
    res = {"po_usecich": {}}
    if TYP == "regrese":
        y = p["y"]
        res["RMSE_Nm_kg"] = float(np.sqrt(np.mean((out - y) ** 2)))
        res["R2"] = float(1 - np.sum((out - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
        for u in np.unique(usek):
            m = usek == u
            resid = p["M"][m] - out[m]                     # moment, který zbude člověku (bez omezení motoru)
            res["po_usecich"][u] = {"RMSE_Nm_kg": round(float(np.sqrt(np.mean((out[m] - y[m]) ** 2))), 3),
                                    "vystup_prumer_Nm_kg": round(float(out[m].mean()), 3),
                                    "spicka_cloveka_bez_exo": round(float(p["M"][m].max()), 2),
                                    "spicka_cloveka_s_exo": round(float(resid.max()), 2)}
    else:
        y, on = p["y"] > 0.5, out > 0.5
        tp, fp, fn = np.sum(on & y), np.sum(on & ~y), np.sum(~on & y)
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res.update(precision=float(prec), recall=float(rec), F1=float(2 * prec * rec / max(prec + rec, 1e-9)))
        for u in np.unique(usek):
            res["po_usecich"][u] = {"zapnuto_%": round(float(on[usek == u].mean() * 100), 1)}
    return res


def summary_line(sc):
    if TYP == "regrese":
        part = ", ".join(f"{u}: {v['spicka_cloveka_bez_exo']:.2f}->{v['spicka_cloveka_s_exo']:.2f}"
                         for u, v in sc["po_usecich"].items() if u in ("sed + vstávání", "schody nahoru", "schody dolů"))
        lev = sc["po_usecich"].get("rovina", {}).get("vystup_prumer_Nm_kg", float("nan"))
        return f"RMSE {sc['RMSE_Nm_kg']:.3f} N m/kg, R2 {sc['R2']:.3f} | špička člověka [N m/kg] {part} | rovina výstup {lev:.3f}"
    return (f"F1 {sc['F1']:.3f}, precision {sc['precision']:.3f}, recall {sc['recall']:.3f} | zapnuto: "
            + str({u: v["zapnuto_%"] for u, v in sc["po_usecich"].items()}))


def save(net, path_npz):
    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    extra = {"typ": np.array(TYP), "jen_extenze": np.array(bool(CFG.get("regrese", {}).get("jen_extenze", True)))}
    np.savez(path_npz, scale=SCALE, fs=FS, win=WIN, vstupy=np.array(CFG["vstupy"]), **extra, **sd)
    torch.save(net.state_dict(), path_npz[:-4] + ".pt")


def plot(persons, preds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(persons), 1, figsize=(13, 2.3 * len(persons)), sharex=True, squeeze=False)
    for a, p, out in zip(axs[:, 0], persons, preds):
        if TYP == "regrese":
            a.plot(p["t"], p["M"], color="#9a9893", lw=0.8, label="moment kolene člověka bez exo")
            a.fill_between(p["t"], 0, p["y"], color="#1baf7a", alpha=0.35, lw=0, label="cíl motoru")
            a.plot(p["t"], out, color="#eb6834", lw=1.0, label="výstup sítě")
            a.set_ylim(-1.0, 3.0)
            a.set_ylabel(f"{p['jmeno']}\n[N m/kg]", fontsize=8)
        else:
            a.fill_between(p["t"], 0, p["y"], step="mid", color="#1baf7a", alpha=0.35, lw=0, label="štítek ON")
            a.plot(p["t"], out, color="#eb6834", lw=1.0, label="P(ON) sítě")
            a.set_ylim(-0.05, 1.05)
            a.set_ylabel(p["jmeno"], fontsize=8)
        a.axhline(0, color="#52514e", lw=0.5)
        for nm, t0, t1 in p["meta"]["useky"]:
            a.axvline(t0, color="#999", lw=0.6, ls=":")
            if a is axs[0, 0]:
                a.text((t0 + t1) / 2, 1.08, nm, ha="center", fontsize=8, color="#52514e",
                       transform=a.get_xaxis_transform())
    axs[0, 0].legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    axs[-1, 0].set_xlabel("Čas [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    files = sorted(glob.glob(os.path.join(HERE, "data", "*.csv")))
    persons = [load_person(f) for f in files]
    if not persons:
        raise SystemExit("Žádná data v neuronova_sit/data/ - nejdřív spusť test5/scripts/run_all.py --data")
    print(f"Typ řízení: {TYP}; záznamy: " + ", ".join(p["jmeno"] for p in persons))
    rng = np.random.default_rng(CFG["uceni"]["seed"])
    res = {"nastaveni": CFG, "leave_one_person_out": {}}
    os.makedirs(os.path.join(HERE, "vysledky"), exist_ok=True)
    groups = sorted({p["skupina"] for p in persons})
    if len(groups) > 1:
        preds = {}
        for g in groups:
            net, _ = train(*dataset([q for q in persons if q["skupina"] != g], rng), CFG["uceni"]["seed"])
            for p in persons:
                if p["skupina"] == g:
                    preds[p["jmeno"]] = predict(net, p)
                    sc = res["leave_one_person_out"][p["jmeno"]] = scores(p, preds[p["jmeno"]])
                    print(f"  test '{p['jmeno']}' (neviděná osoba {g}): {summary_line(sc)}")
        plot(persons, [preds[p["jmeno"]] for p in persons],
             os.path.join(HERE, "vysledky", f"graf_{TYP}_test_neviditelna_osoba.png"))
    W, y, w = dataset(persons, rng)
    net, hist = train(W, y, w, CFG["uceni"]["seed"])
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    save(net, OUT_NPZ)
    preds = [predict(net, p) for p in persons]
    res["vsechny_osoby_trenink"] = {p["jmeno"]: scores(p, o) for p, o in zip(persons, preds)}
    res.update(parametru=int(sum(q.numel() for q in net.parameters())), oken=int(len(y)), ztrata=hist)
    plot(persons, preds, os.path.join(HERE, "vysledky", f"graf_{TYP}_konecna_sit.png"))
    json.dump(res, open(os.path.join(HERE, "vysledky", f"metriky_{TYP}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"Konečná síť ({res['parametru']} parametrů, {len(y)} oken) -> {os.path.relpath(OUT_NPZ, HERE)}")


if __name__ == "__main__":
    main()
