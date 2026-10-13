"""Grafy a čísla pro PDF zprávu (Python 3.11: numpy + matplotlib, bez OpenSim).

Předpoklad: run_exo.py, cnn_train.py, run_moje_rizeni.py cnn_rizeni.
Výstupy: zprava/obrazky/*.pdf|png, zprava/cisla.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "zprava", "obrazky")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C = {"bez": "#52514e", "pas": "#9a9893", "uhel_smer": "#2a78d6", "cnn": "#eb6834", "ideal": "#1baf7a"}
LBL = {"bez": "bez exo", "pas": "pasivní exo", "uhel_smer": "řízení úhel + směr",
       "cnn": "řízení CNN (ON/OFF)", "ideal": "ideální ON/OFF"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "lines.linewidth": 1.8})
M_ON, T_MOTOR = 40.0, 0.02


def cycles(meta):
    Tc = meta["Tc"]
    tk = {int(k): v for k, v in meta["t_k"].items()}
    return [(i * Tc, (i + 1) * Tc) for i in range(1, 7)], [(tk[k], tk[k + 2]) for k in range(4, 15, 2)]


def in_cycles(t, cyc):
    return np.any([(t >= a) & (t < b) for a, b in cyc], axis=0)


def normalize(t, y, cyc, n=101):
    return np.array([np.interp(np.linspace(a, b, n), t, y) for a, b in cyc])


def per_cycle(t, y, cyc, fn):
    return float(np.mean([fn(t[(t >= a) & (t < b)], y[(t >= a) & (t < b)]) for a, b in cyc]))


def motor(t, req):
    a = 1 - np.exp(-(t[1] - t[0]) / T_MOTOR)
    y, out = 0.0, np.zeros_like(req)
    for i, r in enumerate(req):
        y += a * (r - y)
        out[i] = y
    return out


def write_macros(num):
    """Čísla pro text zprávy jako makra LaTeXu (zprava/cisla.tex)."""
    K = {"bez": "Bez", "pas": "Pas", "uhel_smer": "Uhel", "cnn": "Cnn", "ideal": "Ideal"}
    Mx = {"spicka": "Spicka", "flex_spicka": "Flex", "rms": "Rms", "impuls_ext": "Impuls",
          "prace_plus": "Prace", "exo_prace": "ExoPrace", "exo_vykon_max": "ExoVykon"}
    out = []

    def mac(name, val):
        out.append("\\newcommand{\\" + name + "}{" + val + "}")
    for sec in ("rovina", "schody"):
        S_ = sec.capitalize()
        for k, kn in K.items():
            for m, mn in Mx.items():
                v = num[f"{sec}_{k}"][m]
                mac(f"{S_}{kn}{mn}", f"{v:.1f}".replace(".", "{,}").replace("-", "$-$"))
                b = num[f"{sec}_bez"][m]
                if m in ("spicka", "rms", "impuls_ext", "prace_plus") and abs(b) > 1e-9:
                    mac(f"{S_}{kn}{mn}Pct", f"{(v - b) / b * 100:.0f}".replace("-", "$-$"))
                    mac(f"{S_}{kn}{mn}Red", f"{(b - v) / b * 100:.0f}")    # snížení v %
    c = num["cnn"]
    for key, nm in (("cista_data", "Cista"), ("sum_2x", "Sum"), ("rychlost_0.85", "Pomala"),
                    ("rychlost_1.15", "Rychla")):
        for m, mn in (("F1", "F"), ("precision", "Prec"), ("recall", "Rec"), ("presnost", "Acc")):
            mac(f"Cnn{nm}{mn}", f"{c['test'][key][m]:.3f}".replace(".", "{,}"))
    mac("CnnParam", f"{c['parametru']}")
    mac("CnnOken", f"{c['trenovacich_oken']}")
    mac("CnnPodilOn", f"{c['podil_ON'] * 100:.0f}")
    mac("CnnEpochy", f"{c['epochy']}")
    mac("CnnZpozdeniOn", f"{np.mean(num['zpozdeni_zapnuti_ms']):.0f}")
    mac("CnnZpozdeniOff", f"{np.mean(num['zpozdeni_vypnuti_ms']):.0f}")
    mac("CnnFalesna", f"{num['falesna_zapnuti_rovina']}")
    open(os.path.join(ROOT, "zprava", "cisla.tex"), "w", encoding="utf-8").write(chr(10).join(out) + chr(10))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
    r = np.load(os.path.join(RES, "exo_vysledky.npz"))
    t, om = r["t"], r["omega"]
    cn = np.load(os.path.join(RES, "cnn", "cnn_trenink.npz"))
    met = json.load(open(os.path.join(RES, "cnn", "cnn_metriky.json")))
    lines = open(os.path.join(RES, "moje_rizeni", "cnn_rizeni", "prubeh.csv"), encoding="utf-8").read().splitlines()
    head = lines[0].split(";")
    D = np.array([[float(x) for x in l.split(";")[1:]] for l in lines[1:]])
    col = {h: D[:, i - 1] for i, h in enumerate(head) if i > 0}
    tau = {"bez": np.zeros_like(t), "pas": np.zeros_like(t), "uhel_smer": r["tau_uhel_smer"],
           "cnn": col["moment_motoru_Nm"], "ideal": motor(t, M_ON * cn["y"])}
    H = {k: (r["M_bez"] if k == "bez" else r["M_pas"] - v) for k, v in tau.items()}
    lev, sta = cycles(meta)
    keys = ["bez", "pas", "uhel_smer", "cnn", "ideal"]

    # ---------------- čísla
    num = {}
    for sec, cyc in (("rovina", lev), ("schody", sta)):
        m = in_cycles(t, cyc)
        for k in keys:
            h = H[k]
            num[f"{sec}_{k}"] = {
                "spicka": per_cycle(t, h, cyc, lambda a, b: b.max()),
                "flex_spicka": per_cycle(t, h, cyc, lambda a, b: b.min()),
                "rms": float(np.sqrt(np.mean(h[m] ** 2))),
                "impuls_ext": per_cycle(t, h, cyc, lambda a, b: np.sum(np.clip(b, 0, None)) * (a[1] - a[0])),
                "prace_plus": per_cycle(t, h * om, cyc, lambda a, b: np.sum(np.clip(b, 0, None)) * (a[1] - a[0])),
                "exo_prace": per_cycle(t, tau[k] * om, cyc, lambda a, b: np.sum(b) * (a[1] - a[0])),
                "exo_vykon_max": per_cycle(t, tau[k] * om, cyc, lambda a, b: b.max()),
                "on_podil": float((tau[k][m] > 1.0).mean())}
    num["cnn"] = met
    on = col["motor_zapnut"] > 0.5
    lab = cn["y"] > 0.5
    d_on = np.where(np.diff(on.astype(int)) == 1)[0]
    d_lab = np.where(np.diff(lab.astype(int)) == 1)[0]
    d_off = np.where(np.diff(on.astype(int)) == -1)[0]
    d_laboff = np.where(np.diff(lab.astype(int)) == -1)[0]
    delays_on = [t[min(d_on, key=lambda j: abs(j - i))] - t[i] for i in d_lab]
    delays_off = [t[min(d_off, key=lambda j: abs(j - i))] - t[i] for i in d_laboff]
    num["zpozdeni_zapnuti_ms"] = [float(x * 1000) for x in delays_on]
    num["zpozdeni_vypnuti_ms"] = [float(x * 1000) for x in delays_off]
    num["falesna_zapnuti_rovina"] = int(((np.diff(on.astype(int)) == 1) & (t[1:] < 7 * meta["Tc"])).sum())
    json.dump(num, open(os.path.join(ROOT, "zprava", "cisla.json"), "w"), indent=1)
    write_macros(num)

    # ---------------- obr: odlehčení kolene na schodech (plochy)
    x = np.linspace(0, 100, 101)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, k in zip(axs, ("cnn", "uhel_smer")):
        Mb = normalize(t, r["M_pas"], sta).mean(0)
        Mh = normalize(t, H[k], sta).mean(0)
        a.fill_between(x, Mh, Mb, where=Mb > Mh, color=C[k], alpha=0.35, lw=0, label="dodá exoskelet")
        a.plot(x, Mb, color=C["bez"], label="potřebný moment (bez pomoci)")
        a.plot(x, Mh, color=C[k], label="zbývá na člověka")
        a.axhline(0, color=INK2, lw=0.6)
        a.set_title(LBL[k], color=INK)
        a.set_xlabel("Krokový cyklus pravé nohy na schodech [%]")
        a.text(61, -62, "stoj", color=INK2, fontsize=8, ha="right")
        a.text(65, -62, "švih", color=INK2, fontsize=8, ha="left")
        a.axvline(63, color=INK2, lw=0.6, ls=":")
    axs[0].set_ylabel("Extenzní moment kolene [Nm]")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, frameon=False, fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "odlehceni_schody")

    # ---------------- obr: souhrnné sloupce
    metrics = [("spicka", "Špička extenzního\nmomentu [Nm]"), ("rms", "RMS momentu\n[Nm]"),
               ("impuls_ext", "Extenzní impuls\n[Nm·s / krok]"), ("prace_plus", "Kladná práce kolene\n[J / krok]")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.4))
    for a, (mk, ml) in zip(axs, metrics):
        vals = [num[f"schody_{k}"][mk] for k in keys]
        b = a.bar(range(len(keys)), vals, color=[C[k] for k in keys], width=0.7)
        base = vals[0]
        for j, (bb, v) in enumerate(zip(b, vals)):
            txt = f"{v:.0f}" if j == 0 else f"{v:.0f}\n({(v - base) / base * 100:+.0f} %)"
            a.text(bb.get_x() + bb.get_width() / 2, v + base * 0.02, txt, ha="center", va="bottom",
                   fontsize=7, color=INK)
        a.set_xticks(range(len(keys)))
        a.set_xticklabels(["bez\nexo", "pasivní\nexo", "úhel +\nsměr", "CNN\nON/OFF", "ideální\nON/OFF"],
                          fontsize=7)
        a.set_title(ml, fontsize=9, color=INK)
        a.set_ylim(0, base * 1.3)
        a.grid(axis="x", visible=False)
    fig.suptitle("Odlehčení pravého kolene při výstupu do schodů (průměr 6 kroků)", color=INK)
    fig.tight_layout()
    save(fig, "odlehceni_sloupce")

    # ---------------- obr: kumulativní práce člověka v koleni
    fig, ax = plt.subplots(figsize=(10, 3.3))
    for k in ("bez", "uhel_smer", "cnn", "ideal"):
        w = np.cumsum(np.clip(H[k] * om, 0, None)) * (t[1] - t[0])
        ax.plot(t, w, color=C[k], label=LBL[k])
    ax.axvspan(7 * meta["Tc"], meta["t_k"]["2"], color="#f0efec", lw=0, zorder=0)
    ax.axvspan(meta["t_k"]["2"], t[-1], color="#fbe3d9", lw=0, zorder=0)
    ax.text(3.5, 0.95, "rovina", transform=ax.get_xaxis_transform(), color=INK2, ha="center")
    ax.text(15, 0.95, "schody", transform=ax.get_xaxis_transform(), color=INK2, ha="center")
    ax.set_xlabel("Čas [s]")
    ax.set_ylabel("Kumulovaná kladná práce\nsvalů kolene [J]")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout()
    save(fig, "kumulativni_prace")

    # ---------------- obr: rovina – CNN se nezapíná
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for a, (cyc, nm) in zip(axs, ((lev, "Rovina"), (sta, "Schody"))):
        for k in ("bez", "uhel_smer", "cnn"):
            a.plot(x, normalize(t, H[k], cyc).mean(0), color=C[k], label=LBL[k],
                   lw=1.4 if k == "bez" else 1.8)
        a.axhline(0, color=INK2, lw=0.6)
        a.set_title(nm, color=INK)
        a.set_xlabel("Krokový cyklus pravé nohy [%]")
    axs[0].set_ylabel("Moment kolene člověka [Nm]\n(+ extenzní)")
    axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "rovina_vs_schody")

    # ---------------- obr: CNN – učení a test
    hist = cn["hist"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(np.arange(1, len(hist) + 1), hist[:, 0], color=C["uhel_smer"], label="trénink (augmentovaná data)")
    ax.plot(np.arange(1, len(hist) + 1), hist[:, 1], color=C["cnn"], label="validace (čistá data)")
    ax.set_yscale("log")
    ax.set_xlabel("Epocha")
    ax.set_ylabel("Ztráta (vážená BCE)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "cnn_uceni")

    idx, p = cn["idx20"], cn["p20"]
    t20 = t[idx]
    fig, axs = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True)
    X = cn["X"]
    axs[0].plot(t, X[:, 0], color=INK, lw=1.2)
    axs[0].set_ylabel("Flexe kolene [°]")
    axs[1].plot(t, X[:, 1], color=C["uhel_smer"], lw=1.2, label="stehno")
    axs[1].plot(t, X[:, 2], color=INK, lw=1.2, label="holeň")
    axs[1].set_ylabel("Úhlová rychlost\n[°/s]")
    axs[1].legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9, edgecolor="none")
    axs[2].fill_between(t, 0, cn["y"], step="mid", color=C["ideal"], alpha=0.35, lw=0,
                        label="štítek (moment > 40 Nm na schodu)")
    axs[2].step(t20, p, where="post", color=C["cnn"], lw=1.4, label="výstup CNN P(ON), 20 Hz")
    axs[2].axhline(0.5, color=INK2, lw=0.6, ls="--")
    axs[2].set_ylabel("Motor ON")
    axs[2].set_xlabel("Čas [s]")
    axs[2].legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9, edgecolor="none")
    for a in axs:
        for (a0, a1) in met["testovaci_useky_s"]:
            a.axvspan(a0, a1, color="#f0efec", lw=0, zorder=0)
    axs[0].text(np.mean(met["testovaci_useky_s"][0]), 1.02, "test", transform=axs[0].get_xaxis_transform(),
                ha="center", color=INK2)
    axs[0].text(np.mean(met["testovaci_useky_s"][1]), 1.02, "test", transform=axs[0].get_xaxis_transform(),
                ha="center", color=INK2)
    fig.tight_layout()
    save(fig, "cnn_prubeh")

    # detail jednoho kroku na schodech
    a0, a1 = sta[-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    sel = (t >= a0 - 0.2) & (t <= a1)
    s20 = (t20 >= a0 - 0.2) & (t20 <= a1)
    ax.plot(t[sel], r["M_bez"][sel], color=C["bez"], label="moment kolene bez exo")
    ax.axhline(40, color=INK2, lw=0.6, ls="--")
    ax.fill_between(t[sel], 0, 1, where=cn["y"][sel] > 0.5, color=C["ideal"], alpha=0.3, lw=0,
                    transform=ax.get_xaxis_transform(), label="štítek ON")
    ax.plot(t[sel], tau["cnn"][sel], color=C["cnn"], label="moment motoru (CNN)")
    ax.plot(t20[s20], p[s20] * 40, "o", ms=3, color=C["cnn"], alpha=0.6, label="P(ON)·40 (vzorky 20 Hz)")
    ax.set_xlabel("Čas [s]")
    ax.set_ylabel("Moment [Nm]")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "cnn_detail_kroku")
    print(json.dumps({k: v for k, v in num.items() if k.startswith("schody")}, indent=1)[:1500])
    print("zpoždění zapnutí [ms]:", np.round(num["zpozdeni_zapnuti_ms"], 0),
          "vypnutí:", np.round(num["zpozdeni_vypnuti_ms"], 0), "falešná zapnutí rovina:",
          num["falesna_zapnuti_rovina"])


if __name__ == "__main__":
    main()
