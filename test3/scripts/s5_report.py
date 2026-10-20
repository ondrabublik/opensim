"""Krok 5: vyhodnocení - grafy, souhrn a srovnání s test2 (výsledky jsou z OpenSim API)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from t3common import *  # noqa: F401,F403,E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"bez": "#52514e", "pas": "#9a9893", "uhel_smer": "#2a78d6", "cnn": "#eb6834"}
LBL = {"bez": "bez exo", "pas": "pasivní exo", "uhel_smer": "úhel + směr", "cnn": "CNN ON/OFF"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 1.8})


def per_cycle(t, y, cyc, fn):
    return float(np.mean([fn(t[(t >= a) & (t < b)], y[(t >= a) & (t < b)]) for a, b in cyc]))


def norm(t, y, cyc, n=101):
    return np.array([np.interp(np.linspace(a, b, n), t, y) for a, b in cyc])


def pos(a, b):
    return float(np.sum(np.clip(b, 0, None)) * (a[1] - a[0]))


def main():
    meta = json.load(open(META))
    r = dict(np.load(os.path.join(RES, "test3_vysledky.npz")))
    t, om = r["t"], r["omega"]
    lev, sta = cycles(meta)
    H = {"bez": r["M_bez"], "pas": r["M_pas"], "uhel_smer": r["M_uhel_smer"], "cnn": r["M_cnn"]}
    tau = {"bez": 0 * t, "pas": 0 * t, "uhel_smer": r["tau_uhel_smer"], "cnn": r["tau_cnn"]}
    P = {"bez": 0 * t, "pas": 0 * t, "uhel_smer": r["P_uhel_smer"], "cnn": r["P_cnn"]}

    # ---------------- souhrn
    rows = []
    for sec, cyc in (("Rovina", lev), ("Schody", sta)):
        m = in_cycles(t, cyc)
        for k in H:
            rows.append((sec, LBL[k], per_cycle(t, H[k], cyc, lambda a, b: b.max()),
                         per_cycle(t, H[k], cyc, lambda a, b: b.min()), np.sqrt(np.mean(H[k][m] ** 2)),
                         per_cycle(t, H[k], cyc, pos), per_cycle(t, H[k] * om, cyc, pos),
                         per_cycle(t, tau[k], cyc, lambda a, b: b.max()),
                         per_cycle(t, P[k], cyc, lambda a, b: b.max()),
                         per_cycle(t, P[k], cyc, lambda a, b: np.sum(b) * (a[1] - a[0]))))
    r2 = np.load(os.path.join(TEST2, "results", "exo_vysledky.npz"))
    l2 = open(os.path.join(TEST2, "results", "moje_rizeni", "cnn_rizeni", "prubeh.csv"),
              encoding="utf-8").read().splitlines()
    h2 = l2[0].split(";")
    D2 = np.array([[float(x) for x in l.split(";")[1:]] for l in l2[1:]])
    ref = {"bez": r2["M_bez"], "pas": r2["M_pas"], "uhel_smer": r2["M_uhel_smer"],
           "cnn": D2[:, h2.index("moment_kolene_clovek_Nm") - 1]}
    n = min(len(t), len(D2))
    with open(os.path.join(RES, "souhrn_test3.md"), "w", encoding="utf-8") as f:
        f.write("# test3 – exoskelet kolene přes OpenSim API: souhrn\n\n"
                "Moment člověka = výstup InverseDynamicsTool s motorem v modelu; moment a výkon motoru z "
                "AnalyzeTool (ForceReporter, Kinematics). Průměr přes 6 cyklů, + = extenzní / generace.\n\n"
                "| Úsek | Varianta | Ext. špička [Nm] | Flex. špička [Nm] | RMS [Nm] | Ext. impuls [Nm·s] | "
                "Práce člověka + [J] | Motor max [Nm] | Výkon motoru max [W] | Práce motoru [J] |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n")
        for rw in rows:
            f.write(f"| {rw[0]} | {rw[1]} | " + " | ".join(f"{v:.1f}" for v in rw[2:]) + " |\n")
        f.write("\n## Shoda s test2 (moment kolene člověka, celých 15 m)\n\n| Varianta | max rozdíl [Nm] |\n|---|---|\n")
        for k in H:
            f.write(f"| {LBL[k]} | {np.abs(H[k][:n] - ref[k][:n]).max():.4f} |\n")
    print(open(os.path.join(RES, "souhrn_test3.md"), encoding="utf-8").read())

    # ---------------- graf: cyklus
    x = np.linspace(0, 100, 101)
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for c, (cyc, nm) in enumerate(((lev, "Rovina"), (sta, "Schody"))):
        for k in H:
            axs[0, c].plot(x, norm(t, H[k], cyc).mean(0), color=C[k], label=LBL[k],
                           ls="--" if k == "pas" else "-", lw=1.3 if k in ("bez", "pas") else 1.8)
        for k in ("uhel_smer", "cnn"):
            axs[1, c].plot(x, norm(t, tau[k], cyc).mean(0), color=C[k], label=LBL[k])
            axs[2, c].plot(x, norm(t, P[k], cyc).mean(0), color=C[k], label=LBL[k])
        axs[0, c].set_title(nm, color=INK)
        axs[2, c].set_xlabel("Krokový cyklus pravé nohy [%]")
        for a in axs[:, c]:
            a.axhline(0, color=INK2, lw=0.6)
    axs[0, 0].set_ylabel("Moment kolene člověka [Nm]\n(InverseDynamicsTool)")
    axs[1, 0].set_ylabel("Moment motoru [Nm]\n(ForceReporter)")
    axs[2, 0].set_ylabel("Výkon motoru [W]\n(ForceReporter × Kinematics)")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    axs[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("test3: výsledky přímo z OpenSim API (průměr 6 cyklů)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf_t3_cyklus.png"), dpi=150)
    plt.close(fig)

    # ---------------- graf: snímače z BodyKinematics vs. numerická derivace (test2)
    ts, S = table_np(os.path.join(RES, "cnn_rizeni.sto"))
    m = (ts > 12.0) & (ts < 14.6)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for a, (key, nm) in zip(axs, (("omega_stehno_deg_s", "stehno"), ("omega_holen_deg_s", "holeň"))):
        a.plot(ts[m], S[key][m], color=C["cnn"], label="OpenSim BodyKinematics (test3)")
        a.plot(ts[m], D2[:len(ts), h2.index(key) - 1][m], color=INK, lw=1.0, ls="--",
               label="numerická derivace úhlů (test2)")
        a.set_ylabel(f"ω {nm} [°/s]")
        a.fill_between(ts[m], 0, 1, where=S["motor_zapnut"][m] > 0.5, color="#f6d3c4", lw=0,
                       transform=a.get_xaxis_transform(), zorder=0)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    axs[1].set_xlabel("Čas [s] (podbarvení = motor zapnut)")
    fig.suptitle("Snímače pro CNN: úhlové rychlosti segmentů na schodech", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf_t3_snimace.png"), dpi=150)
    plt.close(fig)
    print("Grafy:", RES)


if __name__ == "__main__":
    main()
