"""Krok 2: vyhodnocení dopředných simulací v uzavřené smyčce (grafy, souhrn, kontroly)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from t4common import *  # noqa: F401,F403,E402
from s1_simulace import SCENARE  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"bez_exo": "#52514e", "cnn_clovek_idealni": "#2a78d6", "cnn_clovek_100ms": "#eb6834",
     "cnn_clovek_nekompenzuje": "#4a3aa7", "cnn_sum_snimacu": "#1baf7a"}
LBL = {"bez_exo": "bez exo", "cnn_clovek_idealni": "CNN, člověk kompenzuje ihned",
       "cnn_clovek_100ms": "CNN, člověk kompenzuje za 100 ms",
       "cnn_clovek_nekompenzuje": "CNN, člověk nekompenzuje", "cnn_sum_snimacu": "CNN + šum snímačů (100 ms)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 1.6})
KM = "clovek_" + KNEE


def per_cycle(t, y, cyc, fn):
    return float(np.mean([fn(t[(t >= a) & (t < b)], y[(t >= a) & (t < b)]) for a, b in cyc]))


def norm(t, y, cyc, n=101):
    return np.array([np.interp(np.linspace(a, b, n), t, y) for a, b in cyc])


def pos_work(a, b):
    return float(np.sum(np.clip(b, 0, None)) * (a[1] - a[0]))


def main():
    meta = json.load(open(T3.META))
    lev, sta = cycles(meta)
    D = {k: table_np(os.path.join(RES4, f"sim_{k}.sto")) for k in SCENARE}
    t = D["bez_exo"][0]
    # reference z test3: ID bez exo (pasivní exo) a otevřená smyčka CNN
    ti, idp = table_np(os.path.join(T3.RES, "ID_exo_pasivni.sto"))
    _, idc = table_np(os.path.join(T3.RES, "ID_exo_cnn.sto"))
    t3, c3 = table_np(os.path.join(T3.RES, "cnn_rizeni.sto"))
    on_open = np.interp(t, t3, c3["motor_zapnut"]) > 0.5

    rows, checks = [], {}
    for k, (tk, d) in D.items():
        om = d["u_" + KNEE]
        err = np.degrees(d["chyba_" + KNEE])
        for sec, cyc in (("Rovina", lev), ("Schody", sta)):
            m = in_cycles(tk, cyc)
            rows.append(dict(
                sc=k, sec=sec,
                spicka=per_cycle(tk, d[KM], cyc, lambda a, b: b.max()),
                rms=float(np.sqrt(np.mean(d[KM][m] ** 2))),
                prace=per_cycle(tk, d[KM] * om, cyc, pos_work),
                chyba_rms=float(np.sqrt(np.mean(err[m] ** 2))), chyba_max=float(np.abs(err[m]).max()),
                on=float(d["exo_on"][m].mean() * 100),
                shoda=float(np.mean((d["exo_on"][m] > 0.5) == on_open[m]) * 100),
                exo_prace=per_cycle(tk, d["exo_moment"] * om, cyc, lambda a, b: float(np.sum(b) * (a[1] - a[0]))),
                rez=float(np.sqrt(np.mean(sum(d["clovek_" + p][m] ** 2 for p in ("pelvis_tx", "pelvis_ty")))))))
    # kontroly
    n = len(t)
    Mpas = np.interp(t, ti, idp[KNEE + "_moment"])
    Mcnn = np.interp(t, ti, idc[KNEE + "_moment"])
    for lab, e in (("bez exo vs ID (test3)", D["bez_exo"][1][KM] - Mpas),
                   ("ideální kompenzace vs ID s CNN (test3)", D["cnn_clovek_idealni"][1][KM] - Mcnn)):
        checks[lab + ": RMS [Nm]"] = float(np.sqrt(np.mean(e ** 2)))
        checks[lab + ": 99. percentil absolutního rozdílu [Nm]"] = float(np.percentile(np.abs(e), 99))
    checks["ideální kompenzace: shoda ON/OFF s otevřenou smyčkou [%]"] = float(
        np.mean((D["cnn_clovek_idealni"][1]["exo_on"] > 0.5) == on_open) * 100)

    with open(os.path.join(RES4, "souhrn_test4.md"), "w", encoding="utf-8") as f:
        f.write("# test4 – dopředná simulace v uzavřené smyčce s CNN: souhrn\n\n"
                "Průměr přes 6 cyklů pravé nohy. Moment kolene člověka = aktuátor `clovek_knee_angle_r` "
                "(regulátor s vypočteným momentem, InverseDynamicsSolver). Shoda ON/OFF = podíl času, kdy "
                "se rozhodnutí CNN v uzavřené smyčce shoduje s otevřenou smyčkou (test3).\n\n"
                "| Scénář | Úsek | Ext. špička kolene [Nm] | RMS [Nm] | Kladná práce člověka [J] | "
                "Odchylka kolene RMS / max [°] | Motor ON [%] | Shoda ON/OFF [%] | Práce motoru [J] | "
                "Reziduum pánve RMS [N] |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {LBL[r['sc']]} | {r['sec']} | {r['spicka']:.1f} | {r['rms']:.1f} | {r['prace']:.1f} | "
                    f"{r['chyba_rms']:.2f} / {r['chyba_max']:.2f} | {r['on']:.1f} | "
                    f"{'–' if r['sc'] == 'bez_exo' else format(r['shoda'], '.1f')} | "
                    f"{r['exo_prace']:.1f} | {r['rez']:.1f} |\n")
        f.write("\n## Kontroly\n\n| Kontrola | Hodnota |\n|---|---|\n")
        for k, v in checks.items():
            f.write(f"| {k} | {v:.2f} |\n")
        f.write("\nOjedinělé špičky rozdílu (až ~30 Nm po dobu 1-3 ms) vznikají v okamžiku odrazu chodidla, "
                "kdy reakční síla země skokově klesne: simulace běží po 1 ms, výsledek ID je po 5 ms.\n")
    json.dump({"rows": rows, "checks": checks}, open(os.path.join(RES4, "souhrn_test4.json"), "w"), indent=1)
    print(open(os.path.join(RES4, "souhrn_test4.md"), encoding="utf-8").read())

    # ---------------- graf 1: cyklus na schodech
    x = np.linspace(0, 100, 101)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for k, (tk, d) in D.items():
        axs[0, 0].plot(x, norm(tk, np.degrees(d["chyba_" + KNEE]), sta).mean(0), color=C[k], label=LBL[k])
        axs[0, 1].plot(x, norm(tk, d[KM], sta).mean(0), color=C[k], label=LBL[k])
        axs[1, 0].plot(x, norm(tk, d["exo_moment"], sta).mean(0), color=C[k], label=LBL[k])
        axs[1, 1].plot(x, norm(tk, d[KM] * d["u_" + KNEE], sta).mean(0), color=C[k], label=LBL[k])
    axs[0, 0].set_ylabel("Odchylka flexe kolene od reference [°]\n(+ = méně pokrčené)")
    axs[0, 1].set_ylabel("Moment kolene člověka [Nm]\n(+ extenzní)")
    axs[1, 0].set_ylabel("Moment motoru exo [Nm]")
    axs[1, 1].set_ylabel("Výkon svalů kolene [W]")
    for a in axs.flat:
        a.axhline(0, color=INK2, lw=0.6)
    for a in axs[1]:
        a.set_xlabel("Krokový cyklus pravé nohy na schodech [%]")
    axs[0, 1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("test4: dopředná simulace v uzavřené smyčce – výstup do schodů (průměr 6 kroků)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES4, "graf_t4_cyklus.png"), dpi=150)
    plt.close(fig)

    # ---------------- graf 2: detail jednoho kroku, uzavřená vs otevřená smyčka
    a0, a1 = sta[2][0] - 0.15, sta[2][0] + 0.75
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for k in ("cnn_clovek_idealni", "cnn_clovek_100ms", "cnn_clovek_nekompenzuje"):
        tk, d = D[k]
        sel = (tk >= a0) & (tk <= a1)
        axs[0].plot(tk[sel], d["snimac_koleno_deg"][sel], color=C[k], label=LBL[k], drawstyle="steps-post")
        axs[1].plot(tk[sel], d["exo_p_on"][sel], color=C[k], drawstyle="steps-post")
        axs[2].plot(tk[sel], d[KM][sel], color=C[k])
    sel3 = (t3 >= a0) & (t3 <= a1)
    axs[0].plot(t3[sel3], c3["koleno_flexe_deg"][sel3], color=INK, lw=1, ls="--", label="reference (otevřená smyčka)")
    axs[1].plot(t3[sel3], c3["p_on"][sel3], color=INK, lw=1, ls="--")
    sel = (t >= a0) & (t <= a1)
    axs[2].plot(t[sel], D["bez_exo"][1][KM][sel], color=C["bez_exo"], lw=1.2, label="bez exo")
    axs[0].set_ylabel("Flexe kolene, vstup CNN [°]")
    axs[1].set_ylabel("Výstup CNN P(ON)")
    axs[2].set_ylabel("Moment kolene člověka [Nm]")
    axs[2].set_xlabel("Čas [s]")
    axs[0].legend(frameon=False, fontsize=8, loc="upper right")
    axs[2].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Jeden krok na schodech: síť vidí pohyb, který sama ovlivnila", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES4, "graf_t4_detail_kroku.png"), dpi=150)
    plt.close(fig)
    print("Grafy:", RES4)


if __name__ == "__main__":
    main()
