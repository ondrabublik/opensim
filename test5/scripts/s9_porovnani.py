"""Krok 9: porovnání osob (výsledky z results/<jmeno>/): úhel kolene a moment kolene [N m/kg]
s vypnutým / zapnutým exoskeletem v průměrném kroku na schodech nahoru a dolů a na rovině.

    python s9_porovnani.py [jmeno ...]       (bez argumentu všechny osoby v results/)
Výstup: results/porovnani_osob.png, results/porovnani_osob.md
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 1.6})


def read_sto(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = lines[i + 1].split("\t")
    d = np.array([[float(x) for x in l.split()] for l in lines[i + 2:] if l.strip()])
    return {c: d[:, j] for j, c in enumerate(cols)}


def cycles(meta, phase):
    """Krokové cykly pravé nohy (dopad -> další dopad) v daném úseku."""
    on = [c["t_on"] for c in meta["contacts"]["r"] if c["t_on"] is not None and c["phase"] == phase]
    return list(zip(on[:-1], on[1:]))[1:-1] or list(zip(on[:-1], on[1:]))


def norm(t, y, cyc, n=101):
    return np.array([np.interp(np.linspace(a, b, n), t, y) for a, b in cyc])


VAR = [("pomala", "pomalá", "#2a78d6"), ("", "normální", "#1baf7a"), ("rychla", "rychlá", "#eb6834")]


def level_speed(meta):
    st = [s for s in meta["steps"] if s["phase"] == "rovina"][1:]
    if len(st) < 3:
        return float("nan")
    return (st[-1]["heel"] - st[0]["heel"]) / (st[-1]["t_on"] - st[0]["t_on"])


def speed_figure(people, phases):
    """Mřížka úsek x osoba: moment kolene [N m/kg] pro pomalou / normální / rychlou chůzi."""
    groups = sorted({p["info"].get("skupina", p["nm"]) for p in people},
                    key=lambda g: np.mean([q["info"]["vyska_m"] for q in people if q["info"].get("skupina", q["nm"]) == g]))
    x = np.linspace(0, 100, 101)
    fig, axs = plt.subplots(len(phases), len(groups), figsize=(4.6 * len(groups), 3.0 * len(phases)),
                            sharex=True, sharey="row", squeeze=False)
    rows = []
    for c, g in enumerate(groups):
        for suf, lab, col in VAR:
            nm = g if not suf else f"{g}_{suf}"
            p = next((q for q in people if q["nm"] == nm), None)
            if p is None:
                continue
            v = level_speed(p["meta"])
            for r, (ph_lab, ph) in enumerate(phases):
                cyc = cycles(p["meta"], ph)
                if not cyc:
                    continue
                off = norm(p["t"], p["off"], cyc).mean(0)
                on = norm(p["t"], p["on"], cyc).mean(0)
                axs[r, c].plot(x, off, color=col, lw=1.0, ls="--")
                axs[r, c].plot(x, on, color=col, lw=1.6, label=f"{lab} ({v:.2f} m/s)" if r == 0 else None)
                sel = np.any([(p["t"] >= a) & (p["t"] < b) for a, b in cyc], axis=0)
                rows.append((g, lab, v, ph_lab, np.mean([b - a for a, b in cyc]), off.max(), on.max(),
                             p["ex"][sel].mean() * 100))
        axs[0, c].set_title(f"{g} ({people[[q['nm'] for q in people].index(g)]['info']['vyska_m']:.2f} m)"
                            if g in [q["nm"] for q in people] else g, color=INK)
        axs[0, c].legend(frameon=False, fontsize=8, loc="upper right")
        axs[-1, c].set_xlabel("Krokový cyklus pravé nohy [%]")
    for r, (ph_lab, _) in enumerate(phases):
        axs[r, 0].set_ylabel(f"{ph_lab}\nmoment kolene [N m/kg]")
        for a in axs[r]:
            a.axhline(0, color=INK2, lw=0.5)
    fig.suptitle("Rychlost chůze: moment kolene člověka (čárkovaně exo vypnutý, plně zapnutý)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "porovnani_rychlosti.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(RES, "porovnani_rychlosti.md"), "w", encoding="utf-8") as f:
        f.write("# Rychlost chůze – průměrný krok pravé nohy\n\n| Osoba | Varianta | Rychlost na rovině [m/s] | Úsek | "
                "Doba kroku [s] | Max. moment vyp. [N m/kg] | Max. moment zap. [N m/kg] | Motor ON [%] |\n"
                "|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.0f} |\n")
    print(open(os.path.join(RES, "porovnani_rychlosti.md"), encoding="utf-8").read())


def main():
    names = sys.argv[1:] or sorted(d for d in os.listdir(RES) if os.path.isdir(os.path.join(RES, d)))
    people = []
    for nm in names:
        r = os.path.join(RES, nm)
        if not os.path.exists(os.path.join(r, "ID_exo_zapnuty.sto")):
            continue
        meta = json.load(open(os.path.join(ROOT, "data", nm, "meta.json"), encoding="utf-8"))
        info = json.load(open(os.path.join(ROOT, "neuronova_sit", "data", f"{nm}.json"), encoding="utf-8"))
        c = read_sto(os.path.join(r, "cnn_rizeni.sto"))
        off = read_sto(os.path.join(r, "ID_exo_vypnuty.sto"))
        on = read_sto(os.path.join(r, "ID_exo_zapnuty.sto"))
        n = min(len(c["time"]), len(off["time"]))
        people.append(dict(nm=nm, info=info, meta=meta, t=c["time"][:n], knee=c["koleno_flexe_deg"][:n],
                           off=off["knee_angle_r_moment"][:n] / info["hmotnost_s_exo_kg"],
                           on=on["knee_angle_r_moment"][:n] / info["hmotnost_s_exo_kg"],
                           ex=c["motor_zapnut"][:n]))
    phases = [("rovina", "rovina"), ("schody nahoru", "schody nahoru"), ("schody dolů", "schody dolu")]
    x = np.linspace(0, 100, 101)
    all_people = people
    people = [p for p in all_people if p["info"].get("skupina", p["nm"]) == p["nm"]]     # normální rychlost
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    rows = []
    for c_, (lab, ph) in enumerate(phases):
        for k, p in enumerate(people):
            cyc = cycles(p["meta"], ph)
            if not cyc:
                continue
            col = COL[k % len(COL)]
            lbl = f"{p['nm']} ({p['info']['vyska_m']:.2f} m, {p['info']['hmotnost_kg']:.0f} kg)"
            axs[0, c_].plot(x, norm(p["t"], p["knee"], cyc).mean(0), color=col, label=lbl)
            axs[1, c_].plot(x, norm(p["t"], p["off"], cyc).mean(0), color=col, lw=1.1, ls="--")
            axs[1, c_].plot(x, norm(p["t"], p["on"], cyc).mean(0), color=col)
            T = np.mean([b - a for a, b in cyc])
            sel = np.any([(p["t"] >= a) & (p["t"] < b) for a, b in cyc], axis=0)
            rows.append((lab, p["nm"], T, norm(p["t"], p["knee"], cyc).mean(0).max(),
                         norm(p["t"], p["knee"], cyc).mean(0).min(), norm(p["t"], p["off"], cyc).mean(0).max(),
                         norm(p["t"], p["on"], cyc).mean(0).max(), p["ex"][sel].mean() * 100))
        axs[0, c_].set_title(lab, color=INK)
        axs[1, c_].set_xlabel("Krokový cyklus pravé nohy [%]")
        axs[1, c_].axhline(0, color=INK2, lw=0.5)
    axs[0, 0].set_ylabel("Flexe kolene [°]")
    axs[1, 0].set_ylabel("Moment kolene člověka [N m/kg]\n(čárkovaně vypnutý, plně zapnutý exo)")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Porovnání osob: průměrný krok (exoskelet řízený sítí z test5/neuronova_sit)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "porovnani_osob.png"), dpi=150)
    plt.close(fig)
    speed_figure(all_people, phases)
    with open(os.path.join(RES, "porovnani_osob.md"), "w", encoding="utf-8") as f:
        f.write("# Porovnání osob – průměrný krok pravé nohy\n\n| Úsek | Osoba | Doba kroku [s] | Flexe kolene max / min [°] | "
                "Max. moment vyp. [N m/kg] | Max. moment zap. [N m/kg] | Motor ON [%] |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.0f} / {r[4]:.0f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.0f} |\n")
    print(open(os.path.join(RES, "porovnani_osob.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
