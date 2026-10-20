"""Krok 5: grafy a souhrn – úhel kolene, úhlové rychlosti stehna a holeně, moment kolene
s vypnutým / zapnutým exoskeletem (CNN) pro celý scénář i detaily úseků."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from t5common import *  # noqa: F401,F403,E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_OFF, C_ON, C_TH, C_SH, C_MOT = "#52514e", "#eb6834", "#2a78d6", "#0b0b0b", "#eb6834"
SHADE = ["#f3f1ec", "#e3eefa", "#fbe3d9", "#efe9f7", "#e2f4ec", "#f3f1ec"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 1.4})
K = KNEE + "_moment"


def load():
    t, c = table_np(os.path.join(RES, "cnn_rizeni.sto"))
    _, off = table_np(os.path.join(RES, "ID_exo_vypnuty.sto"))
    _, on = table_np(os.path.join(RES, "ID_exo_zapnuty.sto"))
    n = min(len(t), len(off[K]), len(on[K]))
    d = {k: v[:n] for k, v in c.items()}
    d["M_off"], d["M_on"] = off[K][:n], on[K][:n]
    d["omega_kolene"] = np.gradient(np.radians(-d["koleno_flexe_deg"]), t[:n])   # rad/s, + = extenze
    # neomezený motor (jen regrese): moment = podíl * výstup sítě * hmotnost -> co by zbylo člověku
    d["M_neomez"] = None
    exo = OSOBA.get("exoskelet", {}) or {}
    if "vystup_site" in d and str(exo.get("vahy_site", "")).endswith("regrese.npz"):
        tau_u = float(exo.get("podil_kompenzace", 1.0)) * d["vystup_site"] * (OSOBA["hmotnost_kg"] + EXO_MASS)
        a = 1 - np.exp(-(t[1] - t[0]) / 0.02)
        y, lag = 0.0, np.zeros_like(tau_u)
        for i, v in enumerate(tau_u):
            y += a * (v - y)
            lag[i] = y
        d["M_neomez"] = d["M_off"] - lag
    return t[:n], d


def shade(ax, bounds, labels=False):
    for k, (nm, a, b) in enumerate(bounds):
        ax.axvspan(a, b, color=SHADE[k % len(SHADE)], lw=0, zorder=0)
        if labels:
            ax.text((a + b) / 2, 1.02, nm, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    color=INK2, fontsize=8)


def main():
    m = meta()
    t, d = load()
    bounds = phase_bounds(m)
    mass = OSOBA["hmotnost_kg"] + EXO_MASS

    # ---------------- graf 1: celý scénář
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axs[0].plot(t, d["koleno_flexe_deg"], color=INK)
    axs[0].set_ylabel("Flexe pravého\nkolene [°]")
    axs[1].plot(t, d["omega_stehno_deg_s"], color=C_TH, label="stehno")
    axs[1].plot(t, d["omega_holen_deg_s"], color=C_SH, lw=1.1, label="holeň")
    axs[1].set_ylabel("Úhlová rychlost\nvůči zemi [°/s]\n(+ dopředu)")
    axs[1].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    axs[2].plot(t, d["M_off"], color=C_OFF, lw=1.1, label="exoskelet vypnutý")
    axs[2].plot(t, d["M_on"], color=C_ON, lw=1.4, label="exoskelet zapnutý (řízení sítí)")
    axs[2].set_ylabel("Moment kolene\nčlověka [N m]\n(+ extenzní)")
    axs[2].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    axs[3].plot(t, d["exo_motor_Nm"], color=C_MOT)
    axs[3].fill_between(t, 0, 1, where=d["motor_zapnut"] > 0.5, color=C_ON, alpha=0.18, lw=0,
                        transform=axs[3].get_xaxis_transform())
    axs[3].set_ylabel("Moment motoru\nexo [N m]")
    axs[3].set_xlabel("Čas [s]")
    for k, a in enumerate(axs):
        shade(a, bounds, labels=(k == 0))
        a.axhline(0, color=INK2, lw=0.5)
    fig.suptitle(f"Osoba '{JMENO}' ({OSOBA['vyska_m']:.2f} m, {mass:.1f} kg s exo): vstávání → rovina → "
                 "schody nahoru → podesta → schody dolů", color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf_t5_celek.png"), dpi=150)
    plt.close(fig)

    # ---------------- graf 2: detaily úseků
    zoom = []
    for nm, a, b in bounds:
        if nm == "sed + vstávání":
            zoom.append((nm, 0.2, min(b, 3.6)))
        elif nm in ("rovina", "schody nahoru", "schody dolů"):
            mid = 0.5 * (a + b)
            zoom.append((nm, mid - 1.3, mid + 1.3))
    fig, axs = plt.subplots(3, len(zoom), figsize=(15, 8.5), sharey="row")
    for c, (nm, a, b) in enumerate(zoom):
        sel = (t >= a) & (t <= b)
        tt = t[sel]
        on = d["motor_zapnut"][sel] > 0.5
        for r in range(3):
            axs[r, c].fill_between(tt, 0, 1, where=on, color=C_ON, alpha=0.15, lw=0,
                                   transform=axs[r, c].get_xaxis_transform())
            axs[r, c].axhline(0, color=INK2, lw=0.5)
        axs[0, c].plot(tt, d["koleno_flexe_deg"][sel], color=INK)
        axs[0, c].set_title(nm, color=INK)
        axs[1, c].plot(tt, d["omega_stehno_deg_s"][sel], color=C_TH, label="stehno")
        axs[1, c].plot(tt, d["omega_holen_deg_s"][sel], color=C_SH, lw=1.1, label="holeň")
        axs[2, c].plot(tt, d["M_off"][sel], color=C_OFF, lw=1.1, label="vypnutý")
        axs[2, c].plot(tt, d["M_on"][sel], color=C_ON, label="zapnutý (síť)")
        axs[2, c].set_xlabel("Čas [s]")
    axs[0, 0].set_ylabel("Flexe kolene [°]")
    axs[1, 0].set_ylabel("Úhlová rychlost [°/s]")
    axs[2, 0].set_ylabel("Moment kolene člověka [N m]")
    axs[1, 0].legend(frameon=False, fontsize=8, loc="upper left")
    axs[2, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Detail úseků (podbarvení = motor exoskeletu zapnutý)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf_t5_useky.png"), dpi=150)
    plt.close(fig)

    # ---------------- souhrn
    dt = t[1] - t[0]
    rows = []
    for nm, a, b in bounds:
        sel = (t >= a) & (t < b)
        w = d["omega_kolene"][sel]
        rows.append((nm, b - a, d["motor_zapnut"][sel].mean() * 100,
                     d["M_off"][sel].max(), d["M_on"][sel].max(),
                     np.sum(np.clip(d["M_off"][sel] * w, 0, None)) * dt,
                     np.sum(np.clip(d["M_on"][sel] * w, 0, None)) * dt,
                     np.sum(np.clip(d["M_off"][sel], 0, None)) * dt, np.sum(np.clip(d["M_on"][sel], 0, None)) * dt,
                     d["M_neomez"][sel].max() if d["M_neomez"] is not None else float("nan"),
                     np.mean((d["exo_motor_Nm"][sel] - np.clip(d["M_off"][sel], 0, None)) > 5) * 100,
                     np.max(d["exo_motor_Nm"][sel] - np.clip(d["M_off"][sel], 0, None)),
                     np.sum(d["exo_motor_Nm"][sel] * w) * dt))
    with open(os.path.join(RES, "souhrn_test5.md"), "w", encoding="utf-8") as f:
        f.write("# test5 – vstávání, rovina, schody nahoru a dolů: souhrn\n\n"
                "Moment kolene člověka z InverseDynamicsTool (motor exo v modelu). Práce a impuls za celý úsek. "
                "+ = extenzní moment / generovaná energie.\n\n"
                f"Motor: max {(OSOBA.get('exoskelet') or {}).get('max_moment_Nm', 60)} N m. 'Neomezený motor' = max. "
                "moment člověka, kdyby motor neměl omezení (jen přesnost sítě). 'Přebytek' = motor dává o víc než "
                "5 N m víc, než je potřeba extenze (překompenzace).\n\n"
                "| Úsek | Délka [s] | Motor ON [%] | Max. moment vyp. [N m] | Max. moment zap. [N m] | "
                "Max. moment, neomezený motor [N m] | Přebytek motoru [% času] / max [N m] | "
                "Kladná práce kolene vyp. / zap. [J] | Extenzní impuls vyp. / zap. [N m s] | Práce motoru [J] |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]:.1f} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f} | {r[9]:.1f} | "
                    f"{r[10]:.1f} / {r[11]:.0f} | {r[5]:.0f} / {r[6]:.0f} | {r[7]:.0f} / {r[8]:.0f} | {r[12]:.0f} |\n")
    print(open(os.path.join(RES, "souhrn_test5.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
