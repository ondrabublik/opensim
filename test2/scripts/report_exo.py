"""Krok 4: grafy, CSV a souhrn pro exoskelet pravého kolene."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from osimtools import *  # noqa: F401,F403,E402
import build_exo_model as BM  # noqa: E402
import cadlib as C  # noqa: E402
import exo_control as X  # noqa: E402
import exo_params as E  # noqa: E402
from render import draw_meshes, new_axes, save_fig  # noqa: E402
from run_exo import cycles, in_cycles  # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
CAD = os.path.join(ROOT, "cad")
OSIM_GEOM = r"C:\OpenSim 4.5\Geometry"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_MODE = {"uhel": "#2a78d6", "uhel_smer": "#eb6834"}
C_LEVEL, C_STAIR = "#1baf7a", "#4a3aa7"
LBL = {"uhel": "řízení: úhel", "uhel_smer": "řízení: úhel + směr"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 2})


def normalize(t, y, cyc, n=101):
    return np.array([np.interp(np.linspace(a, b, n), t, y) for a, b in cyc])


def per_cycle(t, y, cyc, fn):
    return np.mean([fn(t[(t >= a) & (t < b)], y[(t >= a) & (t < b)]) for a, b in cyc])


def work(tt, p):
    """(kladná, záporná) práce [J] z výkonu p."""
    dt = np.gradient(tt)
    return (p.clip(min=0) * dt).sum(), (p.clip(max=0) * dt).sum()


# ------------------------------------------------------------------ vizualizace na noze
def leg_meshes(model, state, bodies):
    out = []
    for bname in bodies:
        b = model.getBodySet().get(bname)
        T = b.getTransformInGround(state)
        R = np.array([[T.R().get(i, j) for j in range(3)] for i in range(3)])
        p = np.array([T.p().get(i) for i in range(3)])
        for i in range(b.getPropertyByName("attached_geometry").size()):
            g = b.get_attached_geometry(i)
            mesh = osim.Mesh.safeDownCast(g)
            if mesh is None:
                continue
            f = mesh.get_mesh_file()
            sc = np.array([g.get_scale_factors().get(k) for k in range(3)])
            if f.endswith(".stl"):
                V, F = C.read_stl(os.path.join(ROOT, "model", "Geometry", f))
                col = [g.get_Appearance().get_color().get(k) for k in range(3)]
            else:
                V, F = C.read_vtp_points_polys(os.path.join(OSIM_GEOM, f))
                col = (0.92, 0.89, 0.80)
            out.append(((V * sc) @ R.T + p, F, col, 1.0))
    return out


def render_on_leg():
    model = osim.Model(BM.MAIN_MODEL)
    s = model.initSystem()
    cs = model.getCoordinateSet()
    bodies = ["femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r", "exo_thigh_r", "exo_shank_r"]
    fig = plt.figure(figsize=(14, 4.4))
    poses = [(0, 0, 0, -90, "natažené koleno (0°)"), (60, 30, 0, -90, "flexe 60° (dopad na schod)"),
             (100, 60, 0, -90, "flexe 100° (švih na schodech)"), (60, 30, 20, -40, "flexe 60°, izometrie")]
    for k, (knee, hip, el, az, title) in enumerate(poses):
        for c in COORDS:
            cs.get(c).setValue(s, 0.0, False)
        cs.get("pelvis_ty").setValue(s, 1.0, False)
        cs.get("hip_flexion_r").setValue(s, np.radians(hip), False)
        cs.get("knee_angle_r").setValue(s, -np.radians(knee), False)
        model.realizePosition(s)
        ax = new_axes(fig, 141 + k)
        draw_meshes(ax, leg_meshes(model, s, bodies), elev=el, azim=az, zoom=1.7)
        ax.set_title(title, fontsize=10, color=INK)
    fig.suptitle("Exoskelet na pravém koleni modelu (kosti z OpenSim, laterální pohled)", fontsize=11)
    save_fig(fig, os.path.join(CAD, "exo_na_noze.png"))


# ------------------------------------------------------------------ hlavní
def main():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    r = dict(np.load(os.path.join(RES, "exo_vysledky.npz")))
    tab = json.load(open(BM.TABLE_FILE))
    hinge = json.load(open(os.path.join(CAD, "osa_kloubu.json")))
    props = json.load(open(os.path.join(CAD, "exo_hmotnosti.json")))
    m_exo = sum(v["mass"] for v in props.values())
    mass = meta["mass"]
    t, th, om = r["t"], r["theta"], r["omega"]
    lev, sta = cycles(meta)
    tk = {int(k): v for k, v in meta["t_k"].items()}
    t_lastHS = 7 * meta["Tc"]
    seg = np.where(t < t_lastHS, "rovina", np.where(t < tk[2], "prechod", "schody"))
    gc, gd = read_mot(os.path.join(DATA, "walk_stairs_grf_exo.mot"))
    stance = gd[:len(t), gc.index("ground_force_vy")] > 20
    modes = list(E.CONTROL_MODES)
    H = {m: r[f"M_{m}"] for m in modes}           # moment člověka s aktivním exo
    tau = {m: r[f"tau_{m}"] for m in modes}
    P = {m: tau[m] * om for m in modes}

    # ---------------- CSV
    cols = {"cas_s": t, "koleno_flexe_deg": th, "koleno_rychlost_extenze_rad_s": om,
            "moment_bez_exo_Nm": r["M_bez"], "moment_pasivni_exo_Nm": r["M_pas"]}
    for m in modes:
        cols[f"exo_moment_{m}_Nm"] = tau[m]
        cols[f"exo_vykon_{m}_W"] = P[m]
        cols[f"moment_clovek_{m}_Nm"] = H[m]
    with open(os.path.join(RES, "exo_koleno_15m.csv"), "w", encoding="utf-8") as f:
        f.write("usek;faze;" + ";".join(cols) + "\n")
        for i in range(len(t)):
            f.write(f"{seg[i]};{'stoj' if stance[i] else 'svih'};"
                    + ";".join(f"{cols[c][i]:.4f}" for c in cols) + "\n")

    # ---------------- graf 1: řízení
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axs[0]
    mk = r["mask"].astype(bool)
    for cyc, col, lab in ((lev, C_LEVEL, "potřebný moment kolene – rovina"),
                          (sta, C_STAIR, "potřebný moment kolene – schody")):
        sel = in_cycles(t, cyc) & mk
        ax.scatter(th[sel][::2], r["M_pas"][sel][::2], s=4, color=col, alpha=0.35, lw=0, label=lab)
    thg = np.linspace(-5, 125, 400)
    for m in modes:
        ax.plot(thg, X.curve(thg, tab["table"][m]), color=C_MODE[m], label=f"G(θ) – {LBL[m]}")
        ax.plot(X.NODES, tab["table"][m], "o", color=C_MODE[m], ms=4)
    ax.axhline(E.TAU_MAX, color=INK2, lw=0.8, ls="--")
    ax.text(122, E.TAU_MAX + 3, f"max. moment pohonu {E.TAU_MAX:.0f} Nm", ha="right", color=INK2, fontsize=8)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Flexe kolene θ [deg]")
    ax.set_ylabel("Moment [Nm] (+ extenzní)")
    ax.set_title("Křivka asistence G(θ) a potřebný moment kolene", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=3)
    ax = axs[1]
    w = np.linspace(-8, 8, 400)
    ax.plot(w, X.gate(w, "uhel"), color=C_MODE["uhel"], label=LBL["uhel"])
    ax.plot(w, X.gate(w, "uhel_smer"), color=C_MODE["uhel_smer"], label=LBL["uhel_smer"])
    ax.set_xlabel("Rychlost kolene ω [rad/s] (+ extenze)")
    ax.set_ylabel("Násobitel s(ω) [–]")
    ax.set_title("Brána podle směru pohybu", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax = axs[2]
    ax.plot(hinge["theta_deg"], hinge["misalignment_naive_mm"], color=INK2, ls="--",
            label="osa v počátku tibie")
    ax.plot(hinge["theta_deg"], hinge["misalignment_mm"], color=INK, label="optimalizovaná osa")
    ax.set_xlabel("Flexe kolene [deg]")
    ax.set_ylabel("Posun osy exo vůči bérci [mm]")
    ax.set_title("Nesouosost pevné osy exo a kolene", color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Řízení exoskeletu: τ = G(θ)·s(ω), vyladěno na {E.ASSIST_RATIO:.0%} momentu kolene",
                 color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf_exo1_rizeni.png"), dpi=150)
    plt.close(fig)

    # ---------------- graf 2: 15 m
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axs[0].plot(t, th, color=INK, lw=1.2)
    axs[0].set_ylabel("Flexe kolene [deg]")
    for m in modes:
        axs[1].plot(t, tau[m], color=C_MODE[m], lw=1.2, label=LBL[m])
    axs[1].set_ylabel("Moment exo [Nm]\n(+ extenzní)")
    axs[1].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    axs[2].plot(t, r["M_bez"], color=INK2, lw=1.0, label="bez exo")
    axs[2].plot(t, H["uhel_smer"], color=C_MODE["uhel_smer"], lw=1.2, label=f"člověk + exo ({LBL['uhel_smer']})")
    axs[2].set_ylabel("Moment kolene člověka [Nm]\n(+ extenzní)")
    axs[2].set_xlabel("Čas [s]")
    axs[2].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    for ax in axs:
        ax.axvspan(t[0], t_lastHS, color="#e3eefa", zorder=0, lw=0)
        ax.axvspan(t_lastHS, tk[2], color="#f0efec", zorder=0, lw=0)
        ax.axvspan(tk[2], t[-1], color="#fbe3d9", zorder=0, lw=0)
    for x0, x1, lab in ((t[0], t_lastHS, "rovina"), (t_lastHS, tk[2], "přechod"), (tk[2], t[-1], "schody")):
        axs[0].text((x0 + x1) / 2, 1.02, lab, transform=axs[0].get_xaxis_transform(), ha="center",
                    va="bottom", color=INK2)
    fig.suptitle(f"Exoskelet pravého kolene během 15 m chůze (model {mass:.1f} kg + exo {m_exo:.2f} kg)",
                 color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf_exo2_15m.png"), dpi=150)
    plt.close(fig)

    # ---------------- graf 3: cyklus
    x = np.linspace(0, 100, 101)
    fig, axs = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    for c, (cyc, name) in enumerate(((lev, "Rovina"), (sta, "Schody"))):
        a = axs[0, c]
        Y = normalize(t, th, cyc)
        a.plot(x, Y.mean(0), color=INK)
        a.set_title(name, color=INK)
        a = axs[1, c]
        a.plot(x, normalize(t, r["M_bez"], cyc).mean(0), color=INK, label="bez exo")
        a.plot(x, normalize(t, r["M_pas"], cyc).mean(0), color=INK2, ls="--", lw=1.4,
               label="pasivní exo (jen hmotnost)")
        for m in modes:
            a.plot(x, normalize(t, H[m], cyc).mean(0), color=C_MODE[m], label=LBL[m])
        a = axs[2, c]
        for m in modes:
            Y = normalize(t, tau[m], cyc)
            a.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=C_MODE[m], alpha=0.15, lw=0)
            a.plot(x, Y.mean(0), color=C_MODE[m], label=LBL[m])
        a = axs[3, c]
        for m in modes:
            a.plot(x, normalize(t, P[m], cyc).mean(0), color=C_MODE[m], label=LBL[m])
        a.set_xlabel("Krokový cyklus pravé nohy [%] (0 = dopad paty)")
        for a in axs[:, c]:
            a.axhline(0, color=INK2, lw=0.6)
    for rr, lab in enumerate(("Flexe kolene [deg]", "Moment kolene člověka [Nm]\n(+ extenzní)",
                              "Moment exo [Nm]", "Výkon exo [W]\n(+ dodává energii)")):
        axs[rr, 0].set_ylabel(lab)
    axs[1, 0].legend(frameon=False, fontsize=8, loc="lower left")
    axs[2, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Krokový cyklus – průměr přes 6 cyklů (pás = ±SD)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf_exo3_cyklus.png"), dpi=150)
    plt.close(fig)

    # ---------------- souhrn
    rows = []
    for cyc, name in ((lev, "Rovina"), (sta, "Schody")):
        def f(y, fn):
            return per_cycle(t, y, cyc, fn)
        for key, lab in (("M_bez", "bez exo"), ("M_pas", "pasivní exo")):
            y = r[key]
            pw = y * om
            rows.append((name, lab, f(y, lambda a, b: b.max()), f(y, lambda a, b: b.min()),
                         np.sqrt(np.mean(y[in_cycles(t, cyc)] ** 2)),
                         f(pw, lambda a, b: work(a, b)[0]), f(pw, lambda a, b: work(a, b)[1]), None))
        for m in modes:
            y = H[m]
            pw = y * om
            ex = (f(tau[m], lambda a, b: b.max()), f(P[m], lambda a, b: b.max()),
                  f(P[m], lambda a, b: work(a, b)[0]), f(P[m], lambda a, b: work(a, b)[1]),
                  np.abs(om[in_cycles(t, cyc) & (np.abs(tau[m]) > 5)]).max())
            rows.append((name, LBL[m], f(y, lambda a, b: b.max()), f(y, lambda a, b: b.min()),
                         np.sqrt(np.mean(y[in_cycles(t, cyc)] ** 2)),
                         f(pw, lambda a, b: work(a, b)[0]), f(pw, lambda a, b: work(a, b)[1]), ex))

    # kompromis rovina / schody podle váhy roviny při ladění (jen Python, stejný zákon)
    mk = r["mask"].astype(bool)
    ml, ms = in_cycles(t, lev), in_cycles(t, sta)
    trade = []
    for m in modes:
        for wl in (1.0, 3.0, 10.0):
            g = X.tune(th[mk], om[mk], E.ASSIST_RATIO * r["M_pas"][mk], m,
                       weights=np.where(ml[mk], wl, 1.0))
            h = r["M_pas"] - X.torque(th, om, g, m)
            trade.append((LBL[m], wl, np.sqrt(np.mean(h[ml] ** 2)), np.sqrt(np.mean(h[ms] ** 2)),
                          per_cycle(t, h, sta, lambda a, b: b.max())))

    mis = np.array(hinge["misalignment_mm"])
    with open(os.path.join(RES, "souhrn_exo.md"), "w", encoding="utf-8") as fo:
        fo.write(f"# Exoskelet pravého kolene – souhrn\n\n"
                 f"Model {mass:.1f} kg + exoskelet {m_exo:.2f} kg "
                 f"(stehno {props['exo_thigh_r']['mass']:.2f} kg, bérec {props['exo_shank_r']['mass']:.2f} kg). "
                 f"Hodnoty jsou průměrem přes 6 cyklů. Moment + = extenzní; práce + = generace.\n\n"
                 f"Řízení: τ = G(θ)·s(ω), ladění na {E.ASSIST_RATIO:.0%} potřebného momentu, "
                 f"max {E.TAU_MAX:.0f} Nm, váha roviny {E.LEVEL_WEIGHT:g}.\n\n"
                 "## Moment kolene člověka\n\n"
                 "| Úsek | Varianta | Extenzní špička [Nm] | Flexní špička [Nm] | RMS [Nm] | Práce + [J/cyklus] | Práce − [J/cyklus] |\n"
                 "|---|---|---|---|---|---|---|\n")
        for row in rows:
            fo.write(f"| {row[0]} | {row[1]} | {row[2]:.1f} | {row[3]:.1f} | {row[4]:.1f} | {row[5]:.1f} | {row[6]:.1f} |\n")
        fo.write("\n## Pohon exoskeletu\n\n"
                 "| Úsek | Řízení | Moment max [Nm] | Výkon max [W] | Práce + [J/cyklus] | Práce − [J/cyklus] | Max. rychlost při momentu > 5 Nm [rad/s] |\n"
                 "|---|---|---|---|---|---|---|\n")
        for row in rows:
            if row[7] is not None:
                e = row[7]
                fo.write(f"| {row[0]} | {row[1]} | {e[0]:.1f} | {e[1]:.0f} | {e[2]:.1f} | {e[3]:.1f} | {e[4]:.1f} |\n")
        fo.write("\n## Křivka asistence G(θ) [Nm]\n\n| θ [deg] | " + " | ".join(f"{n:.0f}" for n in X.NODES)
                 + " |\n|---|" + "---|" * len(X.NODES) + "\n")
        for m in modes:
            fo.write(f"| {LBL[m]} | " + " | ".join(f"{g:.1f}" for g in tab["table"][m]) + " |\n")
        fo.write("\n## Kompromis rovina × schody (váha roviny při ladění)\n\n"
                 "| Řízení | Váha roviny | RMS rovina [Nm] | RMS schody [Nm] | Extenzní špička schody [Nm] |\n"
                 "|---|---|---|---|---|\n")
        base_l = np.sqrt(np.mean(r["M_pas"][ml] ** 2))
        base_s = np.sqrt(np.mean(r["M_pas"][ms] ** 2))
        fo.write(f"| pasivní exo | – | {base_l:.1f} | {base_s:.1f} | "
                 f"{per_cycle(t, r['M_pas'], sta, lambda a, b: b.max()):.1f} |\n")
        for tr in trade:
            fo.write(f"| {tr[0]} | {tr[1]:g} | {tr[2]:.1f} | {tr[3]:.1f} | {tr[4]:.1f} |\n")
        fo.write(f"\n## Osa kloubu\n\nPevná osa exo ve femur_r ({hinge['femur_point'][0] * 1000:.1f}, "
                 f"{hinge['femur_point'][1] * 1000:.1f}) mm; nesouosost s kolenem v rozsahu "
                 f"{E.HINGE_FIT_RANGE_DEG[0]:.0f}–{E.HINGE_FIT_RANGE_DEG[1]:.0f}° max {mis.max():.1f} mm, "
                 f"RMS {np.sqrt(np.mean(mis ** 2)):.1f} mm "
                 f"(osa v počátku tibie by měla max {max(hinge['misalignment_naive_mm']):.1f} mm).\n")
    print(open(os.path.join(RES, "souhrn_exo.md"), encoding="utf-8").read())
    render_on_leg()
    print("Grafy:", RES)


if __name__ == "__main__":
    main()
