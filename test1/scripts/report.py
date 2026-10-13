"""Krok 3: export výsledků pro jednu nohu (CSV), grafy a souhrnná tabulka špiček."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from osimtools import *  # noqa: F401,F403,E402
import params as P  # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
S = P.SIDE
JOINTS = [("hip_flexion", "Kyčel", "flexe +", "flexní +"),
          ("knee_angle", "Koleno", "extenze + (flexe −)", "extenzní +"),
          ("ankle_angle", "Kotník", "dorzální flexe +", "dorzální +")]
C_LEVEL, C_STAIR = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 2})


def load():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    kc, kd = read_mot(os.path.join(DATA, "walk_stairs_ik.mot"))
    ic, idd = read_mot(os.path.join(RES, "ID_walk_stairs.sto"))
    vc, vd = read_mot(os.path.join(RES, "ID_validation_level.sto"))
    gc, gd = read_mot(os.path.join(DATA, "walk_stairs_grf.mot"))
    return meta, (kc, kd), (ic, idd), (vc, vd), (gc, gd)


def cycles(meta):
    Tc = meta["Tc"]
    tk = {int(k): v for k, v in meta["t_k"].items()}
    lev = [(i * Tc, (i + 1) * Tc) for i in range(1, 7)]
    tau_to = [t for (s, e, t) in meta["level_events"] if s == S and e == "off"][0] % Tc
    stairs = [(tk[k], tk[k + 2]) for k in range(4, 15, 2)]
    ds = P.STAIR_DS * P.STAIR_CYCLE
    return lev, stairs, 100 * tau_to / Tc, 100 * (P.STAIR_CYCLE / 2 + ds) / P.STAIR_CYCLE


def normalize(t, y, cyc, n=101):
    out = []
    for a, b in cyc:
        out.append(np.interp(np.linspace(a, b, n), t, y))
    return np.array(out)


def segment_kinematics(kc, kd, n):
    """Absolutní úhel stehna a lýtka vůči zemi (rovina pohybu) a jejich úhlová rychlost.

    Úhel = natočení segmentu kolem příčné osy z, 0 = svisle (stoj vzpřímeně),
    + = distální konec vpředu (rotace ve směru flexe kyčle)."""
    kin = Kin(os.path.join(ROOT, "model", "subject01_stairs.osim"))
    femur = kin.model.getBodySet().get(f"femur_{S}")
    tibia = kin.model.getBodySet().get(f"tibia_{S}")
    q = kd[:n, 1:].copy()
    for j, c in enumerate(COORDS):
        if c in ROT:
            q[:, j] = np.radians(q[:, j])
    kin.set(dict.fromkeys(COORDS, 0.0))
    ref = [b.getTransformInGround(kin.state).R().convertRotationToBodyFixedXYZ().get(2)
           for b in (femur, tibia)]
    out = np.zeros((n, 2))
    for i in range(n):
        kin.set(dict(zip(COORDS, q[i])))
        for k, b in enumerate((femur, tibia)):
            out[i, k] = b.getTransformInGround(kin.state).R().convertRotationToBodyFixedXYZ().get(2) - ref[k]
    out = np.degrees(np.unwrap(out, axis=0))
    t = kd[:n, 0]
    return out[:, 0], out[:, 1], np.gradient(out[:, 0], t), np.gradient(out[:, 1], t)


def main():
    meta, (kc, kd), (ic, idd), (vc, vd), (gc, gd) = load()
    mass = meta["mass"]
    t = kd[:, 0]
    n = min(len(t), len(idd))
    t = t[:n]
    TA, tk = meta["TA"], {int(k): v for k, v in meta["t_k"].items()}
    t_lastHS = 7 * meta["Tc"]
    seg = np.where(t < t_lastHS, "rovina", np.where(t < tk[2], "prechod", "schody"))
    # kontakt pravé nohy
    vy = gd[:n, gc.index("ground_force_vy")]
    stance = np.where(vy > 20, "stoj", "svih")

    cols = {"cas_s": t, "x_panev_m": kd[:n, kc.index("pelvis_tx")],
            "y_panev_m": kd[:n, kc.index("pelvis_ty")]}
    ang, mom, pw = {}, {}, {}
    for key, name, _, _ in JOINTS:
        q = kd[:n, kc.index(f"{key}_{S}")]
        m = idd[:n, ic.index(f"{key}_{S}_moment")]
        w = m * np.gradient(np.radians(q), t)
        ang[key], mom[key], pw[key] = q, m, w
    cols.update({"kycel_flexe_deg": ang["hip_flexion"],
                 "koleno_flexe_deg": -ang["knee_angle"],
                 "kotnik_dorzalni_flexe_deg": ang["ankle_angle"],
                 "kycel_moment_Nm(+flexni)": mom["hip_flexion"],
                 "koleno_moment_Nm(+extenzni)": mom["knee_angle"],
                 "kotnik_moment_Nm(+dorzalni)": mom["ankle_angle"],
                 "kycel_moment_Nm/kg": mom["hip_flexion"] / mass,
                 "koleno_moment_Nm/kg": mom["knee_angle"] / mass,
                 "kotnik_moment_Nm/kg": mom["ankle_angle"] / mass,
                 "kycel_vykon_W": pw["hip_flexion"], "koleno_vykon_W": pw["knee_angle"],
                 "kotnik_vykon_W": pw["ankle_angle"],
                 "GRF_svisla_N": vy, "GRF_predozadni_N": gd[:n, gc.index("ground_force_vx")]})
    th_a, sh_a, th_w, sh_w = segment_kinematics(kc, kd, n)
    cols.update({"stehno_uhel_deg": th_a, "lytko_uhel_deg": sh_a,
                 "stehno_uhl_rychlost_deg_s": th_w, "lytko_uhl_rychlost_deg_s": sh_w})
    names = list(cols)
    out = os.path.join(RES, f"prava_noha_momenty_15m.csv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("usek;faze;" + ";".join(names) + "\n")
        for i in range(n):
            f.write(f"{seg[i]};{stance[i]};" + ";".join(f"{cols[c][i]:.4f}" for c in names) + "\n")
    print("CSV:", out)

    lev, sta, to_lev, to_sta = cycles(meta)

    # ---------------- graf 1: celých 15 m
    fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, (key, name, _, mlab) in zip(axs, JOINTS):
        ax.plot(t, mom[key], color=INK, lw=1.2)
        ax.set_ylabel(f"{name} [Nm]\n({mlab})")
    axs[3].plot(t, vy / (mass * 9.81), color=INK, lw=1.2)
    axs[3].set_ylabel("Svislá GRF pravé\n[násobek váhy]")
    axs[3].set_xlabel("Čas [s]")
    xp = cols["x_panev_m"]
    for ax in axs:
        ax.axvspan(t_lastHS, tk[2], color="#f0efec", zorder=0, lw=0)
        ax.axvspan(tk[2], t[-1], color="#fbe3d9", zorder=0, lw=0)
        ax.axvspan(t[0], t_lastHS, color="#e3eefa", zorder=0, lw=0)
    for x0, x1, lab in ((0, t_lastHS, f"rovina (0–{np.interp(t_lastHS, t, xp):.1f} m)"),
                        (t_lastHS, tk[2], "přechod"),
                        (tk[2], t[-1], f"schody ({P.N_STEPS} stupňů {P.STEP_H*1000:.0f}/{P.STEP_G*1000:.0f} mm)")):
        axs[0].text((x0 + x1) / 2, 1.02, lab, transform=axs[0].get_xaxis_transform(),
                    ha="center", va="bottom", color=INK2)
    fig.suptitle(f"Pravá noha – čisté kloubní momenty během 15 m chůze (model {mass:.1f} kg)",
                 color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf1_momenty_15m.png"), dpi=150)
    plt.close(fig)

    # ---------------- graf 2: normalizovaný cyklus rovina vs schody
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))
    x = np.linspace(0, 100, 101)
    rows = [("Úhel [deg]", lambda k: ang[k] * (-1 if k == "knee_angle" else 1)),
            ("Moment [Nm/kg]", lambda k: mom[k] / mass),
            ("Výkon [W/kg]", lambda k: pw[k] / mass)]
    for r, (rlab, fn) in enumerate(rows):
        for c, (key, name, alab, mlab) in enumerate(JOINTS):
            ax = axs[r, c]
            y = fn(key)
            for cyc, col, lab, to in ((lev, C_LEVEL, "rovina", to_lev), (sta, C_STAIR, "schody", to_sta)):
                Y = normalize(t, y, cyc)
                mu, sd = Y.mean(0), Y.std(0)
                ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
                ax.plot(x, mu, color=col, label=lab)
                ax.axvline(to, color=col, lw=0.8, ls=":")
            ax.axhline(0, color=INK2, lw=0.6)
            if r == 0:
                ax.set_title(name, color=INK)
            if c == 0:
                ax.set_ylabel(rlab)
            if r == 2:
                ax.set_xlabel("Krokový cyklus pravé nohy [%]")
            sub = {0: "flexe +" if key != "ankle_angle" else "dorzální +", 1: mlab, 2: "generace +"}[r]
            ax.text(0.98, 0.96, sub, transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8)
    axs[0, 0].legend(frameon=False, loc="upper center")
    fig.suptitle("Rovina vs. výstup do schodů – průměr ± SD přes cykly (tečkovaně = odraz)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf2_cyklus_rovina_vs_schody.png"), dpi=150)
    plt.close(fig)

    # ---------------- graf 3: validace dopočtu GRF na rovině
    Gv = np.load(os.path.join(DATA, "grf_validation_level.npy"))
    nv = min(len(vd), len(Gv))
    tv = vd[:nv, 0]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    val_stats = []
    for c, (key, name, _, mlab) in enumerate(JOINTS):
        ax = axs[0, c]
        Ym = normalize(t, mom[key] / mass, lev).mean(0)
        Ys = normalize(tv, vd[:nv, vc.index(f"{key}_{S}_moment")] / mass, lev).mean(0)
        ax.plot(x, Ym, color=C_LEVEL, label="naměřené GRF")
        ax.plot(x, Ys, color=C_STAIR, ls="--", label="dopočtené GRF (metoda pro schody)")
        ax.set_title(name, color=INK)
        ax.set_xlabel("Krokový cyklus [%]")
        rms = np.sqrt(np.mean((Ym - Ys) ** 2))
        r_ = np.corrcoef(Ym, Ys)[0, 1]
        val_stats.append((name, rms, r_, np.abs(Ym).max()))
        ax.text(0.02, 0.04, f"RMS rozdíl {rms:.2f} Nm/kg, r = {r_:.2f}", transform=ax.transAxes,
                color=INK2, fontsize=8)
    axs[0, 0].set_ylabel("Moment [Nm/kg]")
    axs[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    for c, (j, lab) in enumerate(((1, "Svislá GRF pravé [N]"), (0, "Předozadní GRF pravé [N]"))):
        ax = axs[1, c]
        Ym = normalize(t, gd[:n, j + 1], lev).mean(0)
        Ys = normalize(tv, Gv[:nv, j], lev).mean(0)
        ax.plot(x, Ym, color=C_LEVEL, label="naměřená")
        ax.plot(x, Ys, color=C_STAIR, ls="--", label="dopočtená")
        ax.set_ylabel(lab)
        ax.set_xlabel("Krokový cyklus [%]")
    axs[1, 0].legend(frameon=False, fontsize=8)
    ax = axs[1, 2]
    ax.axis("off")
    ax.text(0, 0.9, "Ověření metody:\nna rovině jsou k dispozici naměřené\nreakční síly. Stejná metoda, jakou\n"
            "jsou dopočteny síly na schodech,\nzde reprodukuje naměřené momenty.", va="top", color=INK2)
    fig.suptitle("Validace dopočtu reakčních sil (rovinná chůze, pravá noha)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf3_validace_GRF.png"), dpi=150)
    plt.close(fig)

    # ---------------- graf 4 a 5: kinematika (úhly kyčle/kolene, úhlové rychlosti segmentů)
    kin_rows = [("Kyčel – flexe [deg]", ang["hip_flexion"], "flexe +"),
                ("Koleno – flexe [deg]", -ang["knee_angle"], "flexe +"),
                ("Stehno – úhlová rychlost [deg/s]", th_w, "+ = rotace vpřed"),
                ("Lýtko – úhlová rychlost [deg/s]", sh_w, "+ = rotace vpřed")]
    fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, (lab, y, sub) in zip(axs, kin_rows):
        ax.axvspan(t[0], t_lastHS, color="#e3eefa", zorder=0, lw=0)
        ax.axvspan(t_lastHS, tk[2], color="#f0efec", zorder=0, lw=0)
        ax.axvspan(tk[2], t[-1], color="#fbe3d9", zorder=0, lw=0)
        ax.plot(t, y, color=INK, lw=1.2)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylabel(lab.replace(" – ", "\n"))
        ax.text(0.995, 0.95, sub, transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8)
    for x0, x1, lab in ((0, t_lastHS, "rovina"), (t_lastHS, tk[2], "přechod"), (tk[2], t[-1], "schody")):
        axs[0].text((x0 + x1) / 2, 1.02, lab, transform=axs[0].get_xaxis_transform(),
                    ha="center", va="bottom", color=INK2)
    axs[-1].set_xlabel("Čas [s]")
    fig.suptitle("Pravá noha – kinematika během 15 m chůze (úhlové rychlosti stehna a lýtka vůči zemi)",
                 color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf4_kinematika_15m.png"), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (lab, y, sub) in zip(axs.ravel(), kin_rows):
        for cyc, col, clab, to in ((lev, C_LEVEL, "rovina", to_lev), (sta, C_STAIR, "schody", to_sta)):
            Y = normalize(t, y, cyc)
            mu, sd = Y.mean(0), Y.std(0)
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
            ax.plot(x, mu, color=col, label=clab)
            ax.axvline(to, color=col, lw=0.8, ls=":")
            i = int(np.argmax(np.abs(mu)))
            ax.annotate(f"{mu[i]:.0f}", (x[i], mu[i]), xytext=(4, 4 if mu[i] > 0 else -12),
                        textcoords="offset points", color=INK2, fontsize=8)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(lab, color=INK)
        ax.set_xlabel("Krokový cyklus pravé nohy [%]")
        ax.text(0.98, 0.04, sub, transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=8)
    axs[0, 0].legend(frameon=False, loc="upper center")
    fig.suptitle("Kinematika pravé nohy – rovina vs. schody (průměr ± SD, tečkovaně = odraz)", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "graf5_kinematika_cyklus.png"), dpi=150)
    plt.close(fig)
    print("Kinematika (rovina / schody):")
    for lab, y, _ in kin_rows:
        Yl, Ys = normalize(t, y, lev).mean(0), normalize(t, y, sta).mean(0)
        print(f"  {lab}: rovina {Yl.min():.0f} až {Yl.max():.0f}, schody {Ys.min():.0f} až {Ys.max():.0f}")

    # ---------------- souhrn špiček
    lines = ["| Kloub | Veličina | Rovina | Schody | Přechod (orientačně) |", "|---|---|---|---|---|"]
    summary = {}
    for key, name, alab, mlab in JOINTS:
        for lab, y, unit in (("moment max", mom[key], "Nm"), ("moment min", mom[key], "Nm"),
                             ("výkon max", pw[key], "W"), ("výkon min", pw[key], "W")):
            f = np.max if "max" in lab else np.min
            vl = np.mean([f(r) for r in normalize(t, y, lev)])
            vs = np.mean([f(r) for r in normalize(t, y, sta)])
            va = f(y[seg == "prechod"])
            summary[f"{key}_{lab}"] = (vl, vs, va)
            if unit == "Nm":
                cell = lambda v: f"{v:7.1f} Nm ({v / mass:+.2f} Nm/kg)"
            else:
                cell = lambda v: f"{v:7.0f} W"
            lines.append(f"| {name} | {lab} | {cell(vl)} | {cell(vs)} | {cell(va)} |")
        q = ang[key] * (-1 if key == "knee_angle" else 1)
        rl = np.mean([r.max() - r.min() for r in normalize(t, q, lev)])
        rs = np.mean([r.max() - r.min() for r in normalize(t, q, sta)])
        lines.append(f"| {name} | rozsah pohybu | {rl:.0f}° | {rs:.0f}° | |")
    table = "\n".join(lines)
    with open(os.path.join(RES, "souhrn_spicek.md"), "w", encoding="utf-8") as f:
        f.write(f"# Špičkové hodnoty – pravá noha (hmotnost modelu {mass:.1f} kg)\n\n")
        f.write("Znaménka: kyčel + = flexní moment, koleno + = extenzní moment, "
                "kotník + = dorzální (− = plantární). Výkon + = generace, − = absorpce.\n"
                "Rovina = průměr 6 cyklů, schody = průměr 6 ustálených cyklů.\n\n")
        f.write(table + "\n\n## Validace dopočtu GRF (rovina)\n\n| Kloub | RMS rozdíl [Nm/kg] | r |\n|---|---|---|\n")
        for name, rms, r_, _ in val_stats:
            f.write(f"| {name} | {rms:.3f} | {r_:.3f} |\n")
    print(table)
    print("Validace:", [(nm, round(a, 3), round(b, 3)) for nm, a, b, _ in val_stats])
    print("Konec dráhy x =", round(xp[-1], 2), "m, výška", round(cols['y_panev_m'][-1] - cols['y_panev_m'][0], 2), "m")


if __name__ == "__main__":
    main()
