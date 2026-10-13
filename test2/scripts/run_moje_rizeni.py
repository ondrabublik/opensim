"""Vyhodnocení vlastního řízení exoskeletu (moje_rizeni.py nebo jiný soubor).

    python run_moje_rizeni.py [jmeno_modulu]

Předpoklad: proběhl run_exo.py (results/exo_vysledky.npz s momentem kolene pro model
s pasivním exo). Pohyb je předepsaný (kinematika z test1), takže moment, který musí
dodat člověk, je přesně  M_clovek = M_pasivni_exo - tau_motor  (inverzní dynamika je
v přidaném momentu lineární; ověřeno v run_exo.py proti OpenSim, rozdíl < 0,1 Nm).

Výstupy v results/moje_rizeni/<jmeno_modulu>/: CSV, grafy, souhrn.md, moment.sto
"""
import importlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from osimtools import *  # noqa: F401,F403,E402
import report_exo as RE  # noqa: E402
from run_exo import cycles, in_cycles  # noqa: E402

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
C_CTRL, C_REF, C_ON = "#eb6834", "#2a78d6", "#f6d3c4"
INK, INK2 = RE.INK, RE.INK2


def inputs(t):
    """Úhel kolene a absolutní úhlové rychlosti stehna a holeně (stejně jako v test1)."""
    _, kd = read_mot(os.path.join(DATA, "walk_stairs_ik.mot"))
    n = len(t)
    q = {c: kd[:n, 1 + COORDS.index(c)] for c in COORDS}
    stehno = q["pelvis_tilt"] + q["hip_flexion_r"]       # úhel stehna vůči zemi [deg]
    holen = stehno + q["knee_angle_r"]                    # úhel holeně vůči zemi [deg]
    return -q["knee_angle_r"], np.gradient(stehno, t), np.gradient(holen, t)


def simulate(mod, t, uhel, w_st, w_ho):
    """Krok po kroku: řízení -> požadavek -> model motoru (lag 1. řádu + omezení)."""
    tau_max = getattr(mod, "TAU_MAX", 60.0)
    t_m = getattr(mod, "TAU_MOTOR", 0.0)
    m_on = getattr(mod, "MOMENT_ZAPNUTO", 40.0)
    ctrl = mod.Rizeni()
    n = len(t)
    req, tau, on = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    dt = t[1] - t[0]
    a = 1.0 if t_m <= 0 else 1.0 - np.exp(-dt / t_m)
    y = 0.0
    for i in range(n):
        out = ctrl.krok(float(t[i]), float(uhel[i]), float(w_st[i]), float(w_ho[i]), dt)
        if isinstance(out, (bool, np.bool_)):
            on[i] = bool(out)
            req[i] = m_on if out else 0.0
        else:
            req[i] = float(out)
            on[i] = abs(req[i]) > 1e-9
        y += a * (np.clip(req[i], -tau_max, tau_max) - y)
        tau[i] = y
    return req, tau, on


def metrics(t, cyc, M, tau, om):
    m = in_cycles(t, cyc)
    pc = RE.per_cycle
    P = tau * om
    return {"ext_spicka": pc(t, M, cyc, lambda a, b: b.max()),
            "flex_spicka": pc(t, M, cyc, lambda a, b: b.min()),
            "rms": np.sqrt(np.mean(M[m] ** 2)),
            "prace_cloveka_plus": pc(t, M * om, cyc, lambda a, b: RE.work(a, b)[0]),
            "tau_max": pc(t, tau, cyc, lambda a, b: np.abs(b).max()),
            "P_max": pc(t, P, cyc, lambda a, b: b.max()),
            "W_exo_plus": pc(t, P, cyc, lambda a, b: RE.work(a, b)[0]),
            "W_exo_minus": pc(t, P, cyc, lambda a, b: RE.work(a, b)[1])}


def main(name="moje_rizeni"):
    mod = importlib.import_module(name)
    out_dir = os.path.join(RES, "moje_rizeni", name)
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    r = dict(np.load(os.path.join(RES, "exo_vysledky.npz")))
    t, om, M_pas = r["t"], r["omega"], r["M_pas"]
    uhel, w_st, w_ho = inputs(t)
    req, tau, on = simulate(mod, t, uhel, w_st, w_ho)
    H = M_pas - tau
    lev, sta = cycles(meta)
    tk = {int(k): v for k, v in meta["t_k"].items()}
    t_lastHS = 7 * meta["Tc"]
    seg = np.where(t < t_lastHS, "rovina", np.where(t < tk[2], "prechod", "schody"))

    # ---------------- CSV + moment pro OpenSim
    cols = {"cas_s": t, "koleno_flexe_deg": uhel, "omega_stehno_deg_s": w_st, "omega_holen_deg_s": w_ho,
            "motor_zapnut": on.astype(float), "pozadavek_Nm": req, "moment_motoru_Nm": tau,
            "vykon_motoru_W": tau * om, "moment_kolene_pasivni_exo_Nm": M_pas, "moment_kolene_clovek_Nm": H}
    with open(os.path.join(out_dir, "prubeh.csv"), "w", encoding="utf-8") as f:
        f.write("usek;" + ";".join(cols) + "\n")
        for i in range(len(t)):
            f.write(f"{seg[i]};" + ";".join(f"{cols[c][i]:.4f}" for c in cols) + "\n")
    write_mot(os.path.join(out_dir, "moment.sto"), ["time", "exo_knee_r_torque"],
              np.column_stack([t, tau]), "exo_knee_r_torque", in_degrees=False)

    # ---------------- grafy: cyklus rovina vs schody
    x = np.linspace(0, 100, 101)
    nz = RE.normalize
    fig, axs = plt.subplots(5, 2, figsize=(12, 13), sharex=True)
    for c, (cyc, title) in enumerate(((lev, "Rovina"), (sta, "Schody"))):
        onf = nz(t, on.astype(float), cyc).mean(0)
        for a in axs[:, c]:
            a.fill_between(x, 0, 1, where=onf > 0.5, transform=a.get_xaxis_transform(), color=C_ON,
                           lw=0, zorder=0)
            a.axhline(0, color=INK2, lw=0.6)
        axs[0, c].plot(x, nz(t, uhel, cyc).mean(0), color=INK)
        axs[0, c].set_title(title + "  (podbarvení = motor zapnut ve > 50 % cyklů)", color=INK, fontsize=10)
        axs[1, c].plot(x, nz(t, w_st, cyc).mean(0), color=C_REF, label="stehno")
        axs[1, c].plot(x, nz(t, w_ho, cyc).mean(0), color=INK, label="holeň")
        Y = nz(t, tau, cyc)
        axs[2, c].fill_between(x, Y.min(0), Y.max(0), color=C_CTRL, alpha=0.2, lw=0)
        axs[2, c].plot(x, Y.mean(0), color=C_CTRL)
        axs[3, c].plot(x, nz(t, r["M_bez"], cyc).mean(0), color=INK2, label="bez exo")
        axs[3, c].plot(x, nz(t, r["M_uhel_smer"], cyc).mean(0), color=C_REF, lw=1.4,
                       label="vestavěné řízení úhel + směr")
        axs[3, c].plot(x, nz(t, H, cyc).mean(0), color=C_CTRL, label=f"moje řízení ({name})")
        axs[4, c].plot(x, nz(t, tau * om, cyc).mean(0), color=C_CTRL)
        axs[4, c].set_xlabel("Krokový cyklus pravé nohy [%] (0 = dopad)")
    for rr, lab in enumerate(("Flexe kolene [deg]", "Úhlová rychlost [deg/s]\n(+ dopředu)",
                              "Moment motoru [Nm]\n(+ extenzní; pás = min–max)",
                              "Moment kolene člověka [Nm]\n(+ extenzní)", "Výkon motoru [W]\n(+ dodává)")):
        axs[rr, 0].set_ylabel(lab)
    for a in axs.flat:                      # nulový průběh (motor vypnutý) bez "1e-17" na ose
        lo, hi = a.get_ylim()
        if hi - lo < 1.0:
            a.set_ylim(-1.0, 1.0)
    axs[1, 0].legend(frameon=False, fontsize=8)
    axs[3, 1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"Vlastní řízení '{name}' – průměr přes 6 cyklů", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "graf_cyklus.png"), dpi=140)
    plt.close(fig)

    # ---------------- graf: celých 15 m
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(t, uhel, color=INK, lw=1.1)
    axs[0].set_ylabel("Flexe kolene [deg]")
    axs[1].plot(t, tau, color=C_CTRL, lw=1.1)
    axs[1].set_ylabel("Moment motoru [Nm]")
    axs[2].plot(t, r["M_bez"], color=INK2, lw=1.0, label="bez exo")
    axs[2].plot(t, H, color=C_CTRL, lw=1.1, label="s mým řízením")
    axs[2].set_ylabel("Moment kolene člověka [Nm]")
    axs[2].set_xlabel("Čas [s]")
    axs[2].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    for a in axs:
        a.fill_between(t, 0, 1, where=on, transform=a.get_xaxis_transform(), color=C_ON, lw=0, zorder=0)
        a.axvline(t_lastHS, color=INK2, lw=0.6, ls=":")
        a.axvline(tk[2], color=INK2, lw=0.6, ls=":")
    axs[0].text(t_lastHS / 2, 1.02, "rovina", transform=axs[0].get_xaxis_transform(), ha="center", color=INK2)
    axs[0].text((tk[2] + t[-1]) / 2, 1.02, "schody", transform=axs[0].get_xaxis_transform(), ha="center",
                color=INK2)
    fig.suptitle(f"Vlastní řízení '{name}' během 15 m (podbarvení = motor zapnut)", color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "graf_15m.png"), dpi=140)
    plt.close(fig)

    # ---------------- souhrn
    zero = np.zeros_like(t)
    variants = [("bez exo", r["M_bez"], zero), ("pasivní exo", M_pas, zero),
                ("vestavěné úhel + směr", r["M_uhel_smer"], r["tau_uhel_smer"]), (f"moje: {name}", H, tau)]
    lines = [f"# Vlastní řízení `{name}`\n",
             f"Motor: max {getattr(mod, 'TAU_MAX', 60.0):g} Nm, časová konstanta "
             f"{getattr(mod, 'TAU_MOTOR', 0.0) * 1000:g} ms, ON = {getattr(mod, 'MOMENT_ZAPNUTO', 40.0):g} Nm. "
             "Průměr přes 6 cyklů; moment + = extenzní, práce + = generace.\n",
             "| Úsek | Varianta | Extenzní špička člověka [Nm] | Flexní špička [Nm] | RMS [Nm] | "
             "Práce člověka + [J/cyk] | Moment motoru max [Nm] | Výkon max [W] | Práce motoru + / − [J/cyk] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cyc, sec in ((lev, "Rovina"), (sta, "Schody")):
        for lab, M, tq in variants:
            k = metrics(t, cyc, M, tq, om)
            lines.append(f"| {sec} | {lab} | {k['ext_spicka']:.1f} | {k['flex_spicka']:.1f} | {k['rms']:.1f} | "
                         f"{k['prace_cloveka_plus']:.1f} | {k['tau_max']:.1f} | {k['P_max']:.0f} | "
                         f"{k['W_exo_plus']:.1f} / {k['W_exo_minus']:.1f} |")
    share = {s: on[in_cycles(t, c)].mean() * 100 for s, c in (("rovina", lev), ("schody", sta))}
    lines.append(f"\nMotor zapnut: rovina {share['rovina']:.0f} % času, schody {share['schody']:.0f} % času.\n")
    txt = "\n".join(lines)
    open(os.path.join(out_dir, "souhrn.md"), "w", encoding="utf-8").write(txt)
    print(txt)
    print("Výstupy:", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:2])
