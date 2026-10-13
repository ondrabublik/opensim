"""Krok 1b: doladění vodorovného pohybu pánve v přechodu rovina -> schody.

Při zpomalování před schody by s jednoduchou trajektorií muselo působiště reakční
síly (COP) ležet mimo chodidlo. Tady se iterativně opravuje vodorovné zrychlení
pánve podle modelu obráceného kyvadla: posun COP o d vyžaduje změnu zrychlení
těžiště o omega^2 * d (omega^2 = g / výška těžiště). Okrajové podmínky (navázání
na naměřenou chůzi a na ustálený výstup) zůstávají zachované.
"""
import contextlib
import io
import json

from osimtools import *  # noqa: F401,F403
import build_motion as B
import compute_dynamics as C

N_ITER = 12
RELAX = 0.6
OMEGA2 = 9.81 / 0.95


def evaluate():
    with contextlib.redirect_stdout(io.StringIO()):
        B.main()
    meta = json.load(open(os.path.join(C.DATA, "meta.json")))
    kin_file = os.path.join(C.DATA, "walk_stairs_ik.mot")
    _, kd = read_mot(kin_file)
    TA = meta["TA"]
    tk = {int(k): v for k, v in meta["t_k"].items()}
    sel = (kd[:, 0] > TA - 0.3) & (kd[:, 0] < tk[3])
    t = kd[sel, 0]
    Q = kd[sel, 1:].copy()
    for j, c in enumerate(COORDS):
        if c in ROT:
            Q[:, j] = np.radians(Q[:, j])
    kin = Kin(C.MODEL)
    pel, feet, com = C.foot_geometry(kin, Q)
    c0, d0 = C.run_id("ID_tune", kin_file, t[0], t[-1])
    n = min(len(t), len(d0))
    t, pel, com = t[:n], pel[:n], com[:n]
    feet = {s: feet[s][:n] for s in feet}
    Qn = d0[:n][:, [c0.index("pelvis_tilt_moment"), c0.index("pelvis_tx_force"),
                    c0.index("pelvis_ty_force")]]
    cs = {s: [(max(c["t_on"] if c["t_on"] is not None else TA - 1, TA - 1),
               c["t_off"] if c["t_off"] is not None else 99, c["surface"])
              for c in meta["contacts"][s]] for s in C.SIDES}
    _, res = C.synthesize(t, Qn, pel, feet, com, cs)
    return t, res, Qn[:, 2], TA, tk


def main():
    if os.path.exists(B.CORR_FILE):
        os.remove(B.CORR_FILE)
    grid = None
    corr = None
    for it in range(N_ITER + 1):
        t, res, Fy, TA, tk = evaluate()
        m = t > TA
        print(f"iterace {it:2d}: nezařazený moment v přechodu RMS {np.sqrt(np.mean(res[m]**2)):6.1f} Nm, "
              f"max {np.abs(res[m]).max():6.1f} Nm", flush=True)
        if it == N_ITER:
            break
        if grid is None:
            grid = np.arange(TA, tk[2] + 1e-9, 0.005)
            corr = np.zeros_like(grid)
        da = RELAX * OMEGA2 * np.interp(grid, t, res / np.maximum(Fy, 100.0))
        da *= smoothstep((grid - TA) / 0.1)
        da = butter_lowpass(da, 8.0, 200.0)
        dv = np.concatenate([[0], np.cumsum(0.5 * (da[1:] + da[:-1]) * 0.005)])
        dx = np.concatenate([[0], np.cumsum(0.5 * (dv[1:] + dv[:-1]) * 0.005)])
        # odečti polynom tak, aby v t2 byla korekce polohy, rychlosti i zrychlení nulová
        T = grid[-1] - TA
        c = quintic(0, 0, 0, dx[-1], dv[-1], da[-1], T)
        dx -= polyval(c, grid - TA)
        corr = corr + dx
        np.save(B.CORR_FILE, np.vstack([grid, corr]))


if __name__ == "__main__":
    main()
