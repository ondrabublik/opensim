"""Krok 3: reakční síly (chodidla + sedák židle) z dynamiky celého těla, OpenSim API.

1. InverseDynamicsTool bez vnějších sil -> síla a moment, které musí dodat okolí (zbytek na pánvi).
2. Rozdělení: sedák nese při sezení 80 % svislé síly, do odlehnutí plynule klesne na 0;
   chodidla: jedna opora = vše, dvojí opora = plynulý přenos ze zadní na přední nohu
   (průběh z naměřené chůze, jako v test1), stoj na obou nohách = 50/50.
3. Působiště na chodidlech z momentové rovnováhy (společný posun vůči středu opory,
   omezený na plochu chodidla; při zvednuté patě jen přední část).
4. ExternalLoads (3 síly) + kontrolní ID se silami.
"""
from t5common import *  # noqa: F401,F403

SOLE, HEEL, MTP, TOE = foot_points()                  # chodidlo škálované podle osoby
DS_S = np.linspace(0, 1, 11)
DS_W = np.array([1.0, 0.91, 0.82, 0.73, 0.645, 0.54, 0.415, 0.285, 0.17, 0.085, 0.0])
PELVIS = ["pelvis_tx_force", "pelvis_ty_force", "pelvis_tilt_moment"]


def rot2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def smoothstep(s):
    s = np.clip(s, 0, 1)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def geometry():
    """Poloha pánve, kyčle a póza obou chodidel (calcn) pro každý snímek kinematiky."""
    model = osim.Model(MODELS["pasivni"])
    s = model.initSystem()
    t, Q = table_np(KIN)
    cs = model.getCoordinateSet()
    rot = {cs.get(i).getName(): cs.get(i).getMotionType() == 1 for i in range(cs.getSize())}
    bodies = {b: model.getBodySet().get(b) for b in ("pelvis", "femur_r", "calcn_r", "calcn_l")}
    out = {k: np.zeros((len(t), 3)) for k in bodies}
    for i in range(len(t)):
        for c, v in Q.items():
            cs.get(c).setValue(s, float(np.radians(v[i]) if rot[c] else v[i]), False)
        model.realizePosition(s)
        for k, b in bodies.items():
            X = b.getTransformInGround(s)
            out[k][i] = [X.p().get(0), X.p().get(1), X.R().convertRotationToBodyFixedXYZ().get(2)]
    return t, out


def main():
    m = meta()
    print("ID bez vnějších sil ...")
    t, d0 = run_id("ID_bez_sil", MODELS["pasivni"])
    need = np.column_stack([d0[c] for c in PELVIS])            # Fx, Fy, M (k počátku pánve)
    tg, geo = geometry()
    n = len(t)
    contacts = {s: [(c["t_on"] if c["t_on"] is not None else -1e9, c["t_off"] if c["t_off"] is not None else 1e9,
                     c["surf"]) for c in m["contacts"][s]] for s in ("r", "l")}
    t_sit, t_off_seat = m["t_sts0"], m["t_seatoff"]
    ch = m["chair"]
    G = {p + k: np.zeros(n) for _, _, p in LOAD_BODIES
         for k in ("ground_force_vx", "ground_force_vy", "ground_force_vz", "ground_force_px", "ground_force_py",
                   "ground_force_pz", "ground_torque_x", "ground_torque_y", "ground_torque_z")}
    resid = np.zeros(n)
    for i in range(n):
        Fx, Fy, M = need[i]
        O = geo["pelvis"][i, :2]
        ti = t[i]
        # sedák: podíl svislé síly z momentové rovnováhy (sedák pod stehny / chodidla), do odlehnutí
        Fc = np.array([0.0, 0.0])
        pc = np.array([geo["femur_r"][i, 0] + 0.10, ch["seat_h"]])
        if ti < t_off_seat and Fy > 1.0:
            xf = np.mean([geo["calcn_" + s][i, 0] + 0.11 for s in ("r", "l")])
            # (pc_x - Ox) Fyc + (xf - Ox)(Fy - Fyc) - (0 - Oy) Fx = M  ->  Fyc
            fyc = (M + (0 - O[1]) * Fx - (xf - O[0]) * Fy) / (pc[0] - xf)
            ramp = 1 - smoothstep((ti - (t_off_seat - 0.15)) / 0.15)
            Fc[1] = float(np.clip(fyc, 0.0, 0.95 * Fy)) * ramp
        Mc = (pc[0] - O[0]) * Fc[1] - (pc[1] - O[1]) * Fc[0]
        # chodidla
        act = []
        for s in ("r", "l"):
            for a, b, surf in contacts[s]:
                if a <= ti <= b:
                    act.append((s, a, b, surf))
        share = {}
        if len(act) == 1:
            share[act[0][0]] = 1.0
        elif len(act) == 2:
            act.sort(key=lambda c: c[1])
            (st_, at, bt, _), (sl, al, bl, _) = act
            if al < -1e8:                                          # oba stojí od začátku (sed, stoj)
                share = {st_: 0.5, sl: 0.5}
            else:
                w = np.interp((ti - al) / max(bt - al, 1e-6), DS_S, DS_W)
                share = {st_: w, sl: 1 - w}
        Ff = np.array([Fx, Fy]) - Fc
        F = {s: sh * Ff for s, sh in share.items()}
        rng, py, ctr = {}, {}, {}
        for s, a, b, surf in act:
            pose = geo["calcn_" + s][i]
            pts = pose[:2] + np.array([HEEL, MTP, TOE]) @ rot2(pose[2]).T
            heel_up = pts[0, 1] > surf + 0.01
            rng[s] = (pts[1, 0] - 0.02 if heel_up else pts[0, 0], pts[2, 0])
            py[s] = surf
            ctr[s] = 0.5 * sum(rng[s])
        # společný posun působišť u od středů opory: sum((x_s + u - Ox) Fy_s - (y_s - Oy) Fx_s) = M - Mc
        Mr = M - Mc
        fy_sum = sum(F[s][1] for s in F)
        cop = dict(ctr)
        if F and fy_sum > 1.0:
            base = sum((ctr[s] - O[0]) * F[s][1] - (py[s] - O[1]) * F[s][0] for s in F)
            u = (Mr - base) / fy_sum
            cop = {s: float(np.clip(ctr[s] + u, *rng[s])) for s in F}
        resid[i] = Mr - sum((cop[s] - O[0]) * F[s][1] - (py[s] - O[1]) * F[s][0] for s in F)
        if Fc[1] > 1.0 and abs(resid[i]) > 1e-6:
            # zbytek momentu vyrovná posun působiště na sedáku (stehna leží od kyčle ke kolenům)
            x_new = np.clip(pc[0] + resid[i] / Fc[1], geo["femur_r"][i, 0] - 0.05, geo["femur_r"][i, 0] + 0.30)
            resid[i] -= (x_new - pc[0]) * Fc[1]
            pc = np.array([x_new, pc[1]])
        for s, p_ in (("r", ""), ("l", "1_")):
            if s in F:
                G[p_ + "ground_force_vx"][i], G[p_ + "ground_force_vy"][i] = F[s]
                G[p_ + "ground_force_px"][i], G[p_ + "ground_force_py"][i] = cop[s], py[s]
                G[p_ + "ground_force_pz"][i] = 0.0944 if s == "r" else -0.0944
            else:
                G[p_ + "ground_force_px"][i] = geo["calcn_" + s][i, 0]
                G[p_ + "ground_force_py"][i] = geo["calcn_" + s][i, 1]
        G["2_ground_force_vy"][i] = Fc[1]
        G["2_ground_force_px"][i], G["2_ground_force_py"][i] = pc
    write_sto(np_table(t, G), GRF)
    external_loads(LOADS, GRF)
    print(f"Nezařazený moment (působiště mimo chodidlo): RMS {np.sqrt(np.mean(resid ** 2)):.1f} Nm, "
          f"max {np.abs(resid).max():.0f} Nm")
    print("ID s reakčními silami (pasivní exo) ...")
    _, d1 = run_id("ID_exo_vypnuty", MODELS["pasivni"], LOADS)
    for nm, a, b in phase_bounds(m):
        sel = (t >= a) & (t < b)
        print(f"  {nm:15s} zbytek pánve RMS: " + ", ".join(
            f"{c.split('_')[1]} {np.sqrt(np.mean(d1[c][sel] ** 2)):.1f}" for c in PELVIS))
    np.save(os.path.join(DATA, "grf_zbytek_momentu.npy"), np.column_stack([t, resid]))


if __name__ == "__main__":
    main()
