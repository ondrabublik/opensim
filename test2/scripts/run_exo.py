"""Krok 3: inverzní dynamika s exoskeletem, ladění řízení a ověření v OpenSim.

1. Reakční síly země pro model s exo: GRF z test1 + navíc síla potřebná na nesení
   a zrychlování hmoty exoskeletu (rozdíl ID bez vnějších sil: model s exo - bez exo),
   rozdělená mezi chodidla v poměru jejich svislých sil.
2. ID: bez exo (= test1), pasivní exo (jen hmota, pohon bez momentu).
3. Ladění křivky G(theta) pro oba režimy řízení (cíl: exo dodá ASSIST_RATIO momentu
   kolene s pasivním exo, max TAU_MAX) na 6 cyklech roviny a 6 cyklech schodů.
4. Přestavba modelů s vyladěným řízením a ID s aktivním exo. Moment exo se v ID
   počítá přímo z ExpressionBasedCoordinateForce v modelu; kontrola proti Pythonu.
"""
import json

from osimtools import *  # noqa: F401,F403
import build_exo_model as BM
import exo_control as X
import exo_params as E

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
SETUP = os.path.join(ROOT, "setup")
KIN = os.path.join(DATA, "walk_stairs_ik.mot")
GRF_BASE = os.path.join(DATA, "walk_stairs_grf.mot")
GRF_EXO = os.path.join(DATA, "walk_stairs_grf_exo.mot")
K = f"knee_angle_{E.SIDE}_moment"
PELVIS = ["pelvis_tx_force", "pelvis_ty_force", "pelvis_tilt_moment"]


def write_external_loads(path, grf_file):
    sides = (("right", "calcn_r", ""), ("left", "calcn_l", "1_"))
    objs = "".join(f"""
\t\t\t<ExternalForce name="{n}">
\t\t\t\t<applied_to_body>{b}</applied_to_body>
\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>
\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>
\t\t\t\t<force_identifier>{p}ground_force_v</force_identifier>
\t\t\t\t<point_identifier>{p}ground_force_p</point_identifier>
\t\t\t\t<torque_identifier>{p}ground_torque_</torque_identifier>
\t\t\t</ExternalForce>""" for n, b, p in sides)
    with open(path, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
\t<ExternalLoads name="GRF">
\t\t<objects>{objs}
\t\t</objects>
\t\t<datafile>{grf_file}</datafile>
\t</ExternalLoads>
</OpenSimDocument>
""")
    return path


def run_id(name, model, t0, t1, loads=None):
    tool = osim.InverseDynamicsTool()
    tool.setName(name)
    tool.setModelFileName(model)
    tool.setCoordinatesFileName(KIN)
    tool.setLowpassCutoffFrequency(-1)          # kinematika z test1 je už vyfiltrovaná (6 Hz)
    tool.setStartTime(t0)
    tool.setEndTime(t1)
    if loads:
        tool.setExternalLoadsFileName(loads)
    tool.setResultsDir(RES)
    tool.setOutputGenForceFileName(name + ".sto")
    excl = osim.ArrayStr()
    excl.append("Muscles")
    tool.setExcludedForces(excl)
    setup = os.path.join(SETUP, f"setup_ID_{name}.xml")
    tool.printToXML(setup)
    osim.InverseDynamicsTool(setup).run()
    return read_mot(os.path.join(RES, name + ".sto"))


def cycles(meta):
    """6 cyklů roviny a 6 ustálených cyklů na schodech (pravá noha, dopad -> dopad)."""
    Tc = meta["Tc"]
    tk = {int(k): v for k, v in meta["t_k"].items()}
    lev = [(i * Tc, (i + 1) * Tc) for i in range(1, 7)]
    sta = [(tk[k], tk[k + 2]) for k in range(4, 15, 2)]
    return lev, sta


def in_cycles(t, cyc):
    return np.any([(t >= a) & (t < b) for a, b in cyc], axis=0)


def main():
    for d in (RES, SETUP):
        os.makedirs(d, exist_ok=True)
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    _, kd = read_mot(KIN)
    t = kd[:, 0]
    t0, t1 = t[0], t[-1]
    m_base = BM.variant_path(None)
    if not os.path.exists(m_base):
        BM.main()

    # ---------------- 1) GRF pro model s exo
    print("ID bez vnějších sil (bez exo / s exo) ...")
    cb, db = run_id("ID_noGRF_bez_exo", E.BASE_MODEL, t0, t1)
    ce, de = run_id("ID_noGRF_exo", m_base, t0, t1)
    n = min(len(t), len(db), len(de))
    dQ = de[:n, [ce.index(c) for c in PELVIS]] - db[:n, [cb.index(c) for c in PELVIS]]
    gc, gd = read_mot(GRF_BASE)
    G = gd[:n].copy()
    fy_r, fy_l = G[:, gc.index("ground_force_vy")], G[:, gc.index("1_ground_force_vy")]
    share_r = np.clip(fy_r, 0, None) / np.maximum(np.clip(fy_r, 0, None) + np.clip(fy_l, 0, None), 1e-6)
    for p, sh in (("", share_r), ("1_", 1 - share_r)):
        G[:, gc.index(p + "ground_force_vx")] += sh * dQ[:, 0]
        G[:, gc.index(p + "ground_force_vy")] += sh * dQ[:, 1]
    write_mot(GRF_EXO, gc, G, "walk_stairs_grf_exo", in_degrees=False)
    loads_base = write_external_loads(os.path.join(DATA, "walk_stairs_grf.xml"), GRF_BASE)
    loads_exo = write_external_loads(os.path.join(DATA, "walk_stairs_grf_exo.xml"), GRF_EXO)
    mass_exo = sum(json.load(open(os.path.join(ROOT, "cad", "exo_hmotnosti.json")))[k]["mass"]
                   for k in ("exo_thigh_r", "exo_shank_r"))
    print(f"  přídavná svislá síla od exo: průměr {dQ[:, 1].mean():.1f} N (tíha exo {mass_exo * 9.81:.1f} N)")

    # ---------------- 2) ID bez exo a s pasivním exo
    print("ID bez exo a s pasivním exo ...")
    c0, d0 = run_id("ID_bez_exo", E.BASE_MODEL, t0, t1, loads_base)
    cp, dp = run_id("ID_exo_pasivni", m_base, t0, t1, loads_exo)
    n = min(n, len(d0), len(dp))
    # zbylý přírůstek momentu na pánvi (pelvis_tilt) se vyrovná posunem působiště obou
    # sil o dx = dM / Fy (vnější moment k pánvi roste o dx*Fy, zbytkový o tolik klesá)
    j0, jp = c0.index(PELVIS[2]), cp.index(PELVIS[2])
    fy = sum(np.clip(G[:n, gc.index(p + "ground_force_vy")], 0, None) for p in ("", "1_"))
    dx = np.where(fy > 50.0, (dp[:n, jp] - d0[:n, j0]) / np.maximum(fy, 50.0), 0.0)
    for p in ("", "1_"):
        G[:n, gc.index(p + "ground_force_px")] += dx
    print(f"  posun působiště GRF kvůli exo: max {np.abs(dx).max() * 1000:.1f} mm, "
          f"RMS {np.sqrt(np.mean(dx ** 2)) * 1000:.1f} mm")
    write_mot(GRF_EXO, gc, G, "walk_stairs_grf_exo", in_degrees=False)
    cp, dp = run_id("ID_exo_pasivni", m_base, t0, t1, loads_exo)
    t = t[:n]
    tc1, d1 = read_mot(os.path.join(E.TEST1, "results", "ID_walk_stairs.sto"))
    print(f"  kontrola proti test1: max rozdíl momentu kolene "
          f"{np.abs(d0[:n, c0.index(K)] - d1[:n, tc1.index(K)]).max():.2e} Nm")
    for name, c, d in (("bez exo", c0, d0), ("pasivní exo", cp, dp)):
        print(f"  zbytkové síly pánve ({name}): " + ", ".join(
            f"{p} RMS {np.sqrt(np.mean(d[:n, c.index(p)] ** 2)):.1f}" for p in PELVIS) + " (celých 15 m)")

    # ---------------- 3) ladění řízení
    knee = np.radians(kd[:n, 1 + COORDS.index(f"knee_angle_{E.SIDE}")])
    theta = -np.degrees(knee)                       # flexe [deg]
    omega = np.gradient(knee, t)                    # rychlost extenze [rad/s]
    M_pas = dp[:n, cp.index(K)]
    lev, sta = cycles(meta)
    mask = in_cycles(t, lev) | in_cycles(t, sta)
    tables = {}
    for mode in E.CONTROL_MODES:
        if E.CTRL_TABLE_MANUAL is not None:
            tables[mode] = list(map(float, E.CTRL_TABLE_MANUAL))
        else:
            w = np.where(in_cycles(t, lev)[mask], E.LEVEL_WEIGHT, 1.0)
            tables[mode] = X.tune(theta[mask], omega[mask], E.ASSIST_RATIO * M_pas[mask], mode,
                                  weights=w).tolist()
        print(f"  G[{mode}] (Nm) @ {E.CTRL_NODES_DEG[0]}..{E.CTRL_NODES_DEG[-1]} deg: "
              + " ".join(f"{g:.0f}" for g in tables[mode]))
    json.dump({"nodes_deg": E.CTRL_NODES_DEG, "table": tables, "tau_max": E.TAU_MAX,
               "assist_ratio": E.ASSIST_RATIO, "omega0": E.OMEGA0,
               "expression": {m: X.lepton_expression(tables[m], m) for m in E.CONTROL_MODES}},
              open(BM.TABLE_FILE, "w"), indent=1)

    # ---------------- 4) modely s vyladěným řízením + ID
    BM.main()
    out = {"t": t, "theta": theta, "omega": omega, "M_bez": d0[:n, c0.index(K)], "M_pas": M_pas,
           "mask": mask}
    for mode in E.CONTROL_MODES:
        print(f"ID s řízením '{mode}' ...")
        cm, dm = run_id(f"ID_exo_{mode}", BM.variant_path(mode), t0, t1, loads_exo)
        M = dm[:n, cm.index(K)]
        tau_osim = M_pas - M
        tau_py = X.torque(theta, omega, tables[mode], mode)
        print(f"  moment exo z OpenSim vs. Python: max rozdíl {np.abs(tau_osim - tau_py).max():.2f} Nm "
              f"(špička {np.abs(tau_osim).max():.1f} Nm)")
        out[f"M_{mode}"] = M
        out[f"tau_{mode}"] = tau_osim
    np.savez(os.path.join(RES, "exo_vysledky.npz"), **out)
    print("Hotovo.")


if __name__ == "__main__":
    main()
