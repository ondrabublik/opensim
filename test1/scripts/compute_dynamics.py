"""Krok 2: reakční síly od země a inverzní dynamika (OpenSim InverseDynamicsTool).

Rovina: naměřené GRF z ukázkových dat OpenSim.
Schody: GRF dopočtené z dynamiky celého těla - inverzní dynamika bez vnějších sil
dá síly/moment, které musí dodat země (zbytková síla na pánvi). Ty se rozdělí mezi
chodidla (ve dvojí opoře hladkým přechodem) a působiště (COP) se dopočte z momentové
rovnováhy. Metoda je ověřena na rovinném úseku proti naměřeným GRF (validate_*).
"""
import json

from osimtools import *  # noqa: F401,F403
import params as P
from build_motion import GRF_COLS, PIVOT

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
SETUP = os.path.join(ROOT, "setup")
MODEL = os.path.join(ROOT, "model", "subject01_stairs.osim")
for d in (RES, SETUP):
    os.makedirs(d, exist_ok=True)
SIDES = ("r", "l")
# podíl svislé síly zadní nohy během dvojí opory (naměřeno na rovině, s = 0..1)
DS_SHARE_S = np.linspace(0, 1, 11)
DS_SHARE_W = np.array([1.0, 0.91, 0.82, 0.73, 0.645, 0.54, 0.415, 0.285, 0.17, 0.085, 0.0])


def write_external_loads(path, grf_file):
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
\t<ExternalLoads name="GRF">
\t\t<objects>
\t\t\t<ExternalForce name="right">
\t\t\t\t<applied_to_body>calcn_r</applied_to_body>
\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>
\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>
\t\t\t\t<force_identifier>ground_force_v</force_identifier>
\t\t\t\t<point_identifier>ground_force_p</point_identifier>
\t\t\t\t<torque_identifier>ground_torque_</torque_identifier>
\t\t\t</ExternalForce>
\t\t\t<ExternalForce name="left">
\t\t\t\t<applied_to_body>calcn_l</applied_to_body>
\t\t\t\t<force_expressed_in_body>ground</force_expressed_in_body>
\t\t\t\t<point_expressed_in_body>ground</point_expressed_in_body>
\t\t\t\t<force_identifier>1_ground_force_v</force_identifier>
\t\t\t\t<point_identifier>1_ground_force_p</point_identifier>
\t\t\t\t<torque_identifier>1_ground_torque_</torque_identifier>
\t\t\t</ExternalForce>
\t\t</objects>
\t\t<datafile>{grf_file}</datafile>
\t</ExternalLoads>
</OpenSimDocument>
"""
    with open(path, "w") as f:
        f.write(xml)


def run_id(name, kin_file, t0, t1, loads_xml=None):
    tool = osim.InverseDynamicsTool()
    tool.setName(name)
    tool.setModelFileName(MODEL)
    tool.setCoordinatesFileName(kin_file)
    tool.setLowpassCutoffFrequency(-1)          # kinematika už je vyfiltrovaná (6 Hz)
    tool.setStartTime(t0)
    tool.setEndTime(t1)
    if loads_xml:
        tool.setExternalLoadsFileName(loads_xml)
    tool.setResultsDir(RES)
    tool.setOutputGenForceFileName(name + ".sto")
    excl = osim.ArrayStr()
    excl.append("Muscles")
    tool.setExcludedForces(excl)
    setup = os.path.join(SETUP, f"setup_ID_{name}.xml")
    tool.printToXML(setup)
    osim.InverseDynamicsTool(setup).run()
    cols, d = read_mot(os.path.join(RES, name + ".sto"))
    return cols, d


def foot_geometry(kin, Q):
    """Póza pánve a obou chodidel pro každý snímek (z vyfiltrované kinematiky)."""
    pel = np.zeros((len(Q), 2))
    com = np.zeros((len(Q), 2))
    feet = {s: np.zeros((len(Q), 3)) for s in SIDES}
    for i, row in enumerate(Q):
        kin.set(dict(zip(COORDS, row)))
        pel[i] = kin.pos("pelvis")
        c = kin.model.calcMassCenterPosition(kin.state)
        com[i] = c.get(0), c.get(1)
        for s in SIDES:
            feet[s][i] = kin.foot_pose(s)
    return pel, feet, com


def support_range(pose, surface):
    pts = pose[:2] + np.array([HEEL, MTP, TOE]) @ rot2(pose[2]).T
    heel_up = pts[0, 1] > surface + 0.01
    lo = pts[1, 0] - 0.02 if heel_up else pts[0, 0]
    return lo, pts[2, 0]


def synthesize(t, Q_need, pel, feet, com, contacts):
    """Rozdělení potřebné vnější síly mezi chodidla.

    Q_need: (n,3) [tilt moment, Fx, Fy] z ID bez vnějších sil.
    Dvojí opora: svislá síla podle naměřeného průběhu DS_SHARE, vodorovná síla každé
    nohy míří k těžišti (zadní noha tlačí vpřed, přední brzdí) + společná korekce.
    contacts: {side: [(t_on, t_off, surface)]}
    Vrací pole GRF (n,18) ve formátu GRF_COLS a zbytkový moment, který nešlo umístit.
    """
    n = len(t)
    G = np.zeros((n, 18))
    resid = np.zeros(n)
    for i in range(n):
        M, Fx, Fy = Q_need[i]
        O = pel[i]
        act = []
        for s in SIDES:
            for (a, b, surf) in contacts[s]:
                if a <= t[i] <= b:
                    act.append((s, a, b, surf))
        if not act:
            continue
        if len(act) == 1:
            share = {act[0][0]: 1.0}
            trailing, leading = act[0][0], None
        else:
            act.sort(key=lambda c: c[1])                # dřívější nášlap = zadní noha
            (st, at, bt, _), (sl, al, bl, _) = act
            w = np.interp((t[i] - al) / max(bt - al, 1e-6), DS_SHARE_S, DS_SHARE_W)
            share = {st: w, sl: 1.0 - w}
            trailing, leading = st, sl
        surf = {c[0]: c[3] for c in act}
        rng = {s: support_range(feet[s][i], surf[s]) for s in share}
        F = {s: share[s] * np.array([Fx, Fy]) for s in share}
        if leading is not None:
            guess = {trailing: rng[trailing][1] - 0.03, leading: np.mean(rng[leading])}
            r = {s: (com[i, 0] - guess[s]) / max(com[i, 1] - surf[s], 0.3) for s in share}
            c = (Fx - sum(share[s] * Fy * r[s] for s in share)) / Fy
            F = {s: share[s] * Fy * np.array([r[s] + c, 1.0]) for s in share}
        cop = {}
        # momentová rovnováha k počátku pánve: sum((p - O) x F) = M
        def moment(s, px):
            return (px - O[0]) * F[s][1] - (surf[s] - O[1]) * F[s][0]
        if leading is None:
            s = trailing
            px = O[0] + (M + (surf[s] - O[1]) * F[s][0]) / F[s][1]
            cop[s] = np.clip(px, *rng[s])
        else:
            cop[trailing] = rng[trailing][1] - 0.03
            rem = M - moment(trailing, cop[trailing])
            if F[leading][1] > 1.0:
                px = O[0] + (rem + (surf[leading] - O[1]) * F[leading][0]) / F[leading][1]
                cop[leading] = np.clip(px, *rng[leading])
            else:
                cop[leading] = np.mean(rng[leading])
            rem = M - moment(leading, cop[leading])
            if F[trailing][1] > 1.0:
                px = O[0] + (rem + (surf[trailing] - O[1]) * F[trailing][0]) / F[trailing][1]
                cop[trailing] = np.clip(px, *rng[trailing])
        resid[i] = M - sum(moment(s, cop[s]) for s in share)
        for s in share:
            j = 0 if s == "r" else 6
            G[i, j:j + 6] = [F[s][0], F[s][1], 0.0, cop[s], surf[s],
                             0.0944 if s == "r" else -0.0944]
    return G, resid


def level_contacts(meta, t_end):
    ev = meta["level_events"]
    out = {}
    for s in SIDES:
        e = sorted((t, k) for (side, k, t) in ev if side == s)
        iv, t_on = [], None
        # na začátku (t=0, dopad pravé) je levá ve stoji
        if s == "l":
            t_on = -1.0
        for t, k in e:
            if k == "on":
                t_on = t
            elif t_on is not None:
                iv.append((t_on, t, 0.0))
                t_on = None
        if t_on is not None:
            iv.append((t_on, t_end, 0.0))
        out[s] = iv
    return out


def main():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    kin_file = os.path.join(DATA, "walk_stairs_ik.mot")
    cols, kd = read_mot(kin_file)
    t = kd[:, 0]
    Q = kd[:, 1:].copy()
    for j, c in enumerate(COORDS):
        if c in ROT:
            Q[:, j] = np.radians(Q[:, j])
    TA = meta["TA"]
    kin = Kin(MODEL)
    pel, feet, com = foot_geometry(kin, Q)

    # --- 1) ID bez vnějších sil => co musí dodat země
    print("ID bez vnějších sil ...")
    c0, d0 = run_id("ID_no_grf", kin_file, t[0], t[-1])
    Q_need = d0[:, [c0.index("pelvis_tilt_moment"), c0.index("pelvis_tx_force"),
                    c0.index("pelvis_ty_force")]]
    assert np.allclose(d0[:, 0], t)

    # --- 2) syntetické GRF pro přechod a schody
    contacts = {s: [] for s in SIDES}
    for s in SIDES:
        for c in meta["contacts"][s]:
            a = c["t_on"] if c["t_on"] is not None else TA - 1.0
            b = c["t_off"] if c["t_off"] is not None else t[-1] + 1.0
            contacts[s].append((max(a, TA - 1.0), b, c["surface"]))
    G_syn, resid = synthesize(t, Q_need, pel, feet, com, contacts)
    G_meas = np.load(os.path.join(DATA, "grf_measured_part.npy"))
    # plynulý přechod z naměřených na dopočtené síly během 50 ms před koncem měření
    w = smoothstep((t - (TA - 0.05)) / 0.05)[:, None]
    G = np.where(t[:, None] <= TA, (1 - w) * G_meas + w * G_syn, G_syn)
    grf_file = os.path.join(DATA, "walk_stairs_grf.mot")
    write_mot(grf_file, ["time"] + GRF_COLS, np.column_stack([t, G]), "walk_stairs_grf",
              in_degrees=False)
    loads = os.path.join(DATA, "walk_stairs_grf.xml")
    write_external_loads(loads, grf_file)
    st = t > TA
    print(f"Nezařazený moment (COP mimo chodidlo) na schodech: RMS {np.sqrt(np.mean(resid[st]**2)):.2f} Nm, "
          f"max {np.abs(resid[st]).max():.1f} Nm")

    # --- 3) finální ID s GRF
    print("ID s reakčními silami ...")
    c1, d1 = run_id("ID_walk_stairs", kin_file, t[0], t[-1], loads)
    for name in ("pelvis_tx_force", "pelvis_ty_force", "pelvis_tilt_moment"):
        r = d1[:, c1.index(name)]
        print(f"  zbytková {name}: rovina RMS {np.sqrt(np.mean(r[t <= TA]**2)):6.1f}, "
              f"schody RMS {np.sqrt(np.mean(r[st]**2)):6.1f}")

    # --- 4) validace metody na rovinném úseku
    lev = t <= TA
    G_val, _ = synthesize(t[lev], Q_need[lev], pel[lev], {s: feet[s][lev] for s in SIDES},
                          com[lev], level_contacts(meta, TA))
    G_val_full = np.vstack([G_val, G_syn[~lev]])
    vfile = os.path.join(DATA, "validation_level_grf.mot")
    write_mot(vfile, ["time"] + GRF_COLS, np.column_stack([t, G_val_full]),
              "validation_level_grf", in_degrees=False)
    vxml = os.path.join(DATA, "validation_level_grf.xml")
    write_external_loads(vxml, vfile)
    print("ID validace (syntetické GRF na rovině) ...")
    run_id("ID_validation_level", kin_file, t[0], TA, vxml)
    np.save(os.path.join(DATA, "grf_validation_level.npy"), G_val)
    print("Hotovo:", os.path.join(RES, "ID_walk_stairs.sto"))


if __name__ == "__main__":
    main()
