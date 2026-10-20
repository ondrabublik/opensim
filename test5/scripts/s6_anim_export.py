"""Krok 6: animace, část A (Python 3.8 + OpenSim): polohy a natočení těles pro každý snímek,
reakční síly a signály pro grafy. Vykreslení a video: s7_anim_render.py (Python 3.11)."""
import json

from t5common import *  # noqa: F401,F403

FPS = 25
OUT = os.path.join(RES, "animace")
OSIM_GEOM = r"C:\OpenSim 4.5\Geometry"


def main():
    os.makedirs(OUT, exist_ok=True)
    m = meta()
    model = osim.Model(MAIN_MODEL)
    s = model.initSystem()
    cs = model.getCoordinateSet()
    bs = model.getBodySet()
    bodies = [bs.get(i).getName() for i in range(bs.getSize())]
    meshes = []
    for bname in bodies:
        b = bs.get(bname)
        for i in range(b.getPropertyByName("attached_geometry").size()):
            g = b.get_attached_geometry(i)
            mesh = osim.Mesh.safeDownCast(g)
            if mesh is None:
                continue
            f = mesh.get_mesh_file()
            stl = f.endswith(".stl")
            meshes.append({"body": bname,
                           "file": os.path.join(MODEL_DIR, "Geometry", f) if stl else os.path.join(OSIM_GEOM, f),
                           "scale": [g.get_scale_factors().get(k) for k in range(3)],
                           "color": [g.get_Appearance().get_color().get(k) for k in range(3)] if stl
                           else [0.92, 0.89, 0.80]})
    t, Q = table_np(KIN)
    rot = {cs.get(i).getName(): cs.get(i).getMotionType() == 1 for i in range(cs.getSize())}
    tf = np.arange(t[0], t[-1], 1.0 / FPS)
    R = np.zeros((len(tf), len(bodies), 3, 3))
    P = np.zeros((len(tf), len(bodies), 3))
    for i, ti in enumerate(tf):
        for c, v in Q.items():
            x = float(np.interp(ti, t, v))
            cs.get(c).setValue(s, np.radians(x) if rot[c] else x, False)
        model.realizePosition(s)
        for k, bname in enumerate(bodies):
            X = bs.get(bname).getTransformInGround(s)
            R[i, k] = [[X.R().get(a, b) for b in range(3)] for a in range(3)]
            P[i, k] = [X.p().get(a) for a in range(3)]
    tg, G = table_np(GRF)
    grf = {}
    for side, p in (("r", ""), ("l", "1_"), ("ch", "2_")):
        cols = [p + f"ground_force_v{a}" for a in "xyz"] + [p + f"ground_force_p{a}" for a in "xyz"]
        grf[side] = np.column_stack([np.interp(tf, tg, G[c]) for c in cols])
    tc, c = table_np(os.path.join(RES, "cnn_rizeni.sto"))
    _, off = table_np(os.path.join(RES, "ID_exo_vypnuty.sto"))
    _, on = table_np(os.path.join(RES, "ID_exo_zapnuty.sto"))
    n = min(len(tc), len(off[KNEE + "_moment"]))
    np.savez(os.path.join(OUT, "snimky.npz"), t=tf, R=R, P=P, grf_r=grf["r"], grf_l=grf["l"], grf_ch=grf["ch"],
             pelvis=P[:, bodies.index("pelvis")], sig_t=tc[:n], sig_flexe=c["koleno_flexe_deg"][:n],
             sig_w_st=c["omega_stehno_deg_s"][:n], sig_w_ho=c["omega_holen_deg_s"][:n],
             sig_on=c["motor_zapnut"][:n], sig_tau=c["exo_motor_Nm"][:n],
             sig_M_off=off[KNEE + "_moment"][:n], sig_M_on=on[KNEE + "_moment"][:n])
    json.dump({"fps": FPS, "bodies": bodies, "meshes": meshes, "terrain": m["terrain"], "chair": m["chair"],
               "phases": phase_bounds(m)}, open(os.path.join(OUT, "scena.json"), "w"), indent=1)
    print(f"{len(tf)} snímků, {len(meshes)} sítí -> {OUT}")


if __name__ == "__main__":
    main()
