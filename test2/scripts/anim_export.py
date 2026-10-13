"""Animace, krok A (Python 3.8 + OpenSim): polohy a natočení všech těles pro každý snímek.

Výstup results/animace/snimky.npz + sceny.json (seznam sítí, schody, signály pro grafy).
Krok B (vykreslení a video) je anim_render.py.
"""
import json
import re
import xml.etree.ElementTree as ET

from osimtools import *  # noqa: F401,F403
import build_exo_model as BM

FPS = 25
OUT = os.path.join(ROOT, "results", "animace")
OSIM_GEOM = r"C:\OpenSim 4.5\Geometry"


def stairs_from_osim(path):
    root = ET.parse(path).getroot()
    out = []
    for fr in root.iter("PhysicalOffsetFrame"):
        if not fr.get("name", "").startswith("stair_"):
            continue
        tr = [float(v) for v in fr.find("translation").text.split()]
        hl = [float(v) for v in fr.find(".//half_lengths").text.split()]
        out.append({"center": tr, "half": hl})
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    model = osim.Model(BM.MAIN_MODEL)
    s = model.initSystem()
    cs = model.getCoordinateSet()
    bs = model.getBodySet()
    bodies = [bs.get(i).getName() for i in range(bs.getSize())]

    meshes = []
    for bname in bodies:
        b = bs.get(bname)
        for i in range(b.getPropertyByName("attached_geometry").size()):
            g = b.get_attached_geometry(i)
            m = osim.Mesh.safeDownCast(g)
            if m is None:
                continue
            f = m.get_mesh_file()
            path = (os.path.join(ROOT, "model", "Geometry", f) if f.endswith(".stl")
                    else os.path.join(OSIM_GEOM, f))
            col = [g.get_Appearance().get_color().get(k) for k in range(3)]
            if not f.endswith(".stl"):
                col = [0.92, 0.89, 0.80]
            meshes.append({"body": bname, "file": path,
                           "scale": [g.get_scale_factors().get(k) for k in range(3)], "color": col})

    kc, kd = read_mot(os.path.join(ROOT, "data", "walk_stairs_ik.mot"))
    t = kd[:, 0]
    tf = np.arange(t[0], t[-1], 1.0 / FPS)
    Q = np.column_stack([np.interp(tf, t, kd[:, 1 + j]) for j in range(len(COORDS))])
    R = np.zeros((len(tf), len(bodies), 3, 3))
    P = np.zeros((len(tf), len(bodies), 3))
    for i, row in enumerate(Q):
        for j, c in enumerate(COORDS):
            cs.get(c).setValue(s, np.radians(row[j]) if c in ROT else row[j], False)
        model.realizePosition(s)
        for k, bname in enumerate(bodies):
            T = bs.get(bname).getTransformInGround(s)
            R[i, k] = [[T.R().get(a, b) for b in range(3)] for a in range(3)]
            P[i, k] = [T.p().get(a) for a in range(3)]

    gc, gd = read_mot(os.path.join(ROOT, "data", "walk_stairs_grf_exo.mot"))
    grf = {}
    for side, p in (("r", ""), ("l", "1_")):
        cols = [p + f"ground_force_v{a}" for a in "xyz"] + [p + f"ground_force_p{a}" for a in "xyz"]
        grf[side] = np.column_stack([np.interp(tf, gd[:, 0], gd[:, gc.index(c)]) for c in cols])

    # signály pro grafy: řízení CNN (run_moje_rizeni.py cnn_rizeni)
    csv = os.path.join(ROOT, "results", "moje_rizeni", "cnn_rizeni", "prubeh.csv")
    lines = open(csv, encoding="utf-8").read().splitlines()
    head = lines[0].split(";")
    D = np.array([[float(x) for x in l.split(";")[1:]] for l in lines[1:]])
    col = {h: D[:, i - 1] for i, h in enumerate(head) if i > 0}
    r = np.load(os.path.join(ROOT, "results", "exo_vysledky.npz"))
    sig = {"t": col["cas_s"], "flexe": col["koleno_flexe_deg"], "on": col["motor_zapnut"],
           "tau": col["moment_motoru_Nm"], "M_bez": r["M_bez"][:len(D)], "M_clovek": col["moment_kolene_clovek_Nm"]}
    meta = json.load(open(os.path.join(ROOT, "data", "meta.json")))
    np.savez(os.path.join(OUT, "snimky.npz"), t=tf, R=R, P=P, grf_r=grf["r"], grf_l=grf["l"],
             pelvis=P[:, bodies.index("pelvis")], **{"sig_" + k: v for k, v in sig.items()})
    json.dump({"fps": FPS, "bodies": bodies, "meshes": meshes, "stairs": stairs_from_osim(BM.MAIN_MODEL),
               "t_schody": meta["t_k"]["2"], "t_rovina_konec": 7 * meta["Tc"]},
              open(os.path.join(OUT, "scena.json"), "w"), indent=1)
    print(f"{len(tf)} snímků, {len(meshes)} sítí -> {OUT}")


if __name__ == "__main__":
    main()
