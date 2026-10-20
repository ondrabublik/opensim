"""Krok 6: přehrání chůze v Simbody visualizeru (OpenSim API) s kamerou sledující model.

    python s6_vizualizace.py [--rychlost 0.5] [--od 0] [--do 20] [--snimku N]

Otevře okno OpenSim/Simbody visualizeru s modelem s exoskeletem a schodištěm. Kamera se
v každém snímku přesune vedle pánve (z pravé strany, mírně zezadu a shora) a míří na ni.
Čas, úsek a stav motoru (CNN) se vypisují do konzole. Video lze nahrát přímo ve visualizeru
(menu File -> Save Movie), snímky se uloží jako PNG.
"""
import argparse
import json
import time

from t3common import *  # noqa: F401,F403

OSIM_GEOM = r"C:\OpenSim 4.5\Geometry"
OSIM_BIN = r"C:\OpenSim 4.5\bin"                 # zde je simbody-visualizer.exe
os.environ["PATH"] = OSIM_BIN + os.pathsep + os.environ.get("PATH", "")
# Simbody dělí PATH i podle ':' a ztratí písmeno disku ("\OpenSim 4.5\bin" bere z aktuálního
# disku) -> pracovní adresář musí být na stejném disku jako OpenSim.
os.chdir(OSIM_BIN)
CAM_OFFSET = np.array([-1.3, 0.55, 2.6])     # kamera vůči pánvi: za, nad, vpravo [m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rychlost", type=float, default=0.5, help="násobek reálného času")
    ap.add_argument("--od", type=float, default=0.0)
    ap.add_argument("--do", type=float, default=1e9)
    ap.add_argument("--snimku", type=int, default=0, help="jen prvních N snímků (test)")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    osim.ModelVisualizer.addDirToGeometrySearchPaths(OSIM_GEOM)
    osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(MODEL_DIR, "Geometry"))
    model = osim.Model(MAIN_MODEL)
    model.setUseVisualizer(True)
    state = model.initSystem()
    viz = model.getVisualizer()
    sviz = viz.getSimbodyVisualizer()
    sviz.setBackgroundType(sviz.SolidColor)
    sviz.setBackgroundColor(osim.Vec3(0.98, 0.98, 0.97))
    sviz.setShowSimTime(False)

    t, q = table_np(KIN)
    _, ctrl = table_np(os.path.join(RES, "cnn_rizeni.sto"))
    meta = json.load(open(META))
    cs = model.getCoordinateSet()
    coords = [cs.get(i) for i in range(cs.getSize())]
    rot = {c.getName(): c.getMotionType() == 1 for c in coords}       # 1 = Rotational
    pelvis = model.getBodySet().get("pelvis")
    tf = np.arange(max(t[0], a.od), min(t[-1], a.do), 1.0 / a.fps)
    if a.snimku:
        tf = tf[:a.snimku]
    t_wall = time.time()
    last_on = None
    for k, tk in enumerate(tf):
        for c in coords:
            if c.getName() in q:
                v = np.interp(tk, t, q[c.getName()])
                c.setValue(state, np.radians(v) if rot[c.getName()] else v, False)
        model.realizePosition(state)
        p = pelvis.getPositionInGround(state)
        P = np.array([p.get(0), p.get(1), p.get(2)])
        cam = P + CAM_OFFSET
        sviz.setCameraTransform(osim.Transform(osim.Vec3(*cam)))
        sviz.pointCameraAt(osim.Vec3(P[0] + 0.25, P[1] - 0.35, 0.0), osim.Vec3(0, 1, 0))
        on = np.interp(tk, t[:len(ctrl["motor_zapnut"])], ctrl["motor_zapnut"]) > 0.5
        usek = "rovina" if tk < 7 * meta["Tc"] else ("prechod" if tk < meta["t_k"]["2"] else "schody")
        if k % int(a.fps / 2) == 0 or on != last_on:      # stav motoru do konzole
            print(f"t = {tk:5.2f} s | {usek:7s} | motor exo: {'ZAPNUT' if on else 'vypnut'}", flush=True)
        last_on = on
        viz.show(state)
        # přehrávání v (zpomaleném) reálném čase
        wait = t_wall + (tk - tf[0]) / a.rychlost - time.time()
        if wait > 0:
            time.sleep(wait)
    print(f"Přehráno {len(tf)} snímků.")


if __name__ == "__main__":
    main()
