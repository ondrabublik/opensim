"""Krok 2: model z test1 + exoskelet pravého kolene + řízení.

* Najde pevnou osu kloubu exoskeletu, která nejlépe sedí na pohyblivou osu kolene
  (koleno v modelu se při flexi posouvá podle splajnů) a vyčíslí nesouosost.
* Přidá dvě tělesa exoskeletu (článek stehna, článek bérce) s hmotností a setrvačností
  z CAD, přivařená k femur_r a tibia_r (objímky pevně na končetině), s geometrií STL.
* Přidá řízení jako ExpressionBasedCoordinateForce na knee_angle_r pro oba režimy;
  aktivní je jen exo_params.CONTROL_MODE.

Výstupy: model/subject01_stairs_exo.osim (+ varianty pro analýzu), cad/osa_kloubu.json
"""
import json

from osimtools import *  # noqa: F401,F403
import exo_control as X
import exo_params as E

CAD = os.path.join(ROOT, "cad")
MODEL_DIR = os.path.join(ROOT, "model")
MAIN_MODEL = os.path.join(MODEL_DIR, "subject01_stairs_exo.osim")
TABLE_FILE = os.path.join(ROOT, "results", "rizeni_tabulka.json")


def variant_path(mode):
    """mode: None = pasivní exo (pohon bez momentu), jinak režim řízení."""
    return os.path.join(MODEL_DIR, f"subject01_stairs_exo_{mode or 'pasivni'}.osim")


def fit_hinge(model_path=E.BASE_MODEL):
    """Pevná osa exo: bod p ve femur_r a jeho obraz c v tibia_r tak, aby
    sum |R_i^T (p - d_i) - c|^2 přes rozsah flexe bylo minimální (lineární LSQ)."""
    model = osim.Model(model_path)
    s = model.initSystem()
    knee = model.getCoordinateSet().get(f"knee_angle_{E.SIDE}")
    fem = model.getBodySet().get(f"femur_{E.SIDE}")
    tib = model.getBodySet().get(f"tibia_{E.SIDE}")
    th = np.linspace(*E.HINGE_FIT_RANGE_DEG, 111)
    Rs, ds = [], []
    for a in th:
        knee.setValue(s, -np.radians(a))
        model.realizePosition(s)
        X_ft = tib.findTransformBetween(s, fem)        # tibia vyjádřená ve femuru
        R = np.array([[X_ft.R().get(i, j) for j in range(3)] for i in range(3)])[:2, :2]
        Rs.append(R)
        ds.append(np.array([X_ft.p().get(0), X_ft.p().get(1)]))
    A = np.vstack([np.hstack([R.T, -np.eye(2)]) for R in Rs])
    b = np.concatenate([R.T @ d for R, d in zip(Rs, ds)])
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    p, c = sol[:2], sol[2:]
    err = np.array([np.linalg.norm(R.T @ (p - d) - c) for R, d in zip(Rs, ds)])
    # pro srovnání: osa v počátku tibie při natažení (bez optimalizace)
    p0 = ds[0]
    err0 = np.array([np.linalg.norm(R.T @ (p0 - d)) for R, d in zip(Rs, ds)])
    return {"femur_point": p.tolist(), "tibia_point": c.tolist(), "theta_deg": th.tolist(),
            "misalignment_mm": (err * 1000).tolist(), "misalignment_naive_mm": (err0 * 1000).tolist()}


def load_table():
    if E.CTRL_TABLE_MANUAL is not None:
        return {m: list(E.CTRL_TABLE_MANUAL) for m in E.CONTROL_MODES}
    if os.path.exists(TABLE_FILE):
        return json.load(open(TABLE_FILE))["table"]
    return {m: [0.0] * len(X.NODES) for m in E.CONTROL_MODES}


def build(active_mode, tables, hinge, out_path):
    props = json.load(open(os.path.join(CAD, "exo_hmotnosti.json")))
    model = osim.Model(E.BASE_MODEL)
    model.setName("subject01_stairs_exo")
    bs = model.getBodySet()
    pf, pt = hinge["femur_point"], hinge["tibia_point"]
    for name, parent, p in (("exo_thigh_r", "femur_r", pf), ("exo_shank_r", "tibia_r", pt)):
        pr = props[name]
        I = pr["inertia"]
        body = osim.Body(name, pr["mass"], osim.Vec3(*pr["mass_center"]),
                         osim.Inertia(I[0], I[1], I[2], I[3], I[4], I[5]))
        for part in pr["parts"]:
            mesh = osim.Mesh(part["file"])
            mesh.setName(part["name"])
            mesh.setColor(osim.Vec3(*part["color"]))
            body.attachGeometry(mesh)
        model.addBody(body)
        joint = osim.WeldJoint(f"{name}_upevneni", bs.get(parent), osim.Vec3(p[0], p[1], 0.0),
                               osim.Vec3(0, 0, 0), body, osim.Vec3(0, 0, 0), osim.Vec3(0, 0, 0))
        model.addJoint(joint)
    for mode in E.CONTROL_MODES:
        f = osim.ExpressionBasedCoordinateForce(f"knee_angle_{E.SIDE}",
                                                X.lepton_expression(tables[mode], mode))
        f.setName(X.FORCE_NAME[mode])
        f.set_appliesForce(mode == active_mode)
        model.addForce(f)
    model.finalizeConnections()
    model.printToXML(out_path)
    return out_path


def main():
    hinge = fit_hinge()
    json.dump(hinge, open(os.path.join(CAD, "osa_kloubu.json"), "w"), indent=1)
    mis, naive = np.array(hinge["misalignment_mm"]), np.array(hinge["misalignment_naive_mm"])
    print(f"Osa exo ve femur_r: {np.round(np.array(hinge['femur_point']) * 1000, 1)} mm, "
          f"v tibia_r: {np.round(np.array(hinge['tibia_point']) * 1000, 1)} mm")
    print(f"Nesouosost exo/koleno 0-110°: max {mis.max():.1f} mm, RMS {np.sqrt(np.mean(mis**2)):.1f} mm "
          f"(osa v počátku tibie: max {naive.max():.1f} mm)")
    tables = load_table()
    build(E.CONTROL_MODE, tables, hinge, MAIN_MODEL)
    for mode in (None,) + tuple(E.CONTROL_MODES):
        build(mode, tables, hinge, variant_path(mode))
    print("Model:", MAIN_MODEL)


if __name__ == "__main__":
    main()
