"""Krok 1: model s exoskeletem sestavený přes OpenSim API.

* osa kloubu exo: transformace femur -> tibia z API (Frame.findTransformBetween) pro flexi
  0-110°, pevný bod s nejmenší odchylkou (lineární nejmenší čtverce),
* dvě tělesa exo (Body + WeldJoint, hmotnost a setrvačnost z CAD test2, geometrie STL),
* motor = CoordinateActuator "exo_motor" na knee_angle_r (optimal_force 1 => řízení = moment v Nm),
  řízený komponentou Controller podle varianty:
    pasivni    PrescribedController, konstantní 0 Nm,
    uhel_smer  ExpressionBasedCoordinateForce τ = G(θ)·s(ω) (zpětná vazba od stavu modelu),
    cnn        PrescribedController s průběhem z CNN (vytvoří s3_cnn.py).

Modely se ukládají do test3/model/ (geometrie exo do model/Geometry).
"""
import json
import shutil

from t3common import *  # noqa: F401,F403


def fit_hinge(model_path=BASE_MODEL, rng=(0.0, 110.0)):
    model = osim.Model(model_path)
    s = model.initSystem()
    knee = model.getCoordinateSet().get(KNEE)
    fem = model.getBodySet().get(f"femur_{SIDE}")
    tib = model.getBodySet().get(f"tibia_{SIDE}")
    th = np.linspace(*rng, 111)
    A, b, Rs, ds = [], [], [], []
    for a in th:
        knee.setValue(s, -np.radians(a))
        model.realizePosition(s)
        X = tib.findTransformBetween(s, fem)               # rám tibie vyjádřený ve femuru
        R = np.array([[X.R().get(i, j) for j in range(2)] for i in range(2)])
        d = np.array([X.p().get(0), X.p().get(1)])
        A.append(np.hstack([R.T, -np.eye(2)]))
        b.append(R.T @ d)
        Rs.append(R)
        ds.append(d)
    sol = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    p, c = sol[:2], sol[2:]
    err = np.array([np.linalg.norm(R.T @ (p - d) - c) for R, d in zip(Rs, ds)])
    return {"femur_point": p.tolist(), "tibia_point": c.tolist(), "theta_deg": th.tolist(),
            "misalignment_mm": (err * 1000).tolist()}


def build(variant, hinge, control=None, out=None):
    """variant: 'pasivni' | 'uhel_smer' | 'cnn'; control: (t, tau) pro 'cnn'."""
    props = json.load(open(CAD_PROPS))
    model = osim.Model(BASE_MODEL)
    model.setName(f"subject01_stairs_exo_{variant}")
    bs = model.getBodySet()
    for name, parent, p in (("exo_thigh_r", "femur_r", hinge["femur_point"]),
                            ("exo_shank_r", "tibia_r", hinge["tibia_point"])):
        pr = props[name]
        I = pr["inertia"]
        body = osim.Body(name, pr["mass"], osim.Vec3(*pr["mass_center"]), osim.Inertia(*I))
        for part in pr["parts"]:
            mesh = osim.Mesh(part["file"])
            mesh.setName(part["name"])
            mesh.setColor(osim.Vec3(*part["color"]))
            body.attachGeometry(mesh)
        model.addBody(body)
        model.addJoint(osim.WeldJoint(f"{name}_upevneni", bs.get(parent), osim.Vec3(p[0], p[1], 0),
                                      osim.Vec3(0), body, osim.Vec3(0), osim.Vec3(0)))
    # motor exoskeletu
    motor = osim.CoordinateActuator(KNEE)
    motor.setName(MOTOR)
    motor.setOptimalForce(1.0)                 # řízení [-] * 1 Nm = moment [Nm]
    motor.setMinControl(-TAU_MAX)
    motor.setMaxControl(TAU_MAX)
    model.addForce(motor)
    ctrl = osim.PrescribedController()
    ctrl.setName("exo_ridici_jednotka")
    ctrl.addActuator(motor)
    if variant == "cnn":
        t, tau = control
        f = osim.PiecewiseLinearFunction()
        for ti, v in zip(t, tau):
            f.addPoint(float(ti), float(v))
        ctrl.prescribeControlForActuator(MOTOR, f)
    else:
        ctrl.prescribeControlForActuator(MOTOR, osim.Constant(0.0))
    model.addController(ctrl)
    if variant == "uhel_smer":
        tab = json.load(open(CTRL_TABLE))
        law = osim.ExpressionBasedCoordinateForce(KNEE, tab["expression"]["uhel_smer"])
        law.setName("exo_motor_uhel_smer")
        model.addForce(law)
    model.finalizeConnections()
    path = out or MODELS[variant]
    model.printToXML(path)
    return path


def main():
    os.makedirs(os.path.join(MODEL_DIR, "Geometry"), exist_ok=True)
    for f in os.listdir(CAD_GEOM):
        shutil.copy(os.path.join(CAD_GEOM, f), os.path.join(MODEL_DIR, "Geometry", f))
    hinge = fit_hinge()
    json.dump(hinge, open(os.path.join(RES, "osa_kloubu.json"), "w"), indent=1)
    mis = np.array(hinge["misalignment_mm"])
    print(f"Osa exo ve femur_r {np.round(np.array(hinge['femur_point']) * 1000, 1)} mm, "
          f"nesouosost max {mis.max():.1f} mm, RMS {np.sqrt(np.mean(mis ** 2)):.1f} mm")
    for v in ("pasivni", "uhel_smer"):
        build(v, hinge)
    m = osim.Model(MODELS["pasivni"])
    s = m.initSystem()
    print(f"Modely uloženy, hmotnost s exo {m.getTotalMass(s):.2f} kg")


if __name__ == "__main__":
    main()
