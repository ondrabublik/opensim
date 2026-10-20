"""Krok 2: model přes OpenSim API – Gait10dof18musc + trať (schody nahoru, podesta, dolů) + židle
+ exoskelet pravého kolene (CAD z test2) s motorem CoordinateActuator a řídicí jednotkou.

Varianty: pasivni (motor 0 Nm = vypnutý exoskelet), cnn (PrescribedController s průběhem z CNN;
vytvoří s4_cnn.py).
"""
import json
import shutil

from t5common import *  # noqa: F401,F403
import s1_model as T3M                     # test3: osa kloubu exo

CAD_PROPS = T3.CAD_PROPS
CHAIR_COLOR = (0.45, 0.33, 0.22)
STAIR_COLOR = (0.75, 0.70, 0.60)


def add_scene(model, m):
    ground = model.getGround()
    for name, (cx, cy), (hx, hy) in m["terrain"]["blocks"]:
        fr = osim.PhysicalOffsetFrame(f"trat_{name}", ground, osim.Transform(osim.Vec3(cx, cy, 0)))
        br = osim.Brick(osim.Vec3(hx, hy, 0.6))
        br.setColor(osim.Vec3(*STAIR_COLOR))
        fr.attachGeometry(br)
        model.addComponent(fr)
    ch = m["chair"]
    x0 = ch["x_front"] - ch["depth"]
    parts = [("sedak", ((x0 + ch["x_front"]) / 2, ch["seat_h"] - 0.025), (ch["depth"] / 2, 0.025, 0.24)),
             ("operadlo", (x0 + 0.02, ch["seat_h"] + 0.25), (0.02, 0.25, 0.24))]
    for i, (dx, dz) in enumerate(((0.03, -0.2), (0.03, 0.2), (ch["depth"] - 0.03, -0.2), (ch["depth"] - 0.03, 0.2))):
        parts.append((f"noha_{i}", (x0 + dx, (ch["seat_h"] - 0.05) / 2, dz), (0.02, (ch["seat_h"] - 0.05) / 2, 0.02)))
    for p in parts:
        name, c, h = p[0], p[1], p[2]
        cz = c[2] if len(c) > 2 else 0.0
        fr = osim.PhysicalOffsetFrame(f"zidle_{name}", ground, osim.Transform(osim.Vec3(c[0], c[1], cz)))
        br = osim.Brick(osim.Vec3(h[0], h[1], h[2] if len(h) > 2 else 0.02))
        br.setColor(osim.Vec3(*CHAIR_COLOR))
        fr.attachGeometry(br)
        model.addComponent(fr)


def build(variant, hinge, control=None, out=None):
    m = meta()
    props = json.load(open(CAD_PROPS))
    model = osim.Model(PERSON_MODEL)
    model.setName(f"vstavani_schody_exo_{variant}_{JMENO}")
    kseg = segments()[1]
    add_scene(model, m)
    bs = model.getBodySet()
    for name, parent, p, k in (("exo_thigh_r", "femur_r", hinge["femur_point"], kseg["stehno"]),
                               ("exo_shank_r", "tibia_r", hinge["tibia_point"], kseg["berec"])):
        # díly exo se zvětší/zmenší s délkou segmentu (hmotnost stejná): těžiště ~k, setrvačnost ~k^2
        pr = props[name]
        I = [v * k * k for v in pr["inertia"]]
        body = osim.Body(name, pr["mass"], osim.Vec3(*[c * k for c in pr["mass_center"]]), osim.Inertia(*I))
        for part in pr["parts"]:
            mesh = osim.Mesh(part["file"])
            mesh.setName(part["name"])
            mesh.setColor(osim.Vec3(*part["color"]))
            mesh.set_scale_factors(osim.Vec3(k, k, k))
            body.attachGeometry(mesh)
        model.addBody(body)
        model.addJoint(osim.WeldJoint(f"{name}_upevneni", bs.get(parent), osim.Vec3(p[0], p[1], 0),
                                      osim.Vec3(0), body, osim.Vec3(0), osim.Vec3(0)))
    motor = osim.CoordinateActuator(KNEE)
    motor.setName(MOTOR)
    motor.setOptimalForce(1.0)
    tau_max = float((OSOBA.get("exoskelet", {}) or {}).get("max_moment_Nm", T3.TAU_MAX))
    motor.setMinControl(-tau_max)
    motor.setMaxControl(tau_max)
    model.addForce(motor)
    ctrl = osim.PrescribedController()
    ctrl.setName("exo_ridici_jednotka")
    ctrl.addActuator(motor)
    if variant == "cnn":
        f = osim.PiecewiseLinearFunction()
        for ti, v in zip(*control):
            f.addPoint(float(ti), float(v))
        ctrl.prescribeControlForActuator(MOTOR, f)
    else:
        ctrl.prescribeControlForActuator(MOTOR, osim.Constant(0.0))
    model.addController(ctrl)
    model.finalizeConnections()
    path = out or MODELS[variant]
    model.printToXML(path)
    return path


def main():
    os.makedirs(os.path.join(MODEL_DIR, "Geometry"), exist_ok=True)
    for f in os.listdir(T3.CAD_GEOM):
        shutil.copy(os.path.join(T3.CAD_GEOM, f), os.path.join(MODEL_DIR, "Geometry", f))
    hinge = T3M.fit_hinge(PERSON_MODEL)
    json.dump(hinge, open(os.path.join(RES, "osa_kloubu.json"), "w"), indent=1)
    path = build("pasivni", hinge)
    mm = osim.Model(path)
    s = mm.initSystem()
    print(f"[{JMENO}] Model {path}: hmotnost {mm.getTotalMass(s):.2f} kg, nesouosost osy exo max "
          f"{max(hinge['misalignment_mm']):.1f} mm")


if __name__ == "__main__":
    main()
