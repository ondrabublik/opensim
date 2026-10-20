"""Krok 4: řízení exoskeletu neuronovou sítí a moment kolene s vypnutým / zapnutým exoskeletem.

* Snímače: AnalyzeTool + BodyKinematics (úhlová rychlost stehna a holeně vůči zemi),
  úhel kolene ze souřadnice modelu.
* Řídicí smyčka 20 Hz, okno 1 s, síť v numpy (cnn_numpy.py), váhy podle osoba.json:
    typ "regrese": moment motoru = podil_kompenzace * výstup [N m/kg] * hmotnost (osoba + exo),
                   omezený na +-max_moment_Nm (kompenzace momentu kolene),
    typ "onoff":   motor ON (moment_zapnuto_Nm) při P(ON) > 0,5.
  Náběh momentu motoru 1. řádu (20 ms).
* Model s motorem řízeným PrescribedController -> InverseDynamicsTool vrací moment kolene,
  který musí dodat člověk (zapnutý exo). Vypnutý = stejný model, motor 0 N m.
* AnalyzeTool + ForceReporter: kontrola momentu motoru.
"""
import json

from t5common import *  # noqa: F401,F403
import s2_model
from cnn_numpy import CNN

T_MOTOR = 0.02       # časová konstanta náběhu momentu motoru [s]


def sensors():
    bk = osim.BodyKinematics()
    bk.setName("BodyKinematics")
    bk.setInDegrees(True)
    bk.setStepInterval(1)
    b = osim.ArrayStr()
    for name in (f"femur_{SIDE}", f"tibia_{SIDE}"):
        b.append(name)
    bk.setBodiesToRecord(b)
    run_analyze("snimace", MODELS["pasivni"], [bk])
    t, vel = table_np(os.path.join(RES, "snimace_BodyKinematics_vel_global.sto"))
    tk, kin = table_np(KIN)
    n = min(len(t), len(tk))
    return t[:n], -kin[KNEE][:n], vel[f"femur_{SIDE}_Oz"][:n], vel[f"tibia_{SIDE}_Oz"][:n]


def control_loop(t, uhel, w_st, w_ho, net, exo_cfg):
    """Kauzální smyčka: výpočet sítě každých 1/fs s z posledního okna, držení výstupu, náběh motoru."""
    X = np.column_stack([uhel, w_st, w_ho])
    mass = OSOBA["hmotnost_kg"] + EXO_MASS
    tau_max = float(exo_cfg.get("max_moment_Nm", 60.0))
    podil = float(exo_cfg.get("podil_kompenzace", 1.0))
    m_on = float(exo_cfg.get("moment_zapnuto_Nm", 40.0))
    deadband = float(exo_cfg.get("mrtve_pasmo_Nm_kg", 0.1))
    out = np.zeros(len(t))
    req = np.zeros(len(t))
    buf = np.tile(X[0], (net.win, 1))
    t_next, o, r = t[0], 0.0, 0.0
    for i in range(len(t)):
        if t[i] + 1e-9 >= t_next:
            buf = np.vstack([buf[1:], X[i]])
            o = net.output(buf)
            if net.typ == "regrese" and abs(o) < deadband:   # mrtvé pásmo: drobné výstupy (rovina) = 0
                o = 0.0
            r = podil * o * mass if net.typ == "regrese" else (m_on if o > 0.5 else 0.0)
            r = float(np.clip(r, -tau_max, tau_max))
            t_next += 1.0 / net.fs
        out[i], req[i] = o, r
    a = 1 - np.exp(-(t[1] - t[0]) / T_MOTOR)
    tau = np.zeros(len(t))
    y = 0.0
    for i in range(len(t)):
        y += a * (req[i] - y)
        tau[i] = y
    return out, tau


def main():
    exo_cfg = OSOBA.get("exoskelet", {}) or {}
    print("Snímače (BodyKinematics) a síť ...")
    t, uhel, w_st, w_ho = sensors()
    wpath = cnn_weights()
    net = CNN(wpath)
    print(f"  síť: {os.path.relpath(wpath, BASE)} (typ {net.typ}); max. moment motoru "
          f"{exo_cfg.get('max_moment_Nm', 60)} N m, podíl kompenzace {exo_cfg.get('podil_kompenzace', 1.0)}")
    out, tau = control_loop(t, uhel, w_st, w_ho, net, exo_cfg)
    on = np.abs(tau) > 1.0
    write_sto(np_table(t, {"koleno_flexe_deg": uhel, "omega_stehno_deg_s": w_st, "omega_holen_deg_s": w_ho,
                           "vystup_site": out, "motor_zapnut": on.astype(float), "exo_motor_Nm": tau}),
              os.path.join(RES, "cnn_rizeni.sto"))
    hinge = json.load(open(os.path.join(RES, "osa_kloubu.json")))
    s2_model.build("cnn", hinge, control=(t, tau))
    s2_model.build("cnn", hinge, control=(t, tau), out=MAIN_MODEL)
    print("ID s exoskeletem zapnutým ...")
    _, don = run_id("ID_exo_zapnuty", MODELS["cnn"], LOADS)
    _, doff = table_np(os.path.join(RES, "ID_exo_vypnuty.sto"))
    fr = osim.ForceReporter()
    fr.setName("ForceReporter")
    fr.setStepInterval(1)
    run_analyze("exo_cnn", MODELS["cnn"], [fr])
    tf, F = table_np(os.path.join(RES, "exo_cnn_ForceReporter_forces.sto"))
    k = KNEE + "_moment"
    n = min(len(t), len(don[k]), len(doff[k]))
    print(f"  kontrola: moment motoru (ForceReporter) vs. rozdíl ID vypnutý - zapnutý: max "
          f"{np.abs(np.interp(t[:n], tf, F[MOTOR]) - (doff[k][:n] - don[k][:n])).max():.3f} Nm")
    m = meta()
    for nm, a, b in phase_bounds(m):
        sel = (t[:n] >= a) & (t[:n] < b)
        print(f"  {nm:15s} motor zapnut {on[:n][sel].mean() * 100:5.1f} % (max {np.abs(tau[:n][sel]).max():5.1f} Nm)"
              f" | max moment kolene vypnutý {doff[k][:n][sel].max():6.1f} Nm, zapnutý {don[k][:n][sel].max():6.1f} Nm")


if __name__ == "__main__":
    main()
