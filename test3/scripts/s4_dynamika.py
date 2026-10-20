"""Krok 4: inverzní dynamika všech variant a moment/výkon motoru (OpenSim API).

* InverseDynamicsTool s modelem, ve kterém motor exo působí (CoordinateActuator +
  PrescribedController, resp. ExpressionBasedCoordinateForce). Výsledný moment kolene je přímo
  moment, který musí dodat člověk - nic se neodečítá ručně.
* AnalyzeTool + ForceReporter (moment motoru) + Kinematics (rychlost kolene) -> výkon motoru.
* Kontrola: moment motoru z ForceReporter = rozdíl ID (pasivní - aktivní).
"""
from t3common import *  # noqa: F401,F403

K = KNEE + "_moment"
VARIANTS = {"uhel_smer": "exo_motor_uhel_smer", "cnn": MOTOR}


def main():
    out = {}
    t, d = table_np(os.path.join(RES, "ID_bez_exo.sto"))
    out["t"], out["M_bez"] = t, d[K]
    _, d = table_np(os.path.join(RES, "ID_exo_pasivni.sto"))
    out["M_pas"] = d[K]
    for v, force in VARIANTS.items():
        print(f"ID + AnalyzeTool: {v} ...")
        _, d = run_id(f"ID_exo_{v}", MODELS[v], LOADS_EXO)
        out[f"M_{v}"] = d[K]
        fr = osim.ForceReporter()
        fr.setName("ForceReporter")
        fr.setStepInterval(1)
        km = osim.Kinematics()
        km.setName("Kinematics")
        km.setStepInterval(1)
        km.setInDegrees(False)
        run_analyze(f"exo_{v}", MODELS[v], [fr, km], LOADS_EXO)
        tf, F = table_np(os.path.join(RES, f"exo_{v}_ForceReporter_forces.sto"))
        tu, U = table_np(os.path.join(RES, f"exo_{v}_Kinematics_u.sto"))
        n = len(t)
        tau = np.interp(t, tf, F[force])
        om = np.interp(t, tu, U[KNEE])
        out[f"tau_{v}"], out[f"P_{v}"] = tau, tau * om
        out["omega"] = om
        print(f"  moment motoru: max {tau.max():.1f} Nm; ForceReporter vs. rozdíl ID: "
              f"max {np.abs(tau - (out['M_pas'] - out[f'M_{v}'])).max():.3f} Nm")
    np.savez(os.path.join(RES, "test3_vysledky.npz"), **out)

    # srovnání s test2
    r2 = np.load(os.path.join(TEST2, "results", "exo_vysledky.npz"))
    l2 = open(os.path.join(TEST2, "results", "moje_rizeni", "cnn_rizeni", "prubeh.csv"),
              encoding="utf-8").read().splitlines()
    h2 = l2[0].split(";")
    D2 = np.array([[float(x) for x in l.split(";")[1:]] for l in l2[1:]])
    M2_cnn = D2[:, h2.index("moment_kolene_clovek_Nm") - 1]
    n = min(len(t), len(M2_cnn))
    print("Srovnání momentu kolene člověka s test2 (max |rozdíl|):")
    for key, ref in (("M_bez", r2["M_bez"]), ("M_pas", r2["M_pas"]), ("M_uhel_smer", r2["M_uhel_smer"]),
                     ("M_cnn", M2_cnn)):
        print(f"  {key:12s} {np.abs(out[key][:n] - ref[:n]).max():.4f} Nm")


if __name__ == "__main__":
    main()
