"""Krok 3: řízení motoru konvoluční sítí, snímače z OpenSim API.

* Snímače: AnalyzeTool + BodyKinematics -> úhlová rychlost stehna (femur_r) a holeně (tibia_r)
  vůči zemi kolem osy z [deg/s] (= gyroskopy IMU na objímkách); úhel kolene = souřadnice
  knee_angle_r (= enkodér motoru).
* Řídicí smyčka 20 Hz: CNN natrénovaná v test2 (váhy cnn_vahy.npz), okno 1 s, výstup ON/OFF,
  tvarovač nultého řádu; náběh momentu motoru 1. řádu (20 ms). Síť se počítá v numpy, protože
  OpenSim neumožňuje psát vlastní Controller v Pythonu.
* Výsledný průběh momentu se vloží do modelu jako PrescribedController motoru exo_motor
  (PiecewiseLinearFunction) -> model subject01_stairs_exo_cnn.osim.
"""
import json

from t3common import *  # noqa: F401,F403
import s1_model

M_ON = 40.0          # moment zapnutého motoru [Nm]
T_MOTOR = 0.02       # časová konstanta náběhu [s]
FS_CTRL = 20.0       # frekvence řídicí smyčky [Hz]


class CNN:
    def __init__(self, path=CNN_WEIGHTS):
        d = np.load(path)
        self.w = {k: d[k] for k in d.files}
        self.scale, self.win = d["scale"], int(d["win"])

    @staticmethod
    def conv(x, w, b):
        K = w.shape[2]
        L = x.shape[1] - K + 1
        return np.einsum("oik,ilk->ol", w, np.stack([x[:, k:k + L] for k in range(K)], 2)) + b[:, None]

    def prob(self, window):
        w = self.w
        x = (np.asarray(window) / self.scale).T
        for c in ("c1", "c2", "c3"):
            x = np.maximum(self.conv(x, w[c + ".weight"], w[c + ".bias"]), 0)
        x = np.maximum(w["f1.weight"] @ x.reshape(-1) + w["f1.bias"], 0)
        return float(1 / (1 + np.exp(-(w["f2.weight"] @ x + w["f2.bias"]))))


def sensors():
    """Úhel kolene + úhlové rychlosti segmentů z OpenSim (AnalyzeTool, BodyKinematics)."""
    bk = osim.BodyKinematics()
    bk.setName("BodyKinematics")
    bk.setInDegrees(True)
    bk.setStepInterval(1)
    bodies = osim.ArrayStr()
    for b in (f"femur_{SIDE}", f"tibia_{SIDE}"):        # jen segmenty s IMU
        bodies.append(b)
    bk.setBodiesToRecord(bodies)
    run_analyze("snimace", MODELS["pasivni"], [bk])
    t, vel = table_np(os.path.join(RES, "snimace_BodyKinematics_vel_global.sto"))
    tk, kin = table_np(KIN)
    n = min(len(t), len(tk))
    return (t[:n], -kin[KNEE][:n], vel[f"femur_{SIDE}_Oz"][:n], vel[f"tibia_{SIDE}_Oz"][:n])


def control_loop(t, uhel, w_st, w_ho, net):
    """Kauzální smyčka: rozhodnutí každých 50 ms z posledních 20 vzorků, pak náběh motoru."""
    X = np.column_stack([uhel, w_st, w_ho])
    on = np.zeros(len(t), bool)
    p_out = np.zeros(len(t))
    buf = np.tile(X[0], (net.win, 1))
    t_next, state, p = t[0], False, 0.0
    for i in range(len(t)):
        if t[i] + 1e-9 >= t_next:
            buf = np.vstack([buf[1:], X[i]])
            p = net.prob(buf)
            state = p > 0.5
            t_next += 1 / FS_CTRL
        on[i], p_out[i] = state, p
    a = 1 - np.exp(-(t[1] - t[0]) / T_MOTOR)
    tau = np.zeros(len(t))
    y = 0.0
    for i in range(len(t)):
        y += a * (M_ON * on[i] - y)
        tau[i] = y
    return on, p_out, tau


def main():
    print("Snímače z AnalyzeTool/BodyKinematics ...")
    t, uhel, w_st, w_ho = sensors()
    on, p, tau = control_loop(t, uhel, w_st, w_ho, CNN())
    write_sto(np_table(t, {"koleno_flexe_deg": uhel, "omega_stehno_deg_s": w_st, "omega_holen_deg_s": w_ho,
                           "p_on": p, "motor_zapnut": on.astype(float), "exo_motor_Nm": tau}),
              os.path.join(RES, "cnn_rizeni.sto"))
    hinge = json.load(open(os.path.join(RES, "osa_kloubu.json")))
    path = s1_model.build("cnn", hinge, control=(t, tau))
    s1_model.build("cnn", hinge, control=(t, tau), out=MAIN_MODEL)
    # srovnání snímačů s test2 (tam numerická derivace úhlů segmentů)
    l2 = open(os.path.join(TEST2, "results", "moje_rizeni", "cnn_rizeni", "prubeh.csv"),
              encoding="utf-8").read().splitlines()
    h2 = l2[0].split(";")
    D2 = np.array([[float(x) for x in l.split(";")[1:]] for l in l2[1:]])
    c2 = {k: D2[:, i - 1] for i, k in enumerate(h2) if i > 0}
    n = min(len(t), len(D2))
    print(f"  rozdíl snímačů proti test2: stehno max {np.abs(w_st[:n] - c2['omega_stehno_deg_s'][:n]).max():.2f} °/s,"
          f" holeň max {np.abs(w_ho[:n] - c2['omega_holen_deg_s'][:n]).max():.2f} °/s")
    print(f"  shoda ON/OFF s test2: {np.mean(on[:n] == (c2['motor_zapnut'][:n] > 0.5)) * 100:.2f} % vzorků; "
          f"zapnuto {on.mean() * 100:.1f} % času")
    print("Model s řízením CNN:", path)


if __name__ == "__main__":
    main()
