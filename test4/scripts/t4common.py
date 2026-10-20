"""Společné nastavení test4: dopředná simulace v uzavřené smyčce s CNN (OpenSim API).

Vychází z test3 (model s exoskeletem, reakční síly pro hmotu exo, CNN) a test1 (kinematika).
"""
import os
import sys

TEST3_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "test3", "scripts")
if TEST3_SCRIPTS not in sys.path:
    sys.path.insert(0, TEST3_SCRIPTS)
import t3common as T3  # noqa: E402
from t3common import osim, np, table_np, np_table, write_sto, cycles, in_cycles  # noqa: E402,F401

ROOT4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES4, MODEL4 = os.path.join(ROOT4, "results"), os.path.join(ROOT4, "model")
for _d in (RES4, MODEL4):
    os.makedirs(_d, exist_ok=True)

KNEE, MOTOR, SIDE = T3.KNEE, T3.MOTOR, T3.SIDE
DT_HUMAN = 0.001          # regulátor člověka 1 kHz
FS_CNN = 20.0             # řídicí smyčka exoskeletu 20 Hz
WN, ZETA = 40.0, 1.0      # dynamika sledování pohybu člověkem [rad/s]
M_ON, T_MOTOR = 40.0, 0.02


def build_forward_model():
    """Model z test3 (pasivní exo) pro dopřednou simulaci:
    svaly odebrané, člověk = CoordinateActuator v každé souřadnici (u pánve jde o reziduální
    aktuátory jako v CMC/RRA), reakční síly země = ExternalForce ze souboru (předepsané)."""
    m = osim.Model(T3.MODELS["pasivni"])
    m.setName("subject01_stairs_exo_dopredna")
    fs = m.updForceSet()
    for i in reversed(range(fs.getSize())):
        if "Muscle" in fs.get(i).getConcreteClassName():
            fs.remove(i)
    cs = m.getCoordinateSet()
    names = [cs.get(i).getName() for i in range(cs.getSize())]
    for n in names:
        a = osim.CoordinateActuator(n)
        a.setName("clovek_" + n)
        a.setOptimalForce(1.0)
        m.addForce(a)
    st = osim.Storage(T3.GRF_EXO)
    for side, p, body in (("r", "", "calcn_r"), ("l", "1_", "calcn_l")):
        ef = osim.ExternalForce(st, p + "ground_force_v", p + "ground_force_p", p + "ground_torque_",
                                body, "ground", "ground")
        ef.setName("grf_" + side)
        m.addForce(ef)
    m.finalizeConnections()
    m._grf_storage = st          # ExternalForce drží jen odkaz -> Storage musí žít s modelem
    return m, names


def reference(names, model):
    """Referenční q, u, udot z GCV splajnů OpenSim (jako InverseDynamicsTool), mřížka 1 ms."""
    kst = osim.Storage(T3.KIN)
    spl = osim.GCVSplineSet(5, kst)
    cs = model.getCoordinateSet()
    t = np.arange(kst.getFirstTime(), kst.getLastTime() + 1e-9, DT_HUMAN)
    d1, d2 = osim.StdVectorInt(), osim.StdVectorInt()
    d1.push_back(0)
    d2.push_back(0)
    d2.push_back(0)
    Q, U, A = {}, {}, {}
    for n in names:
        f = spl.get(n)
        k = np.radians(1.0) if cs.get(n).getMotionType() == 1 else 1.0
        x = [osim.Vector(1, float(ti)) for ti in t]
        Q[n] = np.array([f.calcValue(v) for v in x]) * k
        U[n] = np.array([f.calcDerivative(d1, v) for v in x]) * k
        A[n] = np.array([f.calcDerivative(d2, v) for v in x]) * k
    return t, Q, U, A
