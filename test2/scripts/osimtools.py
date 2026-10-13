"""Pomocné funkce: načtení OpenSim (Python 3.8), I/O souborů .mot/.sto,
vyhlazování, kvintické přechody a kinematika rovinného modelu Gait10dof18musc."""
import os
import sys

OPENSIM_SDK = r"C:\OpenSim 4.5\sdk\Python"
if OPENSIM_SDK not in sys.path:
    sys.path.insert(0, OPENSIM_SDK)
import opensim as osim  # noqa: E402
import numpy as np  # noqa: E402

osim.Logger.setLevelString("error")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_DIR = os.path.join(os.path.expanduser("~"), "Documents", "OpenSim", "4.5",
                           "Models", "Gait10dof18musc")

COORDS = ["pelvis_tilt", "pelvis_tx", "pelvis_ty",
          "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
          "hip_flexion_l", "knee_angle_l", "ankle_angle_l", "lumbar_extension"]
ROT = {c for c in COORDS if c not in ("pelvis_tx", "pelvis_ty")}

# Geometrie chodidla v rámci calcn (calcn_r i calcn_l mají stejné hodnoty v x,y)
SOLE = 0.045                         # výška počátku calcn nad chodidlem
HEEL = np.array([0.0, -SOLE])        # pata
MTP = np.array([0.18, -SOLE])        # bod pod hlavičkami metatarzů (pivot při odvalu)
TOE = np.array([0.245, -SOLE])       # špička
FOOT_PTS = np.array([HEEL, MTP, TOE, [0.12, -SOLE + 0.0], [0.245, -SOLE + 0.03]])


# ----------------------------------------------------------------- soubory
def read_mot(path):
    lines = open(path).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().lower() == "endheader")
    cols = [c.strip() for c in lines[i + 1].split("\t") if c.strip()]
    data = np.array([[float(x) for x in l.split()] for l in lines[i + 2:] if l.strip()])
    return cols, data


def write_mot(path, cols, data, name="data", in_degrees=True):
    with open(path, "w") as f:
        f.write(f"{name}\nversion=1\nnRows={data.shape[0]}\nnColumns={data.shape[1]}\n")
        f.write(f"inDegrees={'yes' if in_degrees else 'no'}\nendheader\n")
        f.write("\t".join(cols) + "\n")
        for row in data:
            f.write("\t".join(f"{v:.8f}" for v in row) + "\n")


# ------------------------------------------------------------ signály
def butter_lowpass(x, fc, fs):
    """Bezfázový Butterworth 2. řádu (filtr tam a zpět => 4. řád), po sloupcích."""
    x = np.asarray(x, float)
    w = np.tan(np.pi * fc / fs)
    k = np.sqrt(2.0)
    a0 = 1 + k * w + w * w
    b = np.array([w * w, 2 * w * w, w * w]) / a0
    a = np.array([1.0, 2 * (w * w - 1) / a0, (1 - k * w + w * w) / a0])

    def run(y):
        out = np.empty_like(y)
        x1 = x2 = y[0]
        y1 = y2 = y[0]
        for n, xn in enumerate(y):
            yn = b[0] * xn + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            x2, x1, y2, y1 = x1, xn, y1, yn
            out[n] = yn
        return out

    def filt(y):
        pad = min(len(y) - 1, int(fs / fc))
        yp = np.concatenate([2 * y[0] - y[pad:0:-1], y, 2 * y[-1] - y[-2:-pad - 2:-1]])
        yp = run(run(yp)[::-1])[::-1]
        return yp[pad:pad + len(y)]

    if x.ndim == 1:
        return filt(x)
    return np.column_stack([filt(x[:, j]) for j in range(x.shape[1])])


def cubic_spline(t, y):
    """Přirozený kubický splajn; vrací funkci f(tq, der=0)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(t)
    h = np.diff(t)
    A = np.zeros((n, n))
    r = np.zeros((n,) + y.shape[1:])
    A[0, 0] = A[-1, -1] = 1
    for i in range(1, n - 1):
        A[i, i - 1:i + 2] = [h[i - 1], 2 * (h[i - 1] + h[i]), h[i]]
        r[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, r)
    b = (np.diff(y, axis=0) / h[:, None] if y.ndim > 1 else np.diff(y) / h)
    hh = h[:, None] if y.ndim > 1 else h
    b = b - hh * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * hh)

    def f(tq, der=0):
        tq = np.asarray(tq, float)
        i = np.clip(np.searchsorted(t, tq) - 1, 0, n - 2)
        dx = tq - t[i]
        if y.ndim > 1:
            dx = dx[:, None]
        if der == 0:
            return y[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
        if der == 1:
            return b[i] + 2 * c[i] * dx + 3 * d[i] * dx ** 2
        return 2 * c[i] + 6 * d[i] * dx
    return f


def smoothstep(s):
    """Kvintický přechod 0->1 s nulovou 1. i 2. derivací na koncích."""
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def quintic(p0, v0, a0, p1, v1, a1, T):
    """Koeficienty kvintického polynomu v čase 0..T s okrajovými podmínkami."""
    M = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 2, 0, 0, 0],
                  [1, T, T**2, T**3, T**4, T**5],
                  [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
                  [0, 0, 2, 6*T, 12*T**2, 20*T**3]], float)
    return np.linalg.solve(M, [p0, v0, a0, p1, v1, a1])


def polyval(c, t):
    return sum(ci * t ** i for i, ci in enumerate(c))


def bump(s):
    """Hrbol s nulovou hodnotou, 1. i 2. derivací na koncích, max 1 v s=0.5."""
    s = np.clip(s, 0.0, 1.0)
    return 64.0 * s ** 3 * (1 - s) ** 3


def rot2(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


# ------------------------------------------------------------ model
class Kin:
    """Dopředná a inverzní kinematika rovinného modelu."""

    def __init__(self, model_path):
        self.model = osim.Model(model_path)
        self.state = self.model.initSystem()
        cs = self.model.getCoordinateSet()
        self.coord = {c: cs.get(c) for c in COORDS}
        bs = self.model.getBodySet()
        self.body = {n: bs.get(n) for n in ["pelvis", "femur_r", "femur_l", "talus_r",
                                            "talus_l", "calcn_r", "calcn_l", "torso"]}
        self.mass = self.model.getTotalMass(self.state)
        self.q = dict.fromkeys(COORDS, 0.0)
        self._set_all()
        self.model.realizePosition(self.state)
        # poloha hlezna (talus) v rámci calcn
        cal = self.body["calcn_r"]
        p = self.body["talus_r"].findStationLocationInAnotherFrame(
            self.state, osim.Vec3(0, 0, 0), cal)
        self.ankle_in_calcn = np.array([p.get(0), p.get(1)])
        # dosah kyčel-hlezno jako funkce flexe kolene
        self.knee_tab = np.radians(np.linspace(-120, 10, 261))
        dist = []
        for k in self.knee_tab:
            self.set({"knee_angle_r": k})
            dist.append(np.linalg.norm(self.pos("talus_r") - self.pos("femur_r")))
        self.reach_tab = np.array(dist)
        self.set({"knee_angle_r": 0.0})

    def _set_all(self):
        for c, v in self.q.items():
            self.coord[c].setValue(self.state, float(v), False)

    def set(self, q):
        for c, v in q.items():
            self.q[c] = v
            self.coord[c].setValue(self.state, float(v), False)
        self.model.realizePosition(self.state)

    def pos(self, body):
        p = self.body[body].getPositionInGround(self.state)
        return np.array([p.get(0), p.get(1)])

    def angle(self, body):
        R = self.body[body].getTransformInGround(self.state).R()
        return R.convertRotationToBodyFixedXYZ().get(2)

    def foot_pose(self, side):
        return np.r_[self.pos("calcn_" + side), self.angle("calcn_" + side)]

    def reach(self, knee_flex_deg):
        """Vzdálenost kyčel-hlezno pro danou flexi kolene (stupně, kladně = flexe)."""
        return np.interp(-np.radians(knee_flex_deg), self.knee_tab, self.reach_tab)

    def max_reach(self):
        return self.reach_tab.max()

    def leg_guess(self, side, target):
        """Analytický odhad (hip, knee, ankle) pro větev s flexí kolene."""
        names = [f"hip_flexion_{side}", f"knee_angle_{side}", f"ankle_angle_{side}"]
        ankle_t = target[:2] + rot2(target[2]) @ self.ankle_in_calcn
        hip = self.pos("femur_" + side)
        d = np.linalg.norm(ankle_t - hip)
        branch = self.knee_tab <= np.radians(2.0)
        knee = np.interp(d, self.reach_tab[branch], self.knee_tab[branch])
        self.set({names[0]: 0.0, names[1]: knee, names[2]: 0.0})
        v0 = self.pos("talus_" + side) - hip
        v1 = ankle_t - hip
        h = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        h = (h + np.pi) % (2 * np.pi) - np.pi
        self.set({names[0]: h})
        a = target[2] - self.angle("calcn_" + side)
        return np.array([h, knee, a])

    def solve_leg(self, side, target, guess=None):
        """Najde (hip, knee, ankle) tak, aby calcn mělo pózu target=(x, y, phi)."""
        names = [f"hip_flexion_{side}", f"knee_angle_{side}", f"ankle_angle_{side}"]
        x = self.leg_guess(side, target) if guess is None else np.array(guess, float)

        def res(x):
            self.set(dict(zip(names, x)))
            return self.foot_pose(side) - target

        r = res(x)
        for _ in range(30):
            if np.abs(r).max() < 1e-7:
                break
            J = np.empty((3, 3))
            for j in range(3):
                dx = np.zeros(3)
                dx[j] = 1e-6
                J[:, j] = (res(x + dx) - r) / 1e-6
            step = np.linalg.solve(J, -r)
            step = np.clip(step, -0.3, 0.3)
            x = x + step
            r = res(x)
        return x, np.abs(r[:2]).max()
