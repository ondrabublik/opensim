"""Krok 1: syntéza pohybu – vstávání ze židle, 5 m po rovině, 5 m do schodů, podesta, 5 m ze schodů.

Pohyb není naměřený. Sestaví se z plánu kroků:
  * chodidla stojí na plánovaných místech (při sezení i v chůzi), ve stoji se odvalují přes
    hlavičky metatarzů (zvedání paty řízené cílovou flexí kolene před odrazem, jako v test1),
  * švih chodidla = kvintický polynom z pózy při odrazu do pózy při dopadu + "hrb" pro vůli,
    tvar se hledá tak, aby chodidlo nezavadilo o stupně,
  * pánev: při vstávání podle bodů kyčle (sed -> předklon -> odlehnutí -> vzpřim), v chůzi
    monotónní kubická interpolace bodů v okamžicích dopadu (rovina, výstup, podesta, sestup),
  * úhly kyčle, kolene a kotníku = inverzní kinematika modelu (Kin.solve_leg) pro polohu chodidla.

Výstupy: data/pohyb_raw.mot, data/pohyb_ik.mot (filtr 6 Hz), data/meta.json, results/kontrola_pohybu.png
"""
import json

from osimtools import *  # noqa: F401,F403
import p5 as P
import t5common as T5

# osoba z osoba.json: škálovaný model, parametry pohybu, rozměry chodidla
T5.apply_person(P)
DATA, RES = T5.DATA, T5.RES
CORR = os.path.join(DATA, "korekce_panve.npy")     # (t, dx) z s3b_rovnovaha.py
SRC_MODEL = T5.PERSON_MODEL
SOLE, HEEL, MTP, TOE = T5.foot_points()
_kf = T5.segments()[1]["chodidlo"]
FOOT_PTS = FOOT_PTS * _kf
PIVOT = TOE - [0.01 * _kf, 0.0]


def pchip(t, y):
    """Monotónní kubická interpolace (Fritsch-Carlson); vrací funkci f(tq)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    h = np.diff(t)
    d = np.diff(y) / h
    m = np.zeros_like(y)
    for k in range(1, len(y) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def f(tq):
        tq = np.clip(np.asarray(tq, float), t[0], t[-1])
        i = np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2)
        s = (tq - t[i]) / h[i]
        h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
        h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
        return h00 * y[i] + h10 * h[i] * m[i] + h01 * y[i + 1] + h11 * h[i] * m[i + 1]
    return f


# ======================================================== trať
class Terrain:
    def __init__(self):
        self.XS = P.X_FEET + P.WALK_LEN                         # první stupeň nahoru
        self.XD = self.XS + (P.N_STEPS - 1) * P.STEP_G + P.LANDING  # hrana podesty (sestup)
        self.X_END = self.XD + (P.N_STEPS - 1) * P.STEP_G        # pata schodů dolů

    def height(self, x):
        x = np.asarray(x, float)
        up = np.clip(np.floor((x - self.XS) / P.STEP_G) + 1, 0, P.N_STEPS) * P.STEP_H
        m = np.floor((x - self.XD) / P.STEP_G) + 1
        down = np.clip(P.N_STEPS - m, 0, P.N_STEPS) * P.STEP_H
        return np.where(x < self.XD, up, down)

    def blocks(self):
        """Kvádry pro model/animaci: (název, střed (x,y), poloviční rozměry (x,y))."""
        out = []
        for j in range(1, P.N_STEPS):
            x0 = self.XS + (j - 1) * P.STEP_G
            out.append((f"nahoru_{j:02d}", (x0 + P.STEP_G / 2, j * P.STEP_H / 2), (P.STEP_G / 2, j * P.STEP_H / 2)))
        x0, x1 = self.XS + (P.N_STEPS - 1) * P.STEP_G, self.XD
        out.append(("podesta", ((x0 + x1) / 2, P.N_STEPS * P.STEP_H / 2),
                    ((x1 - x0) / 2, P.N_STEPS * P.STEP_H / 2)))
        for m in range(1, P.N_STEPS):
            h = (P.N_STEPS - m) * P.STEP_H
            x0 = self.XD + (m - 1) * P.STEP_G
            out.append((f"dolu_{m:02d}", (x0 + P.STEP_G / 2, h / 2), (P.STEP_G / 2, h / 2)))
        return out


# ======================================================== generátor
class Generator:
    def __init__(self, kin, ref_tilt, ref_lumbar):
        self.kin, self.ter = kin, Terrain()
        kin.set(dict.fromkeys(COORDS, 0.0))
        self.hip_local = kin.pos("femur_r")                     # kyčel v rámu pánve (tilt = 0)
        self.tilt0, self.trunk0 = ref_tilt, P.TRUNK_WALK     # pánev jako v měření, trup vzpřímenější
        # stoj: pánev nakloněná o tilt0, nohy svisle (flexe kyčle = -tilt0), chodidla naplocho
        st = self.contact_offset(np.degrees(-ref_tilt), 0.0)
        self.Y_STAND = -st[1] + SOLE                            # výška pánve ve stoji
        self.HEEL2PEL = -st[0]                                  # pánev před patou ve stoji
        # póza nohy při dopadu na schod: pánev vůči calcn vedoucí nohy
        self.off_up = self.contact_offset(P.HIP_IC_UP, P.KNEE_IC_UP)
        self.off_down = self.contact_offset(P.HIP_IC_DOWN, P.KNEE_IC_DOWN, P.TOE_FIRST)
        self._plan_steps()
        self._plan_pelvis()
        self.corr = np.load(CORR) if os.path.exists(CORR) else None
        for side in ("r", "l"):
            for sw in self.swings[side]:
                self._design_swing(sw)

    def contact_offset(self, hip_deg, knee_deg, pitch=0.0):
        """Póza calcn vůči pánvi (x, y) pro danou flexi kyčle a kolene, chodidlo skloněné o pitch."""
        k = self.kin
        k.set({"pelvis_tilt": self.tilt0, "pelvis_tx": 0.0, "pelvis_ty": 0.0, "hip_flexion_r": np.radians(hip_deg),
               "knee_angle_r": -np.radians(knee_deg), "ankle_angle_r": 0.0})
        k.set({"ankle_angle_r": pitch - k.angle("calcn_r")})
        return k.foot_pose("r")[:2]

    @staticmethod
    def heel_first_pose(base, phi):
        """Póza calcn otočená o phi (špička nahoru) kolem paty položené jako u plochého chodidla."""
        heel = base[:2] + HEEL
        return np.r_[heel - rot2(phi) @ HEEL, phi]

    @staticmethod
    def toe_first_pose(base, phi):
        """Póza calcn otočená o phi kolem špičky položené tam, kde je u plochého chodidla."""
        toe = base[:2] + TOE
        return np.r_[toe - rot2(phi) @ TOE, phi]

    # ---------------------------------------------- plán kroků
    def _plan_steps(self):
        ter = self.ter
        seq = [(P.X_FEET + P.FIRST_STEP, 0.0, "rovina", P.FIRST)]
        x_last = ter.XS + P.HEEL_MARGIN - P.FIRST_STEP_LEN
        n = max(1, int(round((x_last - seq[0][0]) / P.LEVEL["length"])))
        for i in range(1, n + 1):
            seq.append((seq[0][0] + i * (x_last - seq[0][0]) / n, 0.0, "rovina", P.LEVEL))
        for k in range(1, P.N_STEPS + 1):
            prm = P.UP if k > 1 else dict(swing=P.UP["swing"], ds=P.LEVEL["ds"])   # 1. nášlap z roviny
            seq.append((ter.XS + (k - 1) * P.STEP_G + P.HEEL_MARGIN, k * P.STEP_H, "schody nahoru", prm))
        heel17 = ter.XS + (P.N_STEPS - 1) * P.STEP_G + P.HEEL_MARGIN
        seq.append((heel17 + 0.30, P.N_STEPS * P.STEP_H, "podesta", P.LAND))     # dva kratší kroky
        seq.append((ter.XD - 0.27, P.N_STEPS * P.STEP_H, "podesta", P.LAND))
        for m in range(1, P.N_STEPS + 1):
            seq.append((ter.XD + (m - 1) * P.STEP_G + P.DESC_MARGIN, (P.N_STEPS - m) * P.STEP_H,
                        "schody dolu", P.DOWN))
        seq.append((seq[-1][0], 0.0, "zastaveni", P.STOP))
        self.t_sts0 = P.T_SIT
        self.t_seatoff = P.T_SIT + P.STS_SEATOFF * P.T_STS
        self.t_stand = P.T_SIT + P.T_STS
        t_lift = self.t_stand + P.T_STAND
        self.steps = []
        for i, (x, surf, phase, prm) in enumerate(seq):
            side = "l" if i % 2 == 0 else "r"
            if i > 0:
                t_lift = self.steps[-1]["t_on"] + prm["ds"]
            self.steps.append(dict(i=i, side=side, heel=x, surf=surf, phase=phase,
                                   t_lift=t_lift, t_on=t_lift + prm["swing"]))
        self.t_end = self.steps[-1]["t_on"] + P.T_END_STAND
        # kontakty (stoj) a švihy
        self.contacts = {s: [dict(t_on=-np.inf, heel=P.X_FEET, surf=0.0, phase="sed")] for s in ("r", "l")}
        for st in self.steps:
            c = self.contacts[st["side"]]
            c[-1]["t_off"] = st["t_lift"]
            c.append(dict(t_on=st["t_on"], heel=st["heel"], surf=st["surf"], phase=st["phase"]))
        for s in ("r", "l"):
            self.contacts[s][-1]["t_off"] = np.inf
            for c in self.contacts[s]:
                c["base"] = np.array([c["heel"] - HEEL[0], c["surf"] + SOLE, 0.0])
                c["toe_first"] = c["phase"] == "schody dolu"
                c["heel_first"] = c["phase"] in ("rovina", "podesta")
                c["land"] = (self.toe_first_pose(c["base"], P.TOE_FIRST) if c["toe_first"] else
                             self.heel_first_pose(c["base"], P.HEEL_FIRST) if c["heel_first"] else c["base"])
            for c0, c1 in zip(self.contacts[s][:-1], self.contacts[s][1:]):
                c0["desc"] = c1["surf"] < c0["surf"] - 1e-6          # další nášlap je níž (sestup)
            self.contacts[s][-1]["desc"] = False
        self.swings = {s: [] for s in ("r", "l")}
        for s in ("r", "l"):
            cs = self.contacts[s]
            for c0, c1 in zip(cs[:-1], cs[1:]):
                self.swings[s].append(dict(t0=c0["t_off"], c0=c0, c1=c1))

    # ---------------------------------------------- pánev a trup
    def _pelvis_from_hip(self, hip, tilt):
        return np.asarray(hip) - rot2(tilt) @ self.hip_local

    def _plan_pelvis(self):
        Ys, h2p = self.Y_STAND, self.HEEL2PEL
        tilt_sit = self.tilt0 + 0.15
        hip_sit = np.array([P.HIP_SEAT_X, P.SEAT_H + P.HIP_ABOVE_SEAT])
        pel_stand = np.array([P.X_FEET + h2p, Ys - 0.005])
        hip_stand = pel_stand + rot2(self.tilt0) @ self.hip_local
        T0, T = P.T_SIT, P.T_STS
        # vstávání: body kyčle (t, x, y), náklon pánve a trupu
        k = [(0.0, *hip_sit, tilt_sit, self.trunk0),
             (T0, *hip_sit, tilt_sit, self.trunk0),
             (T0 + 0.25 * T, hip_sit[0] + 0.02, hip_sit[1], self.tilt0 - 0.25, self.trunk0 - 0.45 * P.STS_LEAN),
             (self.t_seatoff, hip_sit[0] + 0.10, hip_sit[1] + 0.003, self.tilt0 - 0.45, self.trunk0 - P.STS_LEAN),
             (T0 + 0.65 * T, 0.5 * (hip_sit[0] + hip_stand[0]) + 0.08, 0.5 * (hip_sit[1] + hip_stand[1]),
              self.tilt0 - 0.30, self.trunk0 - 0.6 * P.STS_LEAN),
             (T0 + 0.85 * T, hip_stand[0], hip_stand[1] - 0.03, self.tilt0 - 0.08, self.trunk0 - 0.12),
             (self.t_stand, *hip_stand, self.tilt0, self.trunk0)]
        k = np.array(k)
        self.sts = {j: pchip(k[:, 0], k[:, j]) for j in range(1, 5)}
        # chůze: body pánve v okamžicích dopadů (a středu stoje na rovině)
        px = [(self.t_stand, pel_stand[0]), (self.steps[0]["t_lift"], pel_stand[0])]
        py = [(self.t_stand, pel_stand[1]), (self.steps[0]["t_lift"], pel_stand[1])]
        other = {"r": P.X_FEET, "l": P.X_FEET}
        osurf = {"r": 0.0, "l": 0.0}
        for n, st in enumerate(self.steps):
            o = "r" if st["side"] == "l" else "l"
            x_c = 0.5 * (st["heel"] + other[o]) + h2p + (P.DX_CONTACT if st["phase"] in ("rovina", "podesta") else 0.0)
            if st["phase"] in ("rovina", "podesta"):
                y_c = st["surf"] + Ys + P.DY_LEVEL_DS
            elif st["phase"] == "schody nahoru" and osurf[o] == 0.0:
                # první nášlap na schod: zadní noha ještě na podlaze -> pánev jako na rovině
                x_c = 0.5 * (st["heel"] + other[o]) + h2p + P.DX_CONTACT - 0.10
                y_c = Ys + P.DY_LEVEL_DS + 0.02
            elif st["phase"] == "schody nahoru":
                x_c = st["heel"] - HEEL[0] - self.off_up[0] + P.UP_DX
                y_c = st["surf"] + SOLE - self.off_up[1] + P.UP_DY
            elif st["phase"] == "schody dolu":
                land = self.toe_first_pose(np.array([st["heel"] - HEEL[0], st["surf"] + SOLE, 0.0]), P.TOE_FIRST)
                x_c = land[0] - self.off_down[0]
                y_c = land[1] - self.off_down[1] + P.DOWN_DY
            else:                                      # zastavení: nohy u sebe
                y_c = st["surf"] + Ys - 0.02
            px.append((st["t_on"], x_c))
            py.append((st["t_on"], y_c))
            nxt = self.steps[n + 1] if n + 1 < len(self.steps) else None
            if st["phase"] in ("rovina", "podesta") and nxt is not None and nxt["phase"] in ("rovina", "podesta"):
                tm = 0.5 * (nxt["t_lift"] + nxt["t_on"])        # střed stoje: jen výška pánve
                py.append((tm, st["surf"] + Ys + P.DY_LEVEL_SS))
            other[st["side"]], osurf[st["side"]] = st["heel"], st["surf"]
        last = self.steps[-1]
        px.append((self.t_end, last["heel"] + h2p))
        py.append((self.t_end, Ys))
        px, py = np.array(sorted(px)), np.array(sorted(py))
        self.fx, self.fy = pchip(px[:, 0], px[:, 1]), pchip(py[:, 0], py[:, 1])
        # náklon trupu podle úseků (plynulé přechody 0,6 s)
        ph_t = {}
        for st in self.steps:
            ph_t.setdefault(st["phase"], st["t_on"])
        tk = [(self.t_stand, self.trunk0), (ph_t["schody nahoru"] - 0.6, self.trunk0),
              (ph_t["schody nahoru"], self.trunk0 - P.LEAN_UP),
              (ph_t["podesta"] - 0.3, self.trunk0 - P.LEAN_UP), (ph_t["podesta"] + 0.3, self.trunk0),
              (ph_t["schody dolu"], self.trunk0 - P.LEAN_DOWN),
              (ph_t["zastaveni"], self.trunk0 - P.LEAN_DOWN), (self.t_end, self.trunk0)]
        tk = np.array(tk)
        self.ftrunk = pchip(tk[:, 0], tk[:, 1])
        self.phase_t = ph_t

    def pelvis(self, t):
        """(x, y, tilt, lumbar) pánve v čase t (+ korekce rovnováhy těla po odlehnutí od židle)."""
        x, y, tilt, lum = self._pelvis_plan(t)
        if self.corr is not None and t > self.t_seatoff:
            w = smoothstep((t - self.t_seatoff) / 0.3)
            x += w * float(np.interp(t, self.corr[:, 0], self.corr[:, 1]))
        return x, y, tilt, lum

    def _pelvis_plan(self, t):
        if t <= self.t_stand:
            hip = np.array([float(self.sts[1](t)), float(self.sts[2](t))])
            tilt, trunk = float(self.sts[3](t)), float(self.sts[4](t))
            x, y = self._pelvis_from_hip(hip, tilt)
            return x, y, tilt, trunk - tilt
        trunk = float(self.ftrunk(t))
        return float(self.fx(t)), float(self.fy(t)), self.tilt0, trunk - self.tilt0

    def hip(self, t):
        x, y, tilt, _ = self.pelvis(t)
        return np.array([x, y]) + rot2(tilt) @ self.hip_local

    # ---------------------------------------------- stoj s odvalem
    def knee_target(self, c, t):
        if not np.isfinite(c["t_off"]):
            return P.KNEE_STANCE_MIN
        s = (t - (c["t_off"] - P.KNEE_PUSHOFF_TIME)) / P.KNEE_PUSHOFF_TIME
        return P.KNEE_STANCE_MIN + (P.KNEE_TOE_OFF - P.KNEE_STANCE_MIN) * smoothstep(s)

    def stance_pose(self, c, t):
        base = c["base"]
        if c.get("toe_first") and t < c["t_on"] + P.T_TOE_FIRST:
            return self.toe_first_pose(base, P.TOE_FIRST * (1 - smoothstep((t - c["t_on"]) / P.T_TOE_FIRST)))
        if c.get("heel_first") and t < c["t_on"] + P.T_HEEL_FIRST:
            return self.heel_first_pose(base, P.HEEL_FIRST * (1 - smoothstep((t - c["t_on"]) / P.T_HEEL_FIRST)))
        piv = base[:2] + rot2(base[2]) @ PIVOT
        hip = self.hip(t)
        d = self.kin.reach(self.knee_target(c, t))

        def pose(th):
            phi = base[2] - th
            return np.r_[piv - rot2(phi) @ PIVOT, phi]

        def dist(th):
            p = pose(th)
            return np.linalg.norm(p[:2] + rot2(p[2]) @ self.kin.ankle_in_calcn - hip)

        th0 = 0.0
        if c.get("desc") and np.isfinite(c["t_off"]):       # sestup: pata se zvedá postupně
            th0 = P.DESC_HEEL_RISE * smoothstep((t - (c["t_off"] - P.T_DESC_HEEL_RISE)) / P.T_DESC_HEEL_RISE)
        if dist(th0) <= d:
            return pose(th0)
        lo, hi = th0, 1.2
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if dist(mid) > d else (lo, mid)
        return pose(0.5 * (lo + hi))

    def _stance_state(self, c, h=1e-3):
        t = c["t_off"]
        p0, p1, p2 = (self.stance_pose(c, ti) for ti in (t - 2 * h, t - h, t))
        return p2, (p2 - p1) / h, (p2 - 2 * p1 + p0) / h ** 2

    # ---------------------------------------------- švih
    def swing_pose(self, sw, t, shape=None):
        T = sw["c1"]["t_on"] - sw["t0"]
        tt = np.asarray(t, float) - sw["t0"]
        s = tt / T
        if "coef" not in sw:
            p0, v0, a0 = self._stance_state(sw["c0"])
            p1 = sw["c1"]["land"]
            sw["coef"] = [quintic(p0[j], v0[j], a0[j], p1[j], 0, 0, T) for j in range(3)]
        cx, A, B = sw["shape"] if shape is None else shape
        cf = sw["coef"]
        x = polyval(cf[0], tt) + cx * bump(s)
        y = polyval(cf[1], tt) + bump(s) * (A + B * (0.5 - s))
        ph = polyval(cf[2], tt) + P.SWING_TOE_UP * bump(s)
        return np.array([x, y, ph])

    def _design_swing(self, sw):
        p0 = self._stance_state(sw["c0"])[0]
        dx = sw["c1"]["land"][0] - p0[0]
        ss = np.linspace(0, 1, 121)
        T = sw["c1"]["t_on"] - sw["t0"]
        h_start = sw["c0"]["surf"]
        best = None
        for frac in (0.0, -0.1, -0.2, -0.3, 0.1, 0.2, 0.3, 0.4, 0.5):
            for bfac in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0):
                for A in np.arange(0.0, 0.8, 0.005):
                    pose = self.swing_pose(sw, sw["t0"] + ss * T, (frac * dx, A, bfac * A))
                    c, s_ = np.cos(pose[2])[:, None], np.sin(pose[2])[:, None]
                    px = pose[0][:, None] + c * FOOT_PTS[:, 0] - s_ * FOOT_PTS[:, 1]
                    py = pose[1][:, None] + s_ * FOOT_PTS[:, 0] + c * FOOT_PTS[:, 1]
                    Hx = self.ter.height(px)
                    need = np.where(Hx > h_start + 1e-6, P.CLEARANCE_STEP, P.CLEARANCE_FLAT)
                    clr = np.min(py - Hx - need, axis=1)
                    mid = clr[(ss > 0.15) & (ss < 0.8)]
                    end = np.min(py - Hx, axis=1)[ss >= 0.8]
                    # začátek švihu se nekontroluje: tuhé chodidlo se při odrazu otáčí kolem
                    # hlaviček metatarzů a špička prstů krátce mírně "projde" podložkou
                    if mid.min() >= 0 and end.min() >= -1e-3:
                        cost = A + 0.3 * abs(frac) + 0.05 * abs(bfac)
                        if best is None or cost < best[0]:
                            best = (cost, (frac * dx, A, bfac * A), mid.min())
                        break
        if best is None:
            raise RuntimeError(f"Švih {sw['t0']:.2f} s ({sw['c1']['phase']}) nenašel bezkolizní dráhu")
        sw["shape"] = best[1]

    def foot(self, side, t):
        for c in self.contacts[side]:
            if c["t_on"] <= t <= c["t_off"]:
                return self.stance_pose(c, t), True
        for sw in self.swings[side]:
            if sw["t0"] <= t <= sw["c1"]["t_on"]:
                return self.swing_pose(sw, t), False
        raise RuntimeError(f"Chodidlo {side} v t={t} nemá pózu")


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    # náklon pánve a trupu při chůzi: průměr z naměřené chůze (test1, rovina)
    c1, k1 = read_mot(os.path.join(T5.BASE, "test1", "data", "walk_stairs_ik.mot"))
    lev = k1[:, 0] < 8.6
    ref_tilt = np.radians(k1[lev, c1.index("pelvis_tilt")].mean())
    ref_lum = np.radians(k1[lev, c1.index("lumbar_extension")].mean())
    kin = Kin(SRC_MODEL)
    gen = Generator(kin, ref_tilt, ref_lum)
    print(f"[{T5.JMENO}] Y_STAND {gen.Y_STAND:.3f} m; první stupeň x={gen.ter.XS:.2f}, podesta do x={gen.ter.XD:.2f}, "
          f"konec schodů x={gen.ter.X_END:.2f}; konec t={gen.t_end:.2f} s; kroků {len(gen.steps)}")
    print("úseky:", {k: round(v, 2) for k, v in gen.phase_t.items()}, "odlehnutí od židle", round(gen.t_seatoff, 2))

    time = np.arange(0.0, gen.t_end, 1.0 / P.FS)
    Q = np.zeros((len(time), len(COORDS)))
    stance = np.zeros((len(time), 2), bool)
    guess = {s: None for s in ("r", "l")}
    worst = 0.0
    errs = np.zeros((len(time), 2))
    for i, t in enumerate(time):
        x, y, tilt, lum = gen.pelvis(t)
        q = {"pelvis_tilt": tilt, "pelvis_tx": x, "pelvis_ty": y, "lumbar_extension": lum}
        kin.set(q)
        for si, side in enumerate(("r", "l")):
            pose, on = gen.foot(side, t)
            stance[i, si] = on
            sol, err = kin.solve_leg(side, pose, guess[side])
            # koleno smí jen do flexe (knee_angle <= 0); při přeskoku na větev s kolenem ohnutým
            # dozadu nebo při selhání Newtona se řeší znovu z analytického odhadu (větev s flexí)
            if err > 1e-3 or sol[1] > np.radians(1.0):
                kin.set(q)
                sol2, err2 = kin.solve_leg(side, pose, None)
                if sol2[1] <= np.radians(1.0) and (err2 < err + 2e-3 or sol[1] > np.radians(1.0)):
                    sol, err = sol2, err2
            worst = max(worst, err)
            errs[i, si] = err
            guess[side] = sol
            q.update({f"hip_flexion_{side}": sol[0], f"knee_angle_{side}": sol[1], f"ankle_angle_{side}": sol[2]})
        Q[i] = [q[c] for c in COORDS]
    # ojedinělé vzorky, kde Newton nekonvergoval, se nahradí interpolací v čase
    for si, side in enumerate(("r", "l")):
        bad = errs[:, si] > 0.002
        if bad.any() and (~bad).sum() > 2:
            for c in (f"hip_flexion_{side}", f"knee_angle_{side}", f"ankle_angle_{side}"):
                j = COORDS.index(c)
                Q[bad, j] = np.interp(time[bad], time[~bad], Q[~bad, j])
        print(f"  {side}: {bad.sum()} vzorků s chybou IK > 2 mm nahrazeno interpolací")
    print(f"Max chyba IK polohy chodidla (před náhradou): {worst * 1000:.2f} mm")
    for si, side in enumerate(("r", "l")):
        bad = errs[:, si] > 0.005
        if bad.any():
            edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
            segs = [(round(time[a], 2), round(time[b - 1], 2), round(errs[a:b, si].max() * 1000, 1))
                    for a, b in zip(edges[::2], edges[1::2])]
            print(f"  {side}: úseky s chybou > 5 mm (od, do, max mm): {segs[:12]}{' ...' if len(segs) > 12 else ''}")
    np.save(os.path.join(DATA, "ik_chyba.npy"), np.column_stack([time, errs]))
    deg = Q.copy()
    for j, c in enumerate(COORDS):
        if c in ROT:
            deg[:, j] = np.degrees(deg[:, j])
    write_mot(os.path.join(DATA, "pohyb_raw.mot"), ["time"] + COORDS, np.column_stack([time, deg]), "pohyb_raw")
    filt = butter_lowpass(deg, P.LOWPASS_HZ, P.FS)
    write_mot(os.path.join(DATA, "pohyb_ik.mot"), ["time"] + COORDS, np.column_stack([time, filt]), "pohyb_ik")

    meta = dict(
        t_end=gen.t_end, t_seatoff=gen.t_seatoff, t_stand=gen.t_stand, t_sts0=gen.t_sts0,
        phase_t=gen.phase_t, Y_STAND=gen.Y_STAND,
        chair=dict(seat_h=P.SEAT_H, x_front=P.HIP_SEAT_X + 0.13, depth=P.SEAT_DEPTH,
                   hip_x=P.HIP_SEAT_X, hip_y=P.SEAT_H + P.HIP_ABOVE_SEAT),
        terrain=dict(XS=gen.ter.XS, XD=gen.ter.XD, X_END=gen.ter.X_END, step_h=P.STEP_H, step_g=P.STEP_G,
                     n_steps=P.N_STEPS, blocks=gen.ter.blocks()),
        steps=[{k: v for k, v in st.items()} for st in gen.steps],
        contacts={s: [dict(t_on=c["t_on"] if np.isfinite(c["t_on"]) else None,
                           t_off=c["t_off"] if np.isfinite(c["t_off"]) else None,
                           heel=c["heel"], surf=c["surf"], phase=c["phase"]) for c in gen.contacts[s]]
                  for s in ("r", "l")},
        mass=kin.mass)
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=1)

    # kontrolní graf úhlů pravé nohy a výšky pánve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for j, (c, lab) in enumerate((("hip_flexion_r", "Kyčel P [°]"), ("knee_angle_r", "Koleno P [°]"),
                                  ("ankle_angle_r", "Kotník P [°]"))):
        axs[j].plot(time, deg[:, COORDS.index(c)], lw=0.8, color="#bbb", label="surová")
        axs[j].plot(time, filt[:, COORDS.index(c)], lw=1.4, color="#2a78d6", label="filtr 6 Hz")
        axs[j].set_ylabel(lab)
    axs[3].plot(time, Q[:, 2], color="#eb6834", label="pánev y")
    axs[3].plot(time, [gen.ter.height(x) for x in Q[:, 1]], color="#52514e", label="povrch pod pánví")
    axs[3].set_ylabel("Výška [m]")
    axs[3].legend(frameon=False)
    for a in axs:
        for ph, t0 in gen.phase_t.items():
            a.axvline(t0, color="#999", lw=0.6, ls=":")
        a.axvline(gen.t_seatoff, color="#c00", lw=0.6, ls=":")
    axs[0].legend(frameon=False)
    axs[3].set_xlabel("Čas [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "kontrola_pohybu.png"), dpi=110)
    plt.close(fig)
    return gen


if __name__ == "__main__":
    main()
