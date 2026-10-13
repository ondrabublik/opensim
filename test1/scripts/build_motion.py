"""Krok 1: sestaví kinematiku 15 m chůze (rovina z měření + výstup do schodů),
naměřené reakční síly pro rovinný úsek a model se schodištěm pro OpenSim GUI."""
import json
import shutil

from osimtools import *  # noqa: F401,F403
import params as P

DATA = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")
os.makedirs(DATA, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

GRF_COLS = ["ground_force_vx", "ground_force_vy", "ground_force_vz",
            "ground_force_px", "ground_force_py", "ground_force_pz",
            "1_ground_force_vx", "1_ground_force_vy", "1_ground_force_vz",
            "1_ground_force_px", "1_ground_force_py", "1_ground_force_pz",
            "ground_torque_x", "ground_torque_y", "ground_torque_z",
            "1_ground_torque_x", "1_ground_torque_y", "1_ground_torque_z"]
PIVOT = TOE - [0.01, 0.0]          # bod odvalu chodidla při zvednuté patě
CORR_FILE = os.path.join(DATA, "transition_correction.npy")


def crossings(t, f, thr=20.0):
    on = f > thr
    out = []
    for k in np.flatnonzero(np.diff(on.astype(int))):
        tc = t[k] + (thr - f[k]) / (f[k + 1] - f[k]) * (t[k + 1] - t[k])
        out.append((tc, "on" if on[k + 1] else "off"))
    return out


# ======================================================== naměřený cyklus
class Measured:
    def __init__(self, kin):
        cols, ik = read_mot(os.path.join(EXAMPLE_DIR, "IK", "subject01_walk_IK.mot"))
        assert cols[1:] == COORDS
        self.t = ik[:, 0]
        q = ik[:, 1:].copy()
        for j, c in enumerate(COORDS):
            if c in ROT:
                q[:, j] = np.radians(q[:, j])
        self.S = cubic_spline(self.t, q)
        gc, g = read_mot(os.path.join(EXAMPLE_DIR, "subject01_walk_grf.mot"))
        assert gc[1:] == GRF_COLS
        self.tg, self.g = g[:, 0], g[:, 1:]
        ev_r = crossings(self.tg, self.g[:, 1])
        ev_l = crossings(self.tg, self.g[:, 7])
        hs_r = [t for t, e in ev_r if e == "on"]
        self.t0, self.t1 = hs_r[0], hs_r[1]
        self.Tc = self.t1 - self.t0
        rel = lambda ev: [((t - self.t0), e) for t, e in ev if self.t0 - 1e-9 <= t < self.t1]
        self.ev = {"r": rel(ev_r), "l": rel(ev_l)}
        self.tau_lto = next(t for t, e in self.ev["l"] if e == "off")
        # rychlost pásu z pohybu chodidla ve stoji (naplocho)
        vs = []
        for side, (a, b) in (("r", (0.85, 1.25)), ("l", (1.45, 1.85))):
            tt = np.linspace(a, b, 20)
            xs = []
            for ti in tt:
                kin.set(dict(zip(COORDS, self.S(np.array([ti]))[0])))
                xs.append(kin.pos("calcn_" + side)[0])
            vs.append(-np.polyfit(tt, xs, 1)[0])
        self.vb = float(np.mean(vs))
        itx = COORDS.index("pelvis_tx")
        q0, q1 = self.S(np.array([self.t0, self.t1]))
        self.L = q1[itx] + self.vb * self.Tc - q0[itx]
        self.drift = q1 - q0
        self.drift[itx] = 0.0
        self.X0 = -q0[itx]
        g0 = np.array([np.interp(self.t0, self.tg, self.g[:, j]) for j in range(18)])
        g1 = np.array([np.interp(self.t1, self.tg, self.g[:, j]) for j in range(18)])
        self.gdrift = g1 - g0
        for j, c in enumerate(GRF_COLS):
            if "_p" in c:
                self.gdrift[j] = 0.0

    def q(self, t, der=0):
        """Souřadnice (a derivace) pro globální čas t (pole) - opakované cykly."""
        t = np.atleast_1d(t)
        i = np.floor(t / self.Tc + 1e-9)
        tau = t - i * self.Tc
        out = self.S(self.t0 + tau, der)
        itx = COORDS.index("pelvis_tx")
        if der == 0:
            out = out - np.outer(tau / self.Tc, self.drift)
            out[:, itx] += self.vb * tau + i * self.L + self.X0
        elif der == 1:
            out = out - self.drift / self.Tc
            out[:, itx] += self.vb
        return out

    def grf(self, t):
        t = np.atleast_1d(t)
        i = np.floor(t / self.Tc + 1e-9)
        tau = t - i * self.Tc
        out = np.column_stack([np.interp(self.t0 + tau, self.tg, self.g[:, j]) for j in range(18)])
        out -= np.outer(tau / self.Tc, self.gdrift)
        for j in (3, 9):  # COP x: pás -> chůze po zemi
            out[:, j] += self.vb * tau + i * self.L + self.X0
        return out


# ======================================================== schodiště
class Stairs:
    def __init__(self, x_s):
        self.x_s = x_s

    def height(self, x):
        j = np.floor((np.asarray(x) - self.x_s) / P.STEP_G) + 1
        return np.clip(j, 0, P.N_STEPS) * P.STEP_H

    def foot_base(self, k):
        """Póza calcn (x, y, phi) chodidla položeného na stupeň k (k > N => podesta)."""
        j = min(k, P.N_STEPS)
        x = self.x_s + (k - 1) * P.STEP_G + P.HEEL_MARGIN - HEEL[0]
        return np.array([x, j * P.STEP_H + SOLE, 0.0])

    def clearance(self, pose):
        pts = pose[:2] + FOOT_PTS @ rot2(pose[2]).T
        return np.min(pts[:, 1] - self.height(pts[:, 0]))


# ======================================================== generátor
class Generator:
    def __init__(self, kin, meas):
        self.kin, self.meas = kin, meas
        Tc = meas.Tc
        self.TA = P.N_LEVEL_CYCLES * Tc + meas.tau_lto
        TA = self.TA
        qA, vA, aA = (meas.q(TA, d)[0] for d in (0, 1, 2))
        self.qA = dict(zip(COORDS, qA))
        # stav chodidel v okamžiku odrazu levé
        kin.set(self.qA)
        self.hip_local = np.array([-0.0724, -0.0677])
        self._check_hip()
        base_r = kin.foot_pose("r")
        self.left_TO = self._foot_state_measured("l", TA)
        self.stairs = Stairs(base_r[0] + HEEL[0] + P.FIRST_STEP_LEN - P.HEEL_MARGIN)
        # časy dopadů t_k (k liché = levá, sudé = pravá)
        Th = P.STAIR_CYCLE / 2
        self.DS = P.STAIR_DS * P.STAIR_CYCLE
        self.k_max = P.N_STEPS          # poslední nášlap = levá na horní podestu
        t = {1: TA + P.SWING_1}
        t[2] = t[1] + P.DS_1 + P.SWING_2
        for k in range(3, self.k_max + 1):
            t[k] = t[2] + (k - 2) * Th
        self.tk = t
        self.t_end = t[self.k_max] + self.DS - 1e-3   # do odrazu pravé z posledního stupně
        # poloha pánve vůči vedoucímu chodidlu při dopadu (z cílových úhlů)
        self.tilt = float(np.mean(meas.S(meas.t)[:, 0])) - np.radians(P.PELVIS_TILT_EXTRA)
        self.lumbar = float(np.mean(meas.S(meas.t)[:, 9])) - np.radians(P.TRUNK_LEAN_EXTRA)
        self.v = P.STEP_G / Th
        # flexe kyčle při dopadu: zadaná, nebo tak, aby koleno na konci kroku mělo KNEE_RISE_END
        if P.HIP_IC is not None:
            self.hip_ic = P.HIP_IC
            off = self._ic_offset(self.hip_ic)
        else:
            rise = np.array([P.STEP_G, P.STEP_H])
            lo, hi = 0.0, 90.0
            for _ in range(40):
                self.hip_ic = 0.5 * (lo + hi)
                off = self._ic_offset(self.hip_ic)
                hip_end = rot2(self.tilt) @ self.hip_local + rise
                ankle = off[:2] + rot2(off[2]) @ kin.ankle_in_calcn
                if np.linalg.norm(ankle - hip_end) > kin.reach(P.KNEE_RISE_END):
                    lo = self.hip_ic
                else:
                    hi = self.hip_ic
        self.off = off[:2]                                     # calcn - pánev
        b2 = self.stairs.foot_base(2)
        self.X2 = b2[0] - off[0]
        self.Y0 = self.stairs.foot_base(1)[1] - off[1]
        # vodorovně: dva úseky zpomalování (odraz levé -> dopad na 1. stupeň -> dopad na 2. stupeň)
        self.X1 = self.stairs.foot_base(1)[0] - off[0] + P.TRANS_DX1
        self.cx = quintic(qA[1], vA[1], aA[1], self.X1, P.TRANS_V1, 0.0, t[1] - TA)
        self.cx2 = quintic(self.X1, P.TRANS_V1, 0.0, self.X2, self.v, 0.0, t[2] - t[1])
        # iterativní korekce (tune_transition.py), aby COP zůstalo na chodidle
        self.corr = None
        if os.path.exists(CORR_FILE):
            ct, cdx = np.load(CORR_FILE)
            self.corr = cubic_spline(ct, cdx)
        self.vy0 = (1 - P.RISE_BETA) * P.STEP_H / Th
        self.cy = quintic(qA[2], vA[2], aA[2], self.Y0, self.vy0, 0.0, t[1] - TA)
        self.ct = quintic(qA[0], vA[0], aA[0], self.tilt, 0.0, 0.0, t[2] - TA)
        self.cl = quintic(qA[9], vA[9], aA[9], self.lumbar, 0.0, 0.0, t[2] - TA)
        # kontakty (stoj) a švihy
        self.contacts = {"r": [], "l": []}
        self.contacts["r"].append(dict(t_on=-np.inf, t_off=t[1] + P.DS_1, base=base_r, k=0))
        for k in range(1, self.k_max + 1):
            side = "l" if k % 2 else "r"
            t_off = t[k + 1] + (P.DS_1 if k == 0 else self.DS) if k + 1 <= self.k_max else np.inf
            self.contacts[side].append(dict(t_on=t[k], t_off=t_off,
                                            base=self.stairs.foot_base(k), k=k))
        self.swings = {"r": [], "l": []}
        self.swings["l"].append(dict(t0=TA, state0=self.left_TO, c1=self.contacts["l"][0]))
        for side in ("r", "l"):
            cs = self.contacts[side]
            for c0, c1 in zip(cs[:-1], cs[1:]):
                if c0["t_off"] < np.inf:
                    st = self._stance_state(c0)
                    self.swings[side].append(dict(t0=c0["t_off"], state0=st, c1=c1))
        for side in ("r", "l"):
            for sw in self.swings[side]:
                self._design_swing(sw)

    def _ic_offset(self, hip_deg):
        """Póza calcn vůči pánvi při dopadu: daná flexe kyčle a kolene, chodidlo naplocho."""
        kin = self.kin
        kin.set({"pelvis_tilt": self.tilt, "pelvis_tx": 0.0, "pelvis_ty": 0.0,
                 "hip_flexion_r": np.radians(hip_deg), "knee_angle_r": -np.radians(P.KNEE_IC),
                 "ankle_angle_r": 0.0})
        kin.set({"ankle_angle_r": -kin.angle("calcn_r")})
        return kin.foot_pose("r")

    def _check_hip(self):
        q = dict(self.qA)
        self.kin.set(q)
        pel = np.array([q["pelvis_tx"], q["pelvis_ty"]])
        hip = pel + rot2(q["pelvis_tilt"]) @ self.hip_local
        assert np.allclose(hip, self.kin.pos("femur_r"), atol=1e-4), (hip, self.kin.pos("femur_r"))

    def _foot_state_measured(self, side, t, h=1e-3):
        poses = []
        for ti in (t - h, t, t + h):
            self.kin.set(dict(zip(COORDS, self.meas.q(ti)[0])))
            poses.append(self.kin.foot_pose(side))
        p0, p1, p2 = poses
        return p1, (p2 - p0) / (2 * h), (p2 - 2 * p1 + p0) / h ** 2

    # ---------------------------------------------- pánev a trup
    def pelvis(self, t):
        TA, tk = self.TA, self.tk
        if t <= tk[1]:
            x = polyval(self.cx, t - TA)
        elif t <= tk[2]:
            x = polyval(self.cx2, t - tk[1])
        else:
            x = self.X2 + self.v * (t - tk[2])
        if self.corr is not None and TA < t < tk[2]:
            x += float(self.corr(np.array([t]))[0])
        if t <= tk[1]:
            y = polyval(self.cy, t - TA)
        else:
            Th = P.STAIR_CYCLE / 2
            y = self.Y0
            for k in range(1, self.k_max + 1):
                dh = min(k, P.N_STEPS) * P.STEP_H - min(k - 1, P.N_STEPS) * P.STEP_H
                sk = (t - tk[k]) / Th
                y += dh * ((1 - P.RISE_BETA) * np.clip(sk, 0, 1) + P.RISE_BETA * smoothstep(sk))
        if t <= tk[2]:
            tilt, lum = polyval(self.ct, t - TA), polyval(self.cl, t - TA)
        else:
            tilt, lum = self.tilt, self.lumbar
        return x, y, tilt, lum

    def hip(self, t):
        x, y, tilt, _ = self.pelvis(t)
        return np.array([x, y]) + rot2(tilt) @ self.hip_local

    # ---------------------------------------------- stojná fáze s odvalem
    def knee_target(self, c, t):
        if not np.isfinite(c["t_off"]):
            return P.KNEE_STANCE_MIN
        s = (t - (c["t_off"] - P.KNEE_PUSHOFF_TIME)) / P.KNEE_PUSHOFF_TIME
        return P.KNEE_STANCE_MIN + (P.KNEE_TOE_OFF - P.KNEE_STANCE_MIN) * smoothstep(s)

    def stance_pose(self, c, t):
        base = c["base"]
        piv = base[:2] + rot2(base[2]) @ PIVOT
        hip = self.hip(t)
        d = self.kin.reach(self.knee_target(c, t))

        def pose(th):
            phi = base[2] - th
            cal = piv - rot2(phi) @ PIVOT
            return np.r_[cal, phi]

        def dist(th):
            p = pose(th)
            return np.linalg.norm(p[:2] + rot2(p[2]) @ self.kin.ankle_in_calcn - hip)

        if dist(0.0) <= d:
            return pose(0.0)
        lo, hi = 0.0, 1.2
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if dist(mid) > d:
                lo = mid
            else:
                hi = mid
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
            p0, v0, a0 = sw["state0"]
            p1 = sw["c1"]["base"]
            sw["coef"] = [quintic(p0[j], v0[j], a0[j], p1[j], 0, 0, T) for j in range(3)]
        cx, A, B = sw["shape"] if shape is None else shape
        cf = sw["coef"]
        x = polyval(cf[0], tt) + cx * bump(s)
        y = polyval(cf[1], tt) + bump(s) * (A + B * (0.5 - s))
        ph = polyval(cf[2], tt) + P.SWING_TOE_UP * bump(s)
        return np.array([x, y, ph])

    def _design_swing(self, sw):
        p0 = sw["state0"][0]
        dx = sw["c1"]["base"][0] - p0[0]
        ss = np.linspace(0, 1, 101)
        T = sw["c1"]["t_on"] - sw["t0"]
        h_start = self.stairs.height(p0[0] + HEEL[0] + 0.1)
        best = None
        for frac in (0.0, -0.1, -0.2, -0.3, -0.4, -0.5):
            for bfac in (0.0, 0.5, 1.0, 1.5):
                for A in np.arange(0.0, 0.6, 0.005):
                    pose = self.swing_pose(sw, sw["t0"] + ss * T, (frac * dx, A, bfac * A))
                    pts_x = pose[0][:, None] + np.cos(pose[2])[:, None] * FOOT_PTS[:, 0] \
                        - np.sin(pose[2])[:, None] * FOOT_PTS[:, 1]
                    pts_y = pose[1][:, None] + np.sin(pose[2])[:, None] * FOOT_PTS[:, 0] \
                        + np.cos(pose[2])[:, None] * FOOT_PTS[:, 1]
                    Hx = self.stairs.height(pts_x)
                    # nad vyššími stupni požadovaná vůle, nad výchozí plochou jen bez průniku
                    need = np.where(Hx > h_start + 1e-6, P.SWING_CLEARANCE, 0.005)
                    clr = np.min(pts_y - Hx - need, axis=1)
                    mid = clr[(ss > 0.12) & (ss < 0.75)]
                    end = np.min(pts_y - Hx, axis=1)[ss >= 0.75]
                    if mid.min() >= 0 and end.min() >= -1e-3:
                        cost = A + 0.3 * abs(frac)
                        if best is None or cost < best[0]:
                            best = (cost, (frac * dx, A, bfac * A), mid.min())
                        break
        if best is None:
            raise RuntimeError(f"Švih v čase {sw['t0']:.2f} s nenašel bezkolizní dráhu")
        sw["shape"] = best[1]
        sw["min_clearance"] = best[2]

    # ---------------------------------------------- celkový stav
    def foot(self, side, t):
        for c in self.contacts[side]:
            if c["t_on"] <= t <= c["t_off"]:
                return self.stance_pose(c, t), c
        for sw in self.swings[side]:
            if sw["t0"] <= t <= sw["c1"]["t_on"]:
                return self.swing_pose(sw, t), None
        raise RuntimeError(f"Chodidlo {side} nemá definovanou pózu v t={t}")


def add_stairs_to_model(stairs, src, dst):
    m = osim.Model(src)
    m.setName("subject01_stairs")
    ground = m.getGround()
    for j in range(1, P.N_STEPS + 1):
        depth = P.STEP_G if j < P.N_STEPS else 1.5
        hx, hy = depth / 2, j * P.STEP_H / 2
        cx = stairs.x_s + (j - 1) * P.STEP_G + hx
        fr = osim.PhysicalOffsetFrame(f"stair_{j:02d}", ground,
                                      osim.Transform(osim.Vec3(cx, hy, 0)))
        br = osim.Brick(osim.Vec3(hx, hy, 0.6))
        br.setColor(osim.Vec3(0.75, 0.7, 0.6))
        fr.attachGeometry(br)
        m.addComponent(fr)
    m.finalizeConnections()
    m.printToXML(dst)


def main():
    src_model = os.path.join(EXAMPLE_DIR, "subject01.osim")
    kin = Kin(src_model)
    meas = Measured(kin)
    print(f"Naměřený cyklus: T={meas.Tc:.3f} s, délka kroku (stride) {meas.L:.3f} m, "
          f"pás {meas.vb:.3f} m/s, odraz levé {meas.tau_lto:.3f} s po dopadu pravé")
    gen = Generator(kin, meas)
    st = gen.stairs
    print(f"Konec roviny (odraz levé) t={gen.TA:.3f} s, první stupeň x={st.x_s:.3f} m, "
          f"konec simulace t={gen.t_end:.2f} s")
    print(f"Pánev při dopadu vůči calcn vedoucí nohy: {gen.off}, flexe kyčle při dopadu {gen.hip_ic:.1f} deg, rychlost na schodech {gen.v:.3f} m/s")
    for side in ("l", "r"):
        for sw in gen.swings[side][:3] + gen.swings[side][-1:]:
            print(f"  švih {side} {sw['t0']:.2f}->{sw['c1']['t_on']:.2f} s na k={sw['c1']['k']}: "
                  f"tvar={np.round(sw['shape'], 3)}, vůle {sw['min_clearance']:.3f} m")

    time = np.arange(0.0, gen.t_end, 1.0 / P.FS)
    Q = np.zeros((len(time), len(COORDS)))
    contact = np.zeros((len(time), 2))
    ilev = time <= gen.TA + 1e-9
    Q[ilev] = meas.q(time[ilev])
    guess = {s: [meas.q(gen.TA)[0][COORDS.index(f"{n}_{s}")] for n in
                 ("hip_flexion", "knee_angle", "ankle_angle")] for s in ("r", "l")}
    worst = 0.0
    feet = {"r": [], "l": []}
    for i in np.flatnonzero(~ilev):
        t = time[i]
        x, y, tilt, lum = gen.pelvis(t)
        q = {"pelvis_tilt": tilt, "pelvis_tx": x, "pelvis_ty": y, "lumbar_extension": lum}
        kin.set(q)
        for si, side in enumerate(("r", "l")):
            pose, c = gen.foot(side, t)
            feet[side].append(pose)
            contact[i, si] = c is not None
            sol, err = kin.solve_leg(side, pose, guess[side])
            worst = max(worst, err)
            guess[side] = sol
            q.update({f"hip_flexion_{side}": sol[0], f"knee_angle_{side}": sol[1],
                      f"ankle_angle_{side}": sol[2]})
        Q[i] = [q[c] for c in COORDS]
    print(f"Max chyba IK polohy chodidla: {worst * 1000:.3f} mm")

    deg = Q.copy()
    for j, c in enumerate(COORDS):
        if c in ROT:
            deg[:, j] = np.degrees(deg[:, j])
    write_mot(os.path.join(DATA, "walk_stairs_raw.mot"), ["time"] + COORDS,
              np.column_stack([time, deg]), "walk_stairs_raw")
    filt = butter_lowpass(deg, P.LOWPASS_HZ, P.FS)
    write_mot(os.path.join(DATA, "walk_stairs_ik.mot"), ["time"] + COORDS,
              np.column_stack([time, filt]), "walk_stairs_ik")

    # naměřené GRF pro rovinný úsek (zbytek se dopočítá v kroku 2)
    G = np.zeros((len(time), 18))
    G[ilev] = meas.grf(time[ilev])
    np.save(os.path.join(DATA, "grf_measured_part.npy"), G)

    model_path = os.path.join(MODEL_DIR, "subject01_stairs.osim")
    add_stairs_to_model(st, src_model, model_path)

    # metadata pro další kroky
    lev_events = []
    for i in range(P.N_LEVEL_CYCLES + 1):
        for side in ("r", "l"):
            for tau, e in meas.ev[side]:
                tg = i * meas.Tc + tau
                if tg <= gen.TA + 1e-6:
                    lev_events.append((side, e, tg))
    meta = dict(
        TA=gen.TA, t_end=gen.t_end, Tc=meas.Tc, stride_level=meas.L, belt_speed=meas.vb,
        x_stairs=st.x_s, step_h=P.STEP_H, step_g=P.STEP_G, n_steps=P.N_STEPS,
        mass=kin.mass, t_k={str(k): v for k, v in gen.tk.items()},
        contacts={s: [dict(t_on=c["t_on"] if np.isfinite(c["t_on"]) else None,
                           t_off=c["t_off"] if np.isfinite(c["t_off"]) else None,
                           k=c["k"], surface=float(c["base"][1] - SOLE))
                      for c in gen.contacts[s]] for s in ("r", "l")},
        level_events=lev_events, stair_cycle=P.STAIR_CYCLE,
        swings={s: [dict(t0=sw["t0"], t1=sw["c1"]["t_on"], k=sw["c1"]["k"],
                         clearance=sw["min_clearance"]) for sw in gen.swings[s]]
                for s in ("r", "l")})
    with open(os.path.join(DATA, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("Uloženo:", os.path.join(DATA, "walk_stairs_ik.mot"), "|", model_path)
    return gen, worst


if __name__ == "__main__":
    main()
