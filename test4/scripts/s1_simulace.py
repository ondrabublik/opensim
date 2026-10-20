"""Krok 1: dopředná simulace chůze s exoskeletem v uzavřené smyčce (OpenSim API).

Každou 1 ms:
  1. Člověk = regulátor s vypočteným momentem (jako CMC): požadované zrychlení
       udot* = udot_ref + Kp (q_ref - q) + Kd (u_ref - u),
     moment člověka spočte osim.InverseDynamicsSolver z aktuálního stavu modelu.
     Do svého vnitřního modelu člověk započítá moment exo podle scénáře
     (okamžitě / se zpožděním / vůbec) - zbytek je pro něj porucha.
  2. Každých 50 ms (20 Hz) CNN z posledních 1 s SIMULOVANÉHO stavu (úhel kolene,
     úhlová rychlost stehna a holeně z Body.getAngularVelocityInGround) rozhodne ON/OFF.
  3. Motor: náběh 1. řádu (20 ms) k 40 Nm / 0 Nm.
  4. Momenty se nastaví přes ScalarActuator.setOverrideActuation a osim.Manager
     zintegruje stav o 1 ms (Manager se vytváří pro každý krok, protože se mění diskrétní
     proměnné stavu).

Reakční síly země jsou předepsané z dat (model kontaktu chodidla se zemí není).
Výstup: results/sim_<scenar>.sto, results/sim_<scenar>_pohyb.mot (pro GUI).

    python s1_simulace.py [scenar ...]        (bez argumentu všechny)
"""
import collections
import sys
import time

from t4common import *  # noqa: F401,F403
from s3_cnn import CNN

SCENARE = {
    "bez_exo": dict(exo=False),
    "cnn_clovek_idealni": dict(exo=True, delay=0.0),
    "cnn_clovek_100ms": dict(exo=True, delay=0.1),
    "cnn_clovek_nekompenzuje": dict(exo=True, delay=None),
    "cnn_sum_snimacu": dict(exo=True, delay=0.1, noise=(1.0, 12.0)),
}


def simulate(name, exo=True, delay=0.0, noise=None, t_end=None, seed=0):
    model, names = build_forward_model()
    s = model.initSystem()
    cs = model.getCoordinateSet()
    coords = {n: cs.get(n) for n in names}
    fset = model.getForceSet()
    acts = {n: osim.ScalarActuator.safeDownCast(fset.get("clovek_" + n)) for n in names}
    motor = osim.ScalarActuator.safeDownCast(fset.get(MOTOR))
    femur = model.getBodySet().get(f"femur_{SIDE}")
    tibia = model.getBodySet().get(f"tibia_{SIDE}")
    tr, Q, U, A = reference(names, model)
    t_end = tr[-1] - 0.05 if t_end is None else t_end      # konec splajnu má okrajové jevy

    # počáteční stav = reference
    s.setTime(tr[0])
    for n in names:
        coords[n].setValue(s, float(Q[n][0]), False)
        coords[n].setSpeedValue(s, float(U[n][0]))
    for a in list(acts.values()) + [motor]:
        a.overrideActuation(s, True)
        a.setOverrideActuation(s, 0.0)
    # pořadí souřadnic ve vektoru zobecněných rychlostí (mobilit)
    uidx = {}
    save = {n: coords[n].getSpeedValue(s) for n in names}
    for n in names:
        for k in names:
            coords[k].setSpeedValue(s, 1.0 if k == n else 0.0)
        Uv = s.getU()
        uidx[n] = int(np.argmax([abs(Uv.get(i)) for i in range(Uv.size())]))
    for n in names:
        coords[n].setSpeedValue(s, save[n])
    model.realizeVelocity(s)
    s_sens = osim.State(s)                                   # stav pro výpočet snímačů

    ids = osim.InverseDynamicsSolver(model)
    Kp, Kd = WN ** 2, 2 * ZETA * WN
    net = CNN()
    rng = np.random.default_rng(seed)
    buf = None
    k_cnn = int(round(1 / (FS_CNN * DT_HUMAN)))
    a_mot = 1 - np.exp(-DT_HUMAN / T_MOTOR)
    hist = collections.deque()                              # historie momentu exo pro zpoždění člověka
    on, p_on, tau_exo = False, 0.0, 0.0
    x = np.zeros(3)
    log = collections.defaultdict(list)
    n_steps = int(round((t_end - tr[0]) / DT_HUMAN))
    tic = time.time()
    for i in range(n_steps):
        t = tr[0] + i * DT_HUMAN
        j = i                                                # index reference (stejná mřížka)
        model.realizeVelocity(s)
        q = {n: coords[n].getValue(s) for n in names}
        u = {n: coords[n].getSpeedValue(s) for n in names}
        # --- exoskelet: snímače ze simulovaného stavu, CNN 20 Hz, motor
        if exo:
            if i % k_cnn == 0:
                # Frame.getAngularVelocityInGround čte mezipaměť OpenSim, která u stavu
                # zkopírovaného z Manageru zůstává "platná" se starou hodnotou. Snímače se proto
                # počítají na samostatném stavu, do kterého se zkopírují q a u ze simulace.
                for n in names:
                    coords[n].setValue(s_sens, q[n], False)
                    coords[n].setSpeedValue(s_sens, u[n])
                model.realizeVelocity(s_sens)
                x = np.array([-np.degrees(q[KNEE]),
                              np.degrees(femur.getAngularVelocityInGround(s_sens).get(2)),
                              np.degrees(tibia.getAngularVelocityInGround(s_sens).get(2))])
                if noise:
                    x = x + rng.normal(0, 1, 3) * np.array([noise[0], noise[1], noise[1]])
                buf = np.tile(x, (net.win, 1)) if buf is None else np.vstack([buf[1:], x])
                p_on = net.prob(buf)
                on = p_on > 0.5
            tau_exo += a_mot * ((M_ON if on else 0.0) - tau_exo)
        hist.append((t, tau_exo))
        # --- člověk: co ví o momentu exo
        if delay is None or not exo:
            tau_est = 0.0
        else:
            while len(hist) > 1 and hist[1][0] <= t - delay + 1e-12:
                hist.popleft()
            tau_est = hist[0][1] if hist[0][0] <= t - delay + 1e-12 else 0.0
        udot = osim.Vector(len(names), 0.0)
        for n in names:
            udot.set(uidx[n], float(A[n][j] + Kp * (Q[n][j] - q[n]) + Kd * (U[n][j] - u[n])))
        for a in acts.values():
            a.setOverrideActuation(s, 0.0)
        motor.setOverrideActuation(s, float(tau_est))
        tau = ids.solve(s, udot)
        for n in names:
            acts[n].setOverrideActuation(s, float(tau.get(uidx[n])))
        motor.setOverrideActuation(s, float(tau_exo))
        # --- záznam (stav na začátku kroku, akční veličiny platné v kroku)
        log["time"].append(t)
        for n in names:
            log["q_" + n].append(q[n])
            log["u_" + n].append(u[n])
            log["chyba_" + n].append(q[n] - Q[n][j])
            log["clovek_" + n].append(tau.get(uidx[n]))
        log["exo_moment"].append(tau_exo)
        log["exo_on"].append(float(on))
        log["exo_p_on"].append(p_on)
        log["snimac_koleno_deg"].append(x[0])
        log["snimac_stehno_deg_s"].append(x[1])
        log["snimac_holen_deg_s"].append(x[2])
        log["clovek_odhad_exo"].append(tau_est)
        # --- integrace o 1 ms
        mgr = osim.Manager(model)
        mgr.setIntegratorAccuracy(1e-4)
        mgr.initialize(s)
        s = osim.State(mgr.integrate(t + DT_HUMAN))
    t_arr = np.array(log.pop("time"))
    tab = np_table(t_arr, {k: np.array(v) for k, v in log.items()})
    write_sto(tab, os.path.join(RES4, f"sim_{name}.sto"))
    # pohyb pro OpenSim GUI (stupně u rotací)
    mot = {n: np.array(log["q_" + n]) * (180 / np.pi if coords[n].getMotionType() == 1 else 1.0)
           for n in names}
    mt = np_table(t_arr[::5], {n: v[::5] for n, v in mot.items()}, meta={"inDegrees": "yes"})
    osim.STOFileAdapter.write(mt, os.path.join(RES4, f"sim_{name}_pohyb.mot"))
    err = np.degrees(np.abs(np.array(log["chyba_" + KNEE])))
    print(f"  {name}: {time.time() - tic:.0f} s výpočtu, max odchylka kolene {err.max():.2f}°, "
          f"motor zapnut {np.mean(log['exo_on']) * 100:.1f} % času")
    return tab


def main(argv=None):
    names = argv or list(SCENARE)
    for n in names:
        print(f"Scénář {n} ...")
        simulate(n, **SCENARE[n])


if __name__ == "__main__":
    main(sys.argv[1:])
