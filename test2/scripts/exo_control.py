"""Řízení exoskeletu kolene jako funkce úhlu v koleni.

    tau = G(theta) * s(omega)        [Nm], + = extenzní moment

theta = flexe kolene [deg] (= -knee_angle_r), omega = rychlost extenze [rad/s]
(= d knee_angle_r / dt). G je po částech lineární tabulka v uzlech CTRL_NODES_DEG
(mimo rozsah konstantní), s = 1 (režim "uhel") nebo 0.5 (1 + tanh(omega / OMEGA0))
(režim "uhel_smer" - moment jen při natahování kolene).

Stejný zákon se zapisuje do modelu OpenSim jako ExpressionBasedCoordinateForce
(výraz v proměnných q = knee_angle_r [rad] a qdot [rad/s]), takže ho používá
inverzní dynamika i dopředná simulace v GUI.
"""
import numpy as np

import exo_params as E

NODES = np.asarray(E.CTRL_NODES_DEG, float)
FORCE_NAME = {m: f"exo_koleno_r_{m}" for m in E.CONTROL_MODES}


def gate(omega, mode):
    if mode == "uhel":
        return np.ones_like(np.asarray(omega, float))
    return 0.5 * (1.0 + np.tanh(np.asarray(omega, float) / E.OMEGA0))


def curve(theta_deg, table):
    return np.interp(theta_deg, NODES, table)


def torque(theta_deg, omega, table, mode):
    return curve(theta_deg, table) * gate(omega, mode)


def basis(theta_deg):
    """Matice B (n, k): G(theta) = B @ table (lineární interpolace = trojúhelníkové báze)."""
    th = np.clip(np.asarray(theta_deg, float), NODES[0], NODES[-1])
    B = np.zeros((len(th), len(NODES)))
    for j in range(len(NODES)):
        B[:, j] = np.interp(th, NODES, np.eye(len(NODES))[j])
    return B


def lepton_expression(table, mode):
    """Výraz pro ExpressionBasedCoordinateForce (q, qdot v rad, rad/s)."""
    f = np.radians(NODES)
    g = np.asarray(table, float)
    terms = [f"{g[0]:.6g}"]
    for i in range(len(f) - 1):
        m = (g[i + 1] - g[i]) / (f[i + 1] - f[i])
        if abs(m) > 1e-12:
            terms.append(f"{m:.8g}*(min(max(-q,{f[i]:.8g}),{f[i + 1]:.8g})-{f[i]:.8g})")
    G = "(" + "+".join(terms).replace("+-", "-") + ")"
    if mode == "uhel":
        return G
    return f"{G}*0.5*(1+tanh(qdot/{E.OMEGA0:g}))"


def tune(theta_deg, omega, target, mode, weights=None):
    """Najde tabulku G (0 <= G <= TAU_MAX) minimalizující
        sum w (target - G(theta) s(omega))^2 + lambda * sum (druhá diference G)^2
    (kvadratické programování s omezením na intervalu, projektovaný gradient s Nesterovem)."""
    B = basis(theta_deg) * gate(omega, mode)[:, None]
    w = np.ones(len(target)) if weights is None else weights
    w = w / w.sum()
    k = len(NODES)
    D2 = np.diff(np.eye(k), 2, axis=0)
    H = B.T @ (w[:, None] * B) + E.SMOOTH_LAMBDA * 1e-4 * D2.T @ D2
    b = B.T @ (w * target)
    step = 1.0 / np.linalg.eigvalsh(H).max()
    x = np.clip(np.linalg.lstsq(H, b, rcond=None)[0], 0, E.TAU_MAX)
    y, t = x.copy(), 1.0
    for _ in range(20000):
        xn = np.clip(y - step * (H @ y - b), 0, E.TAU_MAX)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = xn + (t - 1) / tn * (xn - x)
        if np.abs(xn - x).max() < 1e-9:
            x = xn
            break
        x, t = xn, tn
    return x
