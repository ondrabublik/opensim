"""Řízení motoru ON/OFF konvoluční sítí (inference v čistém numpy, bez PyTorch).

Síť natrénuje cnn_train.py (váhy results/cnn/cnn_vahy.npz). Řídicí smyčka běží 20 Hz:
každých 50 ms se ze vstupů za poslední 1 s (20 vzorků) spočte pravděpodobnost ON,
výstup se drží do dalšího rozhodnutí (tvarovač nultého řádu).

    python run_moje_rizeni.py cnn_rizeni
"""
import os

import numpy as np

MOMENT_ZAPNUTO = 40.0      # moment zapnutého motoru [Nm] (= práh momentu pro štítky)
TAU_MAX = 60.0
TAU_MOTOR = 0.02           # časová konstanta náběhu momentu [s]
PRAH_P = 0.5               # práh pravděpodobnosti pro ON

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "cnn", "cnn_vahy.npz")


def conv1d(x, w, b):
    """x (C_in, L), w (C_out, C_in, K) -> (C_out, L-K+1), bez paddingu (jako torch Conv1d)."""
    K = w.shape[2]
    L = x.shape[1] - K + 1
    cols = np.stack([x[:, k:k + L] for k in range(K)], axis=2)      # (C_in, L, K)
    return np.einsum("oik,ilk->ol", w, cols) + b[:, None]


class CNN:
    def __init__(self, path=WEIGHTS):
        d = np.load(path)
        self.w = {k: d[k] for k in d.files}
        self.scale = d["scale"]
        self.win = int(d["win"])
        self.fs = float(d["fs"])

    def prob(self, window):
        """window (win, 3) fyzikální jednotky -> P(ON)."""
        w = self.w
        x = (np.asarray(window) / self.scale).T
        x = np.maximum(conv1d(x, w["c1.weight"], w["c1.bias"]), 0)
        x = np.maximum(conv1d(x, w["c2.weight"], w["c2.bias"]), 0)
        x = np.maximum(conv1d(x, w["c3.weight"], w["c3.bias"]), 0)
        x = np.maximum(w["f1.weight"] @ x.reshape(-1) + w["f1.bias"], 0)
        z = float(w["f2.weight"] @ x + w["f2.bias"])
        return 1.0 / (1.0 + np.exp(-z))


class Rizeni:
    def __init__(self):
        self.net = CNN()
        self.buf = None
        self.t_next = None
        self.on = False
        self.p = 0.0

    def krok(self, t, uhel_kolene, omega_stehno, omega_holen, dt):
        x = np.array([uhel_kolene, omega_stehno, omega_holen])
        if self.buf is None:                       # na startu okno vyplněné prvním vzorkem
            self.buf = np.tile(x, (self.net.win, 1))
            self.t_next = t
        if t + 1e-9 >= self.t_next:                # vzorkování 20 Hz
            self.buf = np.vstack([self.buf[1:], x])
            self.p = self.net.prob(self.buf)
            self.on = self.p > PRAH_P
            self.t_next += 1.0 / self.net.fs
        return self.on
