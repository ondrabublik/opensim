"""Výpočet konvoluční sítě v čistém numpy (pro řídicí smyčku exoskeletu, bez PyTorch).

Formát vah: npz z neuronova_sit/trenink.py (nebo test2/cnn_train.py): c1..c3 (Conv1D), f1..f2 (Dense),
scale (normalizace vstupů), fs, win, typ ("onoff" -> P(ON), "regrese" -> moment [N m/kg]).
"""
import numpy as np


class CNN:
    def __init__(self, path):
        d = np.load(path)
        self.w = {k: d[k] for k in d.files}
        self.scale, self.win, self.fs = d["scale"], int(d["win"]), float(d["fs"])
        self.typ = str(d["typ"]) if "typ" in d.files else "onoff"
        self.jen_extenze = bool(d["jen_extenze"]) if "jen_extenze" in d.files else True

    @staticmethod
    def conv(x, w, b):
        K = w.shape[2]
        L = x.shape[1] - K + 1
        return np.einsum("oik,ilk->ol", w, np.stack([x[:, k:k + L] for k in range(K)], 2)) + b[:, None]

    def raw(self, window):
        """window (win, n_vstupů) ve fyzikálních jednotkách -> výstup poslední vrstvy."""
        w = self.w
        x = (np.asarray(window) / self.scale).T
        for c in ("c1", "c2", "c3"):
            x = np.maximum(self.conv(x, w[c + ".weight"], w[c + ".bias"]), 0)
        x = np.maximum(w["f1.weight"] @ x.reshape(-1) + w["f1.bias"], 0)
        return float(w["f2.weight"] @ x + w["f2.bias"])

    def output(self, window):
        """onoff: P(ON); regrese: moment pro motor [N m/kg] (při jen_extenze >= 0)."""
        z = self.raw(window)
        if self.typ == "regrese":
            return max(z, 0.0) if self.jen_extenze else z
        return 1.0 / (1.0 + np.exp(-z))
