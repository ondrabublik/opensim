"""Krok 3b: korekce rovnováhy syntetického pohybu (iterace s1 -> s3).

Zbytkový moment na pánvi znamená, že těžiště není nad oporou (působiště by muselo ležet mimo
chodidlo). Pánev (a s ní horní část těla) se proto posune vodorovně o dx = -zbytek / Fy
(vyhlazeno filtrem 1 Hz) a pohyb i reakční síly se přepočítají. Opakuje se N_ITER-krát.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from osimtools import butter_lowpass  # noqa: E402
import t5common as T5  # noqa: E402

DATA = T5.DATA
CORR = os.path.join(DATA, "korekce_panve.npy")
N_ITER, GAIN = 4, 0.8


def run(script):
    out = subprocess.run([sys.executable, os.path.join(HERE, script)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=HERE)
    lines = [l for l in out.stdout.splitlines() if "Couldn't find" not in l]
    if out.returncode:
        print(out.stderr[-2000:])
        raise SystemExit(out.returncode)
    return lines


def main():
    if os.path.exists(CORR):
        os.remove(CORR)
    for it in range(N_ITER + 1):
        run("s1_pohyb.py")
        lines = run("s3_grf.py")
        print(f"--- iterace {it}: " + " | ".join(l.strip() for l in lines if "RMS" in l))
        if it == N_ITER:
            break
        r = np.load(os.path.join(DATA, "grf_zbytek_momentu.npy"))
        t, resid = r[:, 0], r[:, 1]
        fy = 9.81 * (T5.OSOBA["hmotnost_kg"] + T5.EXO_MASS)
        dx = butter_lowpass(-GAIN * resid / fy, 1.0, 200.0)
        if os.path.exists(CORR):
            c = np.load(CORR)
            dx = dx + np.interp(t, c[:, 0], c[:, 1])
        np.save(CORR, np.column_stack([t, dx]))
        print(f"    korekce pánve: max |dx| {np.abs(dx).max() * 100:.1f} cm")


if __name__ == "__main__":
    main()
