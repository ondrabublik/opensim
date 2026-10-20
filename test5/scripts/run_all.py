"""Výpočet test5 pro jednu nebo více osob (Python 3.8 + OpenSim).

    python run_all.py                         # osoba z test5/osoba.json
    python run_all.py ../osoby/mala.json ...  # vybrané osoby
    python run_all.py --data ...              # jen data pro učení sítě (bez řízení exo)
    python run_all.py --rizeni ...            # jen řízení exo + grafy (data už spočtená, nová síť)

Kroky pro osobu:
  data:   s0 škálovaný model -> s1 pohyb -> s2 model s tratí a exo -> s3b pohyb + reakční síly
          s korekcí rovnováhy -> s8 export pro síť (neuronova_sit/data/<jmeno>.csv)
  řízení: s4 CNN + ID s vypnutým/zapnutým exo -> s5 grafy -> s6 data animace
Animaci vykreslí s7_anim_render.py (Python 3.11), síť naučí neuronova_sit/trenink.py (Python 3.11).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_STEPS = ["s0_osoba.py", "s1_pohyb.py", "s2_model.py", "s3b_rovnovaha.py", "s8_export_nn.py"]
CTRL_STEPS = ["s4_exo.py", "s5_grafy.py", "s6_anim_export.py"]


def run(script, osoba):
    env = dict(os.environ, OSOBA=os.path.abspath(osoba), PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout.splitlines():
        if "Couldn't find" not in line and "[error] Mesh" not in line:
            print("   " + line)
    if p.returncode:
        print(p.stderr[-3000:])
        raise SystemExit(f"{script} selhal pro {osoba}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    osoby = args or [os.path.join(ROOT, "osoba.json")]
    steps = DATA_STEPS + CTRL_STEPS
    if "--data" in sys.argv:
        steps = DATA_STEPS
    if "--rizeni" in sys.argv:
        steps = CTRL_STEPS
    for o in osoby:
        print(f"===== osoba {o} =====")
        for s in steps:
            print(f" -- {s}")
            run(s, o)


if __name__ == "__main__":
    main()
