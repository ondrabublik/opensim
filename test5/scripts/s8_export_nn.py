"""Krok 8: export hodnot chůze pro učení neuronové sítě -> neuronova_sit/data/<jmeno>.csv

Sloupce (200 Hz):
  cas_s, usek, stoj_prava (1 = pravé chodidlo na zemi),
  koleno_flexe_deg, koleno_rychlost_deg_s (+ = flexe),
  stehno_omega_deg_s, holen_omega_deg_s (úhlová rychlost segmentu vůči zemi, + = dopředu; BodyKinematics),
  kycel_flexe_deg, kotnik_dorzalni_flexe_deg,
  moment_kolene_Nm, moment_kolene_Nm_kg (moment kolene člověka bez pomoci exo = s vypnutým exo, + extenzní)
Metadata osoby (výška, hmotnost, délky segmentů) jsou v neuronova_sit/data/<jmeno>.json.
"""
from t5common import *  # noqa: F401,F403
from s4_exo import sensors


def main():
    m = meta()
    t, uhel, w_st, w_ho = sensors()
    tk, Q = table_np(KIN)
    _, ID = table_np(os.path.join(RES, "ID_exo_vypnuty.sto"))
    n = min(len(t), len(ID[KNEE + "_moment"]))
    t = t[:n]
    stance = np.zeros(n, bool)
    for c in m["contacts"]["r"]:
        a = c["t_on"] if c["t_on"] is not None else -1e9
        b = c["t_off"] if c["t_off"] is not None else 1e9
        stance |= (t >= a) & (t <= b)
    usek = np.empty(n, dtype=object)
    for nm, a, b in phase_bounds(m):
        usek[(t >= a) & (t < b)] = nm
    usek[t >= phase_bounds(m)[-1][1]] = phase_bounds(m)[-1][0]
    mass = OSOBA["hmotnost_kg"] + EXO_MASS
    M = ID[KNEE + "_moment"][:n]
    cols = {
        "cas_s": t, "stoj_prava": stance.astype(int),
        "koleno_flexe_deg": uhel[:n], "koleno_rychlost_deg_s": np.gradient(uhel[:n], t),
        "stehno_omega_deg_s": w_st[:n], "holen_omega_deg_s": w_ho[:n],
        "kycel_flexe_deg": Q["hip_flexion_r"][:n], "kotnik_dorzalni_flexe_deg": Q["ankle_angle_r"][:n],
        "moment_kolene_Nm": M, "moment_kolene_Nm_kg": M / mass}
    out_dir = os.path.join(NN_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{JMENO}.csv")
    names = list(cols)
    with open(path, "w", encoding="utf-8") as f:
        f.write("usek;" + ";".join(names) + "\n")
        for i in range(n):
            f.write(f"{usek[i]};" + ";".join(
                f"{cols[c][i]:d}" if c == "stoj_prava" else f"{cols[c][i]:.4f}" for c in names) + "\n")
    L, k = segments()
    json.dump({"jmeno": JMENO, "skupina": OSOBA.get("skupina", JMENO), "vyska_m": OSOBA["vyska_m"], "hmotnost_kg": OSOBA["hmotnost_kg"],
               "hmotnost_s_exo_kg": mass, "segmenty_m": L, "styl_chuze": OSOBA.get("styl_chuze"),
               "prostredi": OSOBA.get("prostredi"), "fs_Hz": 200.0, "useky": phase_bounds(m)},
              open(os.path.join(out_dir, f"{JMENO}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{JMENO}] export pro síť: {os.path.relpath(path, ROOT)} ({n} vzorků, {t[-1]:.1f} s)")


if __name__ == "__main__":
    main()
