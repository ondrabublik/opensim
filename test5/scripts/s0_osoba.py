"""Krok 0: škálovaný model osoby podle osoba.json (OpenSim API, Model.scale).

Měřítka segmentů: stehno (femur), bérec (tibia), chodidlo (talus, calcn, toes) podle zadaných
nebo z výšky odvozených délek; pánev a trup podle výšky. Hmotnost se nastaví na zadanou
a rozdělí mezi segmenty ve stejném poměru jako ve výchozím modelu (preserveMassDistribution).
Výstup: model/<jmeno>/osoba.osim
"""
from t5common import *  # noqa: F401,F403


def main():
    L, k = segments()
    model = osim.Model(EXAMPLE_MODEL)
    model.setName(f"osoba_{JMENO}")
    s = model.initSystem()
    factors = {"pelvis": k["vyska"], "torso": k["vyska"]}
    for side in ("r", "l"):
        factors[f"femur_{side}"] = k["stehno"]
        factors[f"tibia_{side}"] = k["berec"]
        for b in ("talus", "calcn", "toes"):
            factors[f"{b}_{side}"] = k["chodidlo"]
    ss = osim.ScaleSet()
    for body, f in factors.items():
        sc = osim.Scale()
        sc.setName(body)
        sc.setSegmentName(body)
        sc.setScaleFactors(osim.Vec3(f, f, f))
        sc.setApply(True)
        ss.cloneAndAppend(sc)
    model.scale(s, ss, True, float(OSOBA["hmotnost_kg"]))
    model.printToXML(PERSON_MODEL)
    # kontrola rozměrů
    m = osim.Model(PERSON_MODEL)
    s = m.initSystem()
    bs = m.getBodySet()
    p = {b: np.array([bs.get(b).getPositionInGround(s).get(i) for i in range(3)])
         for b in ("femur_r", "tibia_r", "talus_r")}
    print(f"Osoba '{JMENO}': výška {OSOBA['vyska_m']:.2f} m, hmotnost {m.getTotalMass(s):.1f} kg | stehno "
          f"{np.linalg.norm(p['femur_r'] - p['tibia_r']):.3f} m (cíl {L['stehno']:.3f}), bérec "
          f"{np.linalg.norm(p['tibia_r'] - p['talus_r']):.3f} m (cíl {L['berec']:.3f}), chodidlo {L['chodidlo']:.3f} m")
    return PERSON_MODEL


if __name__ == "__main__":
    main()
