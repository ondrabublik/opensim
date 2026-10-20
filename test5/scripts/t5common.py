"""Společné nastavení test5 (vstávání ze židle, rovina, schody nahoru a dolů) - OpenSim API.

Osoba se zadává v JSON souboru (výchozí test5/osoba.json, jiný přes proměnnou prostředí OSOBA
nebo argument run_all.py). Všechny výstupy jsou v podsložkách podle "jmeno":
data/<jmeno>, model/<jmeno>, results/<jmeno>, setup/<jmeno>.
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T3_SCRIPTS = os.path.join(BASE, "test3", "scripts")
if T3_SCRIPTS not in sys.path:
    sys.path.insert(0, T3_SCRIPTS)
import t3common as T3  # noqa: E402
from t3common import osim, np, table_np, np_table, write_sto  # noqa: E402,F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSOBA_FILE = os.path.abspath(os.environ.get("OSOBA", os.path.join(ROOT, "osoba.json")))
OSOBA = json.load(open(OSOBA_FILE, encoding="utf-8"))
JMENO = OSOBA["jmeno"]
DATA, RES, MODEL_DIR, SETUP = (os.path.join(ROOT, d, JMENO) for d in ("data", "results", "model", "setup"))
for _d in (DATA, RES, MODEL_DIR, SETUP):
    os.makedirs(_d, exist_ok=True)
NN_DIR = os.path.join(ROOT, "neuronova_sit")          # učení sítě pro řízení exoskeletu

EXAMPLE_MODEL = os.path.join(os.path.expanduser("~"), "Documents", "OpenSim", "4.5", "Models",
                             "Gait10dof18musc", "subject01.osim")
PERSON_MODEL = os.path.join(MODEL_DIR, "osoba.osim")   # škálovaný model bez tratě a exo
KIN = os.path.join(DATA, "pohyb_ik.mot")
META = os.path.join(DATA, "meta.json")
GRF = os.path.join(DATA, "grf.mot")
LOADS = os.path.join(DATA, "grf.xml")
MODELS = {v: os.path.join(MODEL_DIR, f"vstavani_schody_exo_{v}.osim") for v in ("pasivni", "cnn")}
MAIN_MODEL = os.path.join(MODEL_DIR, "vstavani_schody_exo.osim")
KNEE, MOTOR, SIDE = T3.KNEE, T3.MOTOR, T3.SIDE
LOAD_BODIES = (("right", "calcn_r", ""), ("left", "calcn_l", "1_"), ("chair", "pelvis", "2_"))
PHASES = [("sed + vstávání", None), ("rovina", "rovina"), ("schody nahoru", "schody nahoru"),
          ("podesta", "podesta"), ("schody dolů", "schody dolu"), ("zastavení", "zastaveni")]

# ------------------------------------------------------------------ rozměry osoby
REF = dict(vyska=1.834, stehno=0.454, berec=0.425, chodidlo=0.245)   # výchozí model (subject01)
EXO_MASS = 1.76


def segments():
    """Délky segmentů [m] a měřítka vůči výchozímu modelu."""
    kH = OSOBA["vyska_m"] / REF["vyska"]
    seg = OSOBA.get("segmenty_m", {}) or {}
    L = {k: (seg.get(k) if seg.get(k) else REF[k] * kH) for k in ("stehno", "berec", "chodidlo")}
    k = {"vyska": kH, **{s: L[s] / REF[s] for s in L}}
    k["noha"] = (L["stehno"] + L["berec"]) / (REF["stehno"] + REF["berec"])
    return L, k


def apply_person(P):
    """Přizpůsobí parametry pohybu (modul p5) osobě a prostředí z osoba.json (idempotentní)."""
    if not hasattr(P, "_orig"):
        P._orig = {k: v for k, v in vars(P).items() if not k.startswith("_")}
    o = P._orig
    for k, v in o.items():
        setattr(P, k, dict(v) if isinstance(v, dict) else v)
    L, k = segments()
    st = OSOBA.get("styl_chuze", {})
    kl = k["noha"] if st.get("skalovat_s_delkou_nohy", True) else 1.0
    kt = np.sqrt(kl) * float(st.get("tempo", 1.0))           # dynamická podobnost: čas ~ sqrt(L)
    kd = float(st.get("delka_kroku", 1.0))                   # násobek délky kroku na rovině
    for n in ("FIRST_STEP_LEN", "DY_LEVEL_DS", "DY_LEVEL_SS", "DX_CONTACT", "UP_DX", "UP_DY", "DOWN_DY"):
        setattr(P, n, o[n] * kl)
    P.FIRST_STEP = o["FIRST_STEP"] * kl * kd
    P.HIP_SEAT_X = o["HIP_SEAT_X"] * k["stehno"]
    P.HIP_ABOVE_SEAT = o["HIP_ABOVE_SEAT"] * k["vyska"]
    for n in ("T_STS", "T_STAND", "KNEE_PUSHOFF_TIME", "T_TOE_FIRST", "T_HEEL_FIRST", "T_DESC_HEEL_RISE"):
        setattr(P, n, o[n] * kt)
    for n in ("LEVEL", "FIRST", "UP", "LAND", "DOWN", "STOP"):
        d = dict(o[n])
        for key in ("swing", "ds"):
            d[key] = d[key] * kt
        if "length" in d:
            d["length"] = d["length"] * kl * kd
        setattr(P, n, d)
    env = OSOBA.get("prostredi", {})
    P.SEAT_H = env.get("vyska_sedaku_m", o["SEAT_H"])
    P.WALK_LEN = env.get("delka_roviny_m", o["WALK_LEN"])
    P.STEP_H = env.get("vyska_stupne_m", o["STEP_H"])
    P.STEP_G = env.get("hloubka_stupne_m", o["STEP_G"])
    P.N_STEPS = int(env.get("pocet_stupnu", o["N_STEPS"]))
    P.LANDING = env.get("podesta_m", o["LANDING"])
    # pata při sestupu tak, aby špička zůstala na stupni
    P.DESC_MARGIN = min(o["DESC_MARGIN"], P.STEP_G - L["chodidlo"] - 0.005)
    return P


def foot_points():
    """Body chodidla v rámu calcn (pata, hlavičky metatarzů, špička) škálované s chodidlem."""
    kf = segments()[1]["chodidlo"]
    sole = 0.045 * kf
    return sole, np.array([0.0, -sole]), np.array([0.18 * kf, -sole]), np.array([0.245 * kf, -sole])


def cnn_weights():
    """Váhy sítě pro řízení: z osoba.json (vlastní síť z neuronova_sit/), jinak síť z test2."""
    p = (OSOBA.get("exoskelet", {}) or {}).get("vahy_site")
    if p:
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if os.path.exists(p):
            return p
        print(f"  [upozornění] váhy {p} neexistují -> použije se síť z test2")
    return T3.CNN_WEIGHTS


def meta():
    return json.load(open(META, encoding="utf-8"))


def phase_bounds(m):
    """[(název, t0, t1)] úseků pohybu (začátek = zvednutí nohy do prvního kroku úseku)."""
    starts = {}
    for st in m["steps"]:
        starts.setdefault(st["phase"], st["t_lift"])
    keys = [("sed + vstávání", 0.0)] + [(n, starts[k]) for n, k in PHASES[1:]]
    return [(n, t0, keys[i + 1][1] if i + 1 < len(keys) else m["t_end"]) for i, (n, t0) in enumerate(keys)]


def external_loads(xml_path, grf_file):
    el = osim.ExternalLoads()
    el.setName("GRF_a_zidle")
    for name, body, pref in LOAD_BODIES:
        ef = osim.ExternalForce()
        ef.setName(name)
        ef.set_applied_to_body(body)
        ef.set_force_expressed_in_body("ground")
        ef.set_point_expressed_in_body("ground")
        ef.set_force_identifier(pref + "ground_force_v")
        ef.set_point_identifier(pref + "ground_force_p")
        ef.set_torque_identifier(pref + "ground_torque_")
        el.cloneAndAppend(ef)
    el.setDataFileName(grf_file)
    el.printToXML(xml_path)
    return xml_path


def run_id(name, model, loads=None):
    t, _ = table_np(KIN)
    tool = osim.InverseDynamicsTool()
    tool.setName(name)
    tool.setModelFileName(model)
    tool.setCoordinatesFileName(KIN)
    tool.setLowpassCutoffFrequency(-1)
    tool.setStartTime(t[0])
    tool.setEndTime(t[-1])
    if loads:
        tool.setExternalLoadsFileName(loads)
    tool.setResultsDir(RES)
    tool.setOutputGenForceFileName(name + ".sto")
    excl = osim.ArrayStr()
    excl.append("Muscles")
    tool.setExcludedForces(excl)
    setup = os.path.join(SETUP, f"setup_ID_{name}.xml")
    tool.printToXML(setup)
    osim.InverseDynamicsTool(setup).run()
    return table_np(os.path.join(RES, name + ".sto"))


def run_analyze(name, model, analyses):
    t, _ = table_np(KIN)
    tool = osim.AnalyzeTool()
    tool.setName(name)
    tool.setModelFilename(model)
    tool.setCoordinatesFileName(KIN)
    tool.setLowpassCutoffFrequency(-1)
    tool.setInitialTime(t[0])
    tool.setFinalTime(t[-1])
    tool.setSolveForEquilibrium(False)
    tool.setResultsDir(RES)
    for a in analyses:
        tool.getAnalysisSet().cloneAndAppend(a)
    setup = os.path.join(SETUP, f"setup_Analyze_{name}.xml")
    tool.printToXML(setup)
    osim.AnalyzeTool(setup).run()
