"""Společné nastavení test3: cesty, načtení OpenSim API a převody tabulek OpenSim <-> numpy.

test3 = stejná úloha jako test2, ale všechny kroky simulace běží přes OpenSim API:
model, reakční síly (ExternalLoads), inverzní dynamika (InverseDynamicsTool), snímače
(AnalyzeTool + BodyKinematics), motor exoskeletu (CoordinateActuator + PrescribedController /
ExpressionBasedCoordinateForce), výkon motoru (AnalyzeTool + ForceReporter), vizualizace
(Simbody visualizer). numpy slouží jen k lineární algebře, inferenci sítě a vyhodnocení.
"""
import os
import sys

OPENSIM_SDK = r"C:\OpenSim 4.5\sdk\Python"
if OPENSIM_SDK not in sys.path:
    sys.path.insert(0, OPENSIM_SDK)
import opensim as osim  # noqa: E402
import numpy as np  # noqa: E402

osim.Logger.setLevelString("error")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.dirname(ROOT)
TEST1, TEST2 = os.path.join(BASE, "test1"), os.path.join(BASE, "test2")
DATA, RES, MODEL_DIR, SETUP = (os.path.join(ROOT, d) for d in ("data", "results", "model", "setup"))
for _d in (DATA, RES, MODEL_DIR, SETUP):
    os.makedirs(_d, exist_ok=True)

# vstupy převzaté z předchozích projektů
BASE_MODEL = os.path.join(TEST1, "model", "subject01_stairs.osim")    # model se schodištěm
KIN = os.path.join(TEST1, "data", "walk_stairs_ik.mot")                # kinematika 15 m chůze
GRF_BASE = os.path.join(TEST1, "data", "walk_stairs_grf.mot")          # reakční síly bez exo
META = os.path.join(TEST1, "data", "meta.json")                        # časování cyklů
CAD_PROPS = os.path.join(TEST2, "cad", "exo_hmotnosti.json")           # CAD: hmotnosti, STL
CAD_GEOM = os.path.join(TEST2, "model", "Geometry")
CTRL_TABLE = os.path.join(TEST2, "results", "rizeni_tabulka.json")     # křivka G(theta)
CNN_WEIGHTS = os.path.join(TEST2, "results", "cnn", "cnn_vahy.npz")    # natrénovaná CNN

SIDE = "r"
KNEE = f"knee_angle_{SIDE}"
MOTOR = "exo_motor"                     # CoordinateActuator na knee_angle_r
TAU_MAX = 60.0                          # [Nm]

MODELS = {name: os.path.join(MODEL_DIR, f"subject01_stairs_exo_{name}.osim")
          for name in ("pasivni", "uhel_smer", "cnn")}
MAIN_MODEL = os.path.join(MODEL_DIR, "subject01_stairs_exo.osim")
GRF_EXO = os.path.join(DATA, "walk_stairs_grf_exo.mot")
LOADS_BASE = os.path.join(DATA, "walk_stairs_grf.xml")
LOADS_EXO = os.path.join(DATA, "walk_stairs_grf_exo.xml")


# ------------------------------------------------------------------ tabulky
def table_np(path_or_table):
    """TimeSeriesTable (nebo soubor .sto/.mot) -> (čas, {sloupec: pole})."""
    tab = osim.TimeSeriesTable(path_or_table) if isinstance(path_or_table, str) else path_or_table
    t = np.array(tab.getIndependentColumn())
    M = tab.getMatrix().to_numpy()
    labels = list(tab.getColumnLabels())
    return t, {c: M[:, j] for j, c in enumerate(labels)}


def np_table(t, cols, meta=None):
    """(čas, {sloupec: pole}) -> TimeSeriesTable."""
    labels = list(cols)
    tab = osim.TimeSeriesTable()
    tab.setColumnLabels(labels)
    for i, ti in enumerate(t):
        tab.appendRow(float(ti), osim.RowVector([float(cols[c][i]) for c in labels]))
    for k, v in (meta or {}).items():
        tab.addTableMetaDataString(k, v)
    return tab


def write_sto(tab, path):
    osim.STOFileAdapter.write(tab, path)
    return path


# ------------------------------------------------------------------ nástroje OpenSim
def external_loads(xml_path, grf_file):
    """ExternalLoads pro obě chodidla, sestavené přes API (bez ručního XML)."""
    el = osim.ExternalLoads()
    el.setName("GRF")
    for name, body, pref in (("right", "calcn_r", ""), ("left", "calcn_l", "1_")):
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


def run_id(name, model, loads=None, t0=None, t1=None):
    """InverseDynamicsTool; svaly vyloučené (počítají se čisté momenty kloubů)."""
    t_kin, _ = table_np(KIN)
    tool = osim.InverseDynamicsTool()
    tool.setName(name)
    tool.setModelFileName(model)
    tool.setCoordinatesFileName(KIN)
    tool.setLowpassCutoffFrequency(-1)            # kinematika z test1 je vyfiltrovaná (6 Hz)
    tool.setStartTime(t_kin[0] if t0 is None else t0)
    tool.setEndTime(t_kin[-1] if t1 is None else t1)
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


def run_analyze(name, model, analyses, loads=None):
    """AnalyzeTool nad kinematikou (stavy z q(t), rychlosti derivací splajnu)."""
    t_kin, _ = table_np(KIN)
    tool = osim.AnalyzeTool()
    tool.setName(name)
    tool.setModelFilename(model)
    tool.setCoordinatesFileName(KIN)
    tool.setLowpassCutoffFrequency(-1)
    tool.setInitialTime(t_kin[0])
    tool.setFinalTime(t_kin[-1])
    tool.setSolveForEquilibrium(False)
    if loads:
        tool.setExternalLoadsFileName(loads)
    tool.setResultsDir(RES)
    for a in analyses:
        tool.getAnalysisSet().cloneAndAppend(a)
    setup = os.path.join(SETUP, f"setup_Analyze_{name}.xml")
    tool.printToXML(setup)
    osim.AnalyzeTool(setup).run()


def cycles(meta):
    """6 cyklů roviny a 6 ustálených cyklů schodů (pravá noha, dopad -> dopad)."""
    Tc = meta["Tc"]
    tk = {int(k): v for k, v in meta["t_k"].items()}
    return [(i * Tc, (i + 1) * Tc) for i in range(1, 7)], [(tk[k], tk[k + 2]) for k in range(4, 15, 2)]


def in_cycles(t, cyc):
    return np.any([(t >= a) & (t < b) for a, b in cyc], axis=0)
