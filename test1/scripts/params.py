"""Nastavení simulace: geometrie schodiště, časování a cílové úhly pro výstup do schodů."""

FS = 200.0                 # vzorkovací frekvence výstupních dat [Hz]
LOWPASS_HZ = 6.0           # filtr kinematiky před inverzní dynamikou [Hz]
SIDE = "r"                 # noha, pro kterou se exportují výsledky

# --- rovina: počet opakování naměřeného cyklu (1 cyklus ~ 1.40 m)
N_LEVEL_CYCLES = 7

# --- schodiště (ČSN 73 4130: 2h + b ~ 630 mm)
STEP_H = 0.17              # výška stupně [m]
STEP_G = 0.29              # hloubka stupně [m]
N_STEPS = 17               # počet stupňů (17 x 0.29 = 4.93 m vodorovně, 2.89 m výška)
HEEL_MARGIN = 0.02         # pata 2 cm za hranou stupně
FIRST_STEP_LEN = 0.62      # vzdálenost paty pravé (poslední na rovině) -> paty levé na 1. stupni

# --- časování (výstup do schodů: Riener 2002, Protopapadaki 2007)
SWING_1 = 0.46             # švih levé z roviny na 1. stupeň [s]
DS_1 = 0.15                # dvojí opora při nášlapu na 1. stupeň [s]
SWING_2 = 0.50             # švih pravé z roviny na 2. stupeň [s]
TRANS_V1 = 1.05            # rychlost pánve při dopadu levé na 1. stupeň [m/s]
TRANS_DX1 = -0.15           # posun pánve vpřed při dopadu na 1. stupeň oproti ustálenému vztahu [m]
STAIR_CYCLE = 1.30         # doba kroku (cyklu) na schodech [s]
STAIR_DS = 0.13            # dvojí opora jako podíl cyklu (2x => stoj 63 %)

# --- póza při dopadu na schod (vedoucí noha, chodidlo naplocho)
KNEE_IC = 67.0             # flexe kolene při dopadu [deg]
HIP_IC = 36.0              # flexe kyčle [deg]; None => dopočet z KNEE_RISE_END
KNEE_RISE_END = 10.0       # flexe kolene na konci kroku [deg] (jen pokud HIP_IC = None)
TRUNK_LEAN_EXTRA = 12.0    # předklon trupu navíc oproti rovině [deg]
PELVIS_TILT_EXTRA = 0.0    # přední klopení pánve navíc oproti rovině [deg]
RISE_BETA = 0.5            # zvedání pánve během kroku: 0 = rovnoměrně, 1 = plynulý rozjezd/dojezd

# --- odval chodidla (zvedání paty) řízené cílovou flexí kolene ve stoji
KNEE_STANCE_MIN = 5.0      # nejmenší flexe kolene ve stoji [deg]
KNEE_TOE_OFF = 40.0        # flexe kolene při odrazu [deg]
KNEE_PUSHOFF_TIME = 0.30   # doba nárůstu flexe před odrazem [s]

# --- švih
SWING_CLEARANCE = 0.04     # min. vůle chodidla nad hranou schodu [m]
SWING_TOE_UP = -0.15       # náklon chodidla ve švihu (záporně = špička dolů) [rad]
