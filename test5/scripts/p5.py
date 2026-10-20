"""Parametry scénáře test5: vstávání ze židle, 5 m po rovině, 5 m do schodů, 5 m ze schodů.

Souřadnice: x vpřed [m], y nahoru [m]. Pata chodidel při sezení je v x = X_FEET.
"""
FS = 200.0                  # vzorkování pohybu [Hz]
LOWPASS_HZ = 6.0            # filtr kinematiky [Hz]

# ---------------- židle a sezení
X_FEET = 0.0                # poloha pat při sezení a ve stoji po vstání [m]
SEAT_H = 0.46               # výška sedáku [m]
HIP_ABOVE_SEAT = 0.085      # kyčelní kloub nad sedákem (sedací hrboly) [m]
HIP_SEAT_X = -0.36          # x kyčle při sezení [m]
SEAT_DEPTH = 0.42
T_SIT = 0.6                 # klidné sezení na začátku [s]
T_STS = 1.9                 # vstávání (sit-to-stand) [s]
STS_SEATOFF = 0.42          # odlehnutí od sedáku jako podíl T_STS
STS_LEAN = 0.62             # maximální předklon trupu při vstávání [rad] (~36°)
T_STAND = 0.45              # stání před první krokem [s]

# ---------------- trať
WALK_LEN = 5.0              # rovina: od pat ve stoji k prvnímu stupni [m]
STEP_H, STEP_G = 0.17, 0.29 # stupeň (jako test1)
N_STEPS = 17                # nahoru 17 stupňů (4,9 m), dolů 17 stupňů
LANDING = 0.90              # horní podesta [m]
HEEL_MARGIN = 0.02          # pata za hranou stupně (výstup)
DESC_MARGIN = 0.040         # pata za začátkem stupně (sestup; špička 0,5 cm před hranou)
FIRST_STEP_LEN = 0.62       # pata poslední nohy na rovině -> pata na 1. stupni [m]

# ---------------- kroky: (švih [s], dvojí opora [s], délka kroku [m])
FIRST_STEP = 0.40           # délka prvního kroku z místa [m]
LEVEL = dict(swing=0.42, ds=0.12, length=0.62)
FIRST = dict(swing=0.55, ds=0.14)
UP = dict(swing=0.50, ds=0.17)
LAND = dict(swing=0.46, ds=0.14)
DOWN = dict(swing=0.44, ds=0.14)
STOP = dict(swing=0.50, ds=0.0)
T_END_STAND = 0.7           # stání na konci [s]

# ---------------- pánev (výška vůči stoji s nataženýma nohama Y_STAND)
DY_LEVEL_DS, DY_LEVEL_SS = -0.065, -0.015     # rovina: dvojí opora / střed stoje
DX_CONTACT = 0.15            # pánev při dopadu před středem mezi patami (+ posun pata->pánev) [m]
HIP_IC_UP, KNEE_IC_UP = 36.0, 67.0       # póza vedoucí nohy při dopadu na schod nahoru [deg] (test1)
UP_DX, UP_DY = -0.03, -0.05              # posun pánve při dopadu nahoru (aby zadní noha dosáhla) [m]
HIP_IC_DOWN, KNEE_IC_DOWN = 8.0, 15.0    # dopad na schod dolů: tělo nad přední nohou, koleno mírně pokrčené [deg]
DOWN_DY = -0.02             # snížení pánve při dopadu dolů (švihová noha musí dosáhnout) [m]
TOE_FIRST = -0.25            # sestup: došlap na špičku, náklon chodidla při dopadu [rad] (~14°)
HEEL_FIRST = 0.30            # rovina: dopad na patu, špička nahoru [rad] (~17°)
T_HEEL_FIRST = 0.12          # doba položení špičky po dopadu na patu [s]
T_TOE_FIRST = 0.20           # doba položení paty po došlapu na špičku [s]
DESC_HEEL_RISE = 0.50        # sestup: zvednutí paty zadní nohy do odrazu [rad] (~29°)
T_DESC_HEEL_RISE = 0.40      # doba zvedání paty před odrazem [s]
TRUNK_WALK = -0.09          # náklon trupu vůči svislici při chůzi [rad] (−5°, + = dozadu)
LEAN_UP = 0.21              # předklon navíc do schodů [rad] (12°)
LEAN_DOWN = -0.05           # při sestupu trup mírně vzpřímeněji [rad]

# ---------------- stoj a švih chodidla
KNEE_STANCE_MIN = 5.0       # [deg]
KNEE_TOE_OFF = 40.0
KNEE_PUSHOFF_TIME = 0.30
SWING_TOE_UP = -0.15        # [rad]
CLEARANCE_STEP = 0.04       # vůle nad hranou stupně [m]
CLEARANCE_FLAT = 0.012      # vůle nad rovnou plochou [m]
