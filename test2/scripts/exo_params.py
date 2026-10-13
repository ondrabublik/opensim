"""Parametry exoskeletu pravého kolene: geometrie (CAD), materiály, řízení.

Souřadný systém kolene K (všechny rozměry CAD v metrech):
  počátek na ose kloubu exoskeletu v rovině symetrie kolene,
  x = vpřed (anteriorně), y = nahoru (proximálně), z = laterálně (pro pravou nohu ven).
Osy K jsou rovnoběžné s rámci femur_r a tibia_r modelu (při natažené noze).
"""
import os

TEST1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "test1")
BASE_MODEL = os.path.join(TEST1, "model", "subject01_stairs.osim")
SIDE = "r"

# ------------------------------------------------------------------ osa kloubu
# Anatomické koleno nemá pevnou osu (tibie se při flexi posouvá - valení/klouzání).
# Pevná osa exoskeletu se volí tak, aby její odchylka od kolene byla v tomto
# rozsahu flexe co nejmenší (nejmenší čtverce).
HINGE_FIT_RANGE_DEG = (0.0, 110.0)

# ------------------------------------------------------------------ geometrie
# Šířka kolene: kondyly femuru z = -44..+42 mm, kůže cca +-55 mm.
Z_SHANK_ARM = 0.067        # střed ramen bérce (lat +, med -) u kloubu [m]; vnitřní plocha 63,5 mm
Z_THIGH_ARM = 0.0745       # střed ramen stehna u kloubu (vně ramen bérce)
ARM_T = 0.007              # tloušťka ramen (ve směru z)
THIGH_ARM_W = 0.030        # šířka ramen stehna (ve směru x)
SHANK_ARM_W = 0.026        # šířka ramen bérce

# pohon (laterálně): BLDC motor + vlnový převod v jednom pouzdře
MOTOR_R = 0.042            # poloměr pouzdra [m]
MOTOR_Z = (0.071, 0.113)   # rozsah pouzdra v ose z (stator, spojený s ramenem stehna)
CAP_R, CAP_Z = 0.026, (0.113, 0.128)        # víčko s enkodérem a řídicí deskou
FLANGE_R, FLANGE_Z = 0.034, (0.0635, 0.0705)  # výstupní příruba (patří bérci)
MED_HINGE_R = 0.026        # mediální pasivní kloub (ložisko)

# objímka stehna (přední skořepina + zadní pásek)
THIGH_CUFF_Y = (0.165, 0.255)       # výšková poloha nad osou kloubu
THIGH_CENTER = (0.015, 0.025)       # střed průřezu stehna (x, z) v K
THIGH_R = 0.078                     # poloměr stehna v místě objímky (obvod ~49 cm)
# objímka bérce (zadní skořepina na lýtku + přední pásek)
SHANK_CUFF_Y = (-0.215, -0.135)
SHANK_CENTER = (-0.015, 0.012)
SHANK_R = 0.052                     # poloměr lýtka (obvod ~33 cm)
PAD_T = 0.005              # pěnová výstelka
SHELL_T = 0.0035           # skořepina objímky (kompozit)
STRAP_T = 0.002            # textilní pásek se suchým zipem
STRAP_H = 0.045            # šířka pásku
CUFF_WRAP_DEG = 12.0       # přesah skořepiny za rameno [deg]

# ------------------------------------------------------------------ materiály
DENSITY = {                        # [kg/m3]
    "hlinik_7075": 2810.0,         # ramena, příruba, mediální kloub
    "kompozit_CF": 1500.0,         # skořepiny objímek
    "pena_EVA": 120.0,             # výstelka
    "textil": 600.0,               # pásky
}
MOTOR_MASS = 0.72          # motor + vlnový převod + pouzdro [kg] (např. EC 45 flat + CSD-20)
CAP_MASS = 0.10            # enkodér, driver
COLORS = {                         # RGB 0..1 pro OpenSim
    "hlinik_7075": (0.72, 0.74, 0.78),
    "kompozit_CF": (0.16, 0.17, 0.19),
    "pena_EVA": (0.55, 0.62, 0.70),
    "textil": (0.08, 0.08, 0.09),
    "pohon": (0.90, 0.45, 0.10),
    "elektronika": (0.12, 0.12, 0.13),
}

# ------------------------------------------------------------------ řízení
# Moment exoskeletu je funkcí úhlu flexe kolene theta (0 = natažené, + = flexe):
#     tau = G(theta) * s(omega)
#   G(theta) ... tabulka (uzly po 10 deg, lineární interpolace) - "křivka asistence"
#   s(omega) ... v režimu "uhel" s = 1 (čistě funkce úhlu = nelineární pružina),
#                v režimu "uhel_smer" s = 0.5*(1 + tanh(omega/OMEGA0)), omega = rychlost
#                extenze [rad/s]; moment se zapíná jen když se koleno natahuje.
# + moment = extenzní (stejné znaménko jako knee_angle_r_moment v test1).
CONTROL_MODES = ("uhel", "uhel_smer")
CONTROL_MODE = "uhel_smer"          # režim zapsaný do hlavního modelu
CTRL_NODES_DEG = list(range(0, 121, 10))
TAU_MAX = 60.0             # max. moment pohonu [Nm] (vrcholový moment vlnového převodu)
ASSIST_RATIO = 0.5         # cíl ladění: exo má dodat tento podíl potřebného momentu kolene
OMEGA0 = 0.4               # šířka přechodu brány podle rychlosti [rad/s]
SMOOTH_LAMBDA = 2.0        # penalizace zakřivení křivky G při ladění
LEVEL_WEIGHT = 1.0         # váha roviny proti schodům při ladění (1 = podle času)
# Ruční křivka (Nm) pro uzly CTRL_NODES_DEG; None = vyladit automaticky z dat test1.
CTRL_TABLE_MANUAL = None
