"""VLASTNÍ ŘÍZENÍ EXOSKELETU – šablona k úpravě.

Spouští se:  python run_moje_rizeni.py            (použije tento soubor)
             python run_moje_rizeni.py jiny_soubor (jiný soubor se stejnou třídou Rizeni)

Simulace volá Rizeni.krok() postupně pro každý vzorek (200 Hz, dt = 5 ms), stejně
jako by běžela smyčka v řídicí jednotce. Hodnoty z minulých kroků si můžeš ukládat
do self (stavový automat, filtry, časovače ...).

VSTUPY (každý krok):
  t               čas [s]
  uhel_kolene     flexe kolene [deg]; 0 = natažené, + = pokrčené
  omega_stehno    úhlová rychlost stehna vůči zemi [deg/s]
  omega_holen     úhlová rychlost holeně vůči zemi [deg/s]
                  (+ = segment se otáčí dopředu, tj. ve směru flexe kyčle;
                   rychlost natahování kolene = omega_holen - omega_stehno)
  dt              krok [s]

VÝSTUP – jedno z:
  číslo           požadovaný moment motoru [Nm], + = extenzní (pomáhá natáhnout koleno),
                  - = flexní
  True / False    motor zapnut / vypnut; zapnutý motor dává MOMENT_ZAPNUTO

Požadavek ještě prochází modelem motoru (omezení TAU_MAX, setrvačnost regulace
s časovou konstantou TAU_MOTOR), viz konec souboru.
"""

# ---------------------------------------------------------------- parametry motoru
MOMENT_ZAPNUTO = 40.0      # moment při výstupu True [Nm]
TAU_MAX = 60.0             # omezení momentu motoru [Nm]
TAU_MOTOR = 0.02           # časová konstanta náběhu momentu [s] (0 = okamžitě)


class Rizeni:
    """Příklad: motor ON/OFF pro výstup do schodů.

    Zapne se, když vypadá situace jako stoj na schodu s ohnutým kolenem:
      * koleno pokrčené víc než ZAP_UHEL,
      * stehno se otáčí dozadu (kyčel se napíná, tělo jde nahoru),
      * holeň se otáčí pomalu (chodidlo stojí na zemi; ve švihu se holeň točí rychle).
    Vypne se, když se koleno natáhne pod VYP_UHEL nebo se holeň roztočí (hystereze).
    """

    ZAP_UHEL = 35.0        # [deg]
    VYP_UHEL = 15.0        # [deg]
    STEHNO_DOZADU = -20.0  # [deg/s] stehno musí jít dozadu rychleji než tohle
    HOLEN_KLID = 60.0      # [deg/s] |omega holeně| pod touto hodnotou = noha stojí

    def __init__(self):
        self.zapnuto = False

    def krok(self, t, uhel_kolene, omega_stehno, omega_holen, dt):
        noha_stoji = abs(omega_holen) < self.HOLEN_KLID
        if not self.zapnuto:
            if uhel_kolene > self.ZAP_UHEL and omega_stehno < self.STEHNO_DOZADU and noha_stoji:
                self.zapnuto = True
        else:
            if uhel_kolene < self.VYP_UHEL or abs(omega_holen) > 2 * self.HOLEN_KLID:
                self.zapnuto = False
        return self.zapnuto

        # --- příklad výstupu jako moment místo ON/OFF (smaž "return" výše):
        # if self.zapnuto:
        #     return min(TAU_MAX, 1.2 * uhel_kolene)     # víc pokrčené => větší pomoc
        # return 0.0
