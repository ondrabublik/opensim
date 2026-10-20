# Neuronová síť pro řízení exoskeletu (test5)

Ve složce se síť učí z dat chůze, která simulace test5 exportuje pro každou osobu. Naučené váhy
pak použije řízení exoskeletu v test5 (`osoba.json` → `exoskelet.vahy_site`).

```
data/<osoba>.csv, data/<osoba>.json   <- export ze simulace (test5/scripts/s8_export_nn.py)
nastaveni.json                        <- vstupy, štítek, augmentace, učení
trenink.py                            <- učení (Python 3.11 + PyTorch)
vahy/cnn_regrese.npz (+ .pt)         <- síť pro kompenzaci momentu (výchozí řízení)
vahy/cnn_vahy.npz (+ .pt)             <- síť ON/OFF (předchozí řízení)
vysledky/metriky_<typ>.json, *.png    <- vyhodnocení
```

## Data (`data/<osoba>.csv`, 200 Hz, oddělovač `;`)

| Sloupec | Význam |
|---|---|
| `usek` | sed + vstávání / rovina / schody nahoru / podesta / schody dolů / zastavení |
| `cas_s` | čas [s] |
| `stoj_prava` | 1 = pravé chodidlo na zemi |
| `koleno_flexe_deg`, `koleno_rychlost_deg_s` | flexe pravého kolene a její rychlost (+ = flexe) |
| `stehno_omega_deg_s`, `holen_omega_deg_s` | úhlová rychlost stehna a holeně vůči zemi (+ = dopředu), jako gyroskop IMU |
| `kycel_flexe_deg`, `kotnik_dorzalni_flexe_deg` | další úhly pravé nohy |
| `moment_kolene_Nm`, `moment_kolene_Nm_kg` | moment kolene člověka bez pomoci exoskeletu (+ = extenzní) |

V `data/<osoba>.json` je výška, hmotnost, délky segmentů, styl chůze, prostředí a časy úseků.

## Nastavení (`nastaveni.json`)

* `vstupy`: signály, které síť dostává. Řízení v test5 (`s4_exo.py`) posílá právě
  **úhel kolene, ω stehna a ω holeně** (snímače na exoskeletu), proto je pro řízení nech beze změny.
  Jiné vstupy lze použít pro experimenty s učením.
* `typ`: `"regrese"` (výchozí) nebo `"onoff"`.
* `regrese`: síť odhaduje **moment kolene, který má převzít motor, v N m/kg**. Řízení ho násobí
  hmotností uživatele.
  * `useky`: úseky, kde se pomáhá, a podíl kompenzace. Výchozí jsou `schody nahoru`,
    `sed + vstávání` a `schody dolů`, všude 1,0. Jinde je cíl 0.
  * `jen_extenze`: kompenzuje se jen extenzní moment (zvedání, brzdění).
  * `vaha_useku`: váha chyby při učení. Schody nahoru mají 2,0, protože jsou hlavní.
  * `predstih_s`: síť odhaduje moment o 0,05 s dopředu (vyrovná zpoždění řídicí smyčky a motoru).
  * `pokuta_prekompenzace`: přebytek (motor víc, než je potřeba) se trestá 3× víc než nedostatek.
* `stitek` (jen pro `onoff`): ON, když moment > `prah_Nm_kg` v `useky` a pravá noha stojí.
* `augmentace`: variace rychlosti chůze, zesílení, offsetu a šumu snímačů. Data jsou syntetická
  a pravidelná, proto je augmentace nutná.
* `uceni`: počet epoch, krok, velikost dávky, seed.

## Postup

```bash
# 1) data pro osoby (Python 3.8 + OpenSim): všechny záznamy z test5/osoby/*.json (3 osoby x 3 rychlosti)
cd ../scripts && python run_all.py --data ../osoby/*.json
# 2) učení (Python 3.11 + PyTorch)
cd ../neuronova_sit && py -3.11 trenink.py
# 3) řízení exoskeletu novou sítí a grafy
cd ../scripts && python run_all.py --rizeni ../osoby/*.json
```

Výsledek regrese na osobě, kterou síť neviděla: R² 0,76–0,96, chyba 0,15–0,35 N m/kg,
na rovině výstup 0,004–0,015 N m/kg ([`vysledky/metriky_regrese.json`](vysledky/metriky_regrese.json)).

`trenink.py` nejdřív vyhodnotí síť metodou **leave-one-person-out**: učí se na ostatních osobách (všechny jejich rychlosti; skupina = pole `skupina` v osoba.json)
a testuje na osobě, kterou nikdy neviděla. Potom naučí konečnou síť ze všech osob. Graf
`vysledky/graf_test_neviditelna_osoba.png` ukazuje štítek a výstup sítě pro každou testovací osobu.
