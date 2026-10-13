# Simulace: 10 m po rovině + výstup do schodů (OpenSim 4.5)

Cíl: čisté kloubní momenty pravé nohy (kyčel, koleno, kotník) během ~15 m chůze.

## Výsledky (`results/`)

| Soubor | Obsah |
|---|---|
| `prava_noha_momenty_15m.csv` | časová řada (200 Hz, oddělovač `;`): úsek, stoj/švih, úhly, momenty [Nm, Nm/kg], výkony [W], GRF |
| `souhrn_spicek.md` | špičkové momenty a výkony: rovina vs. schody |
| `graf1_momenty_15m.png` | momenty v kyčli, koleni a kotníku za celých 15 m |
| `graf2_cyklus_rovina_vs_schody.png` | úhly, momenty a výkony normované na krokový cyklus |
| `graf3_validace_GRF.png` | ověření metody dopočtu reakčních sil na rovině |
| `graf4_kinematika_15m.png` | úhel kyčle a kolene, úhlová rychlost stehna a lýtka za celých 15 m |
| `graf5_kinematika_cyklus.png` | totéž normované na krokový cyklus: rovina vs. schody |

Úhlové rychlosti stehna a lýtka jsou **absolutní** (vůči zemi, kolem příčné osy),
+ = segment se otáčí vpřed (ve směru flexe kyčle). V CSV jsou ve sloupcích `stehno_*` a `lytko_*`.
| `ID_walk_stairs.sto` | kompletní výstup OpenSim Inverse Dynamics (všechny klouby, obě nohy) |

Znaménka: kyčel + = flexní moment, koleno + = extenzní, kotník + = dorzální (− = plantární).
Výkon + = sval energii generuje, − = absorbuje.

## Scénář

* Model: `Gait10dof18musc / subject01` (72,6 kg, rovinný sagitální model, 10 stupňů volnosti).
* **Rovina 0–10,6 m:** naměřená chůze z ukázkových dat OpenSim (kinematika + silové plošiny,
  1,13 m/s), převedená z běžeckého pásu na chůzi po zemi a zopakovaná 7×.
* **Přechod:** zpomalení na schodovou rychlost a první dva kroky (rovina → 1. → 2. stupeň).
* **Schody:** 17 stupňů 170/290 mm (4,93 m vodorovně, 2,89 m výška), krok přes stupeň,
  cyklus 1,30 s. Konec simulace = levá noha na horní podestě.

## Metoda

1. `build_motion.py` – kinematika schodů se generuje: chodidla se kladou přesně na stupně,
   pánev se na každý krok zvedne o výšku stupně, švih se navrhne s vůlí 4 cm nad hranou
   a úhly kloubů se dopočtou inverzní kinematikou na modelu.
2. `tune_transition.py` – doladí zpomalení před schody tak, aby výsledné reakční síly
   působily uvnitř chodidla (fyzikálně proveditelný pohyb).
3. `compute_dynamics.py` – reakční síly od země na schodech nejsou naměřené, takže se dopočtou
   z dynamiky celého těla (ID bez vnějších sil → rozdělení mezi chodidla podle
   naměřených vzorů z roviny). Pak běží OpenSim **InverseDynamicsTool** s těmito silami.
4. `report.py` – CSV, grafy, souhrn.

Spuštění (OpenSim 4.5 vyžaduje Python 3.8):

```bash
cd scripts
PY=/c/Users/bubli/.pyenv/pyenv-win/versions/3.8.10/python.exe
$PY tune_transition.py && $PY build_motion.py && $PY compute_dynamics.py && $PY report.py
```

Parametry (výška a hloubka stupně, počet stupňů, časování, póza) jsou v `scripts/params.py`.

## Zobrazení v OpenSim GUI

1. File → Open Model → `model/subject01_stairs.osim` (model se schodištěm).
2. File → Load Motion → `data/walk_stairs_ik.mot`.
3. Pravým tlačítkem na pohyb → *Associate Motion Data* → `data/walk_stairs_grf.mot`,
   tím se zobrazí šipky reakčních sil.
4. ID lze znovu spustit z `setup/setup_ID_ID_walk_stairs.xml` (Tools → Inverse Dynamics → Load).

## Kontrola kvality

* IK: chodidla na stupních bez skluzu (chyba polohy 0,000 mm).
* Schody: zbytkové síly na pánvi 0 N, zbytkový moment 1,8 Nm RMS (rovina z měření: 17–22 N, 14 Nm).
* Validace dopočtu sil na rovině proti silovým plošinám: korelace momentů r = 0,82 (kyčel),
  0,96 (koleno), 0,98 (kotník); RMS rozdíl 0,13–0,21 Nm/kg.

## Omezení (důležité pro návrh exoskeletu)

* **Pohyb na schodech je syntetický**, není naměřený. Úhly odpovídají literatuře (koleno
  67° při došlapu → 9° → ~110° ve švihu, kotník −30° při odrazu), ale nejde o konkrétního člověka.
* **Extenzní moment kolene na schodech (~2,2 Nm/kg) je pravděpodobně nadhodnocený.**
  Měření v literatuře uvádějí typicky ~1,0–1,6 Nm/kg. Momenty v kyčli (−0,75 Nm/kg) a v kotníku
  (−1,56 Nm/kg) odpovídají literatuře lépe. Pro dimenzování kolene je to konzervativní odhad.
* Úsek **přechodu** (8,6–9,9 s) obsahuje artefakty z napojení naměřených a generovaných dat. Ber ho jen orientačně.
* Rovinný model = pouze sagitální momenty (flexe/extenze). Abdukce/rotace v kyčli chybí.
* Rozběh a zastavení nejsou modelované, chůze začíná a končí v ustáleném pohybu.
* Momenty se škálují zhruba s hmotností; pro jinou osobu použij sloupce Nm/kg.
