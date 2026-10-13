# Exoskelet pravého kolene – souhrn

Model 72.6 kg + exoskelet 1.76 kg (stehno 1.34 kg, bérec 0.42 kg). Hodnoty jsou průměrem přes 6 cyklů. Moment + = extenzní; práce + = generace.

Řízení: τ = G(θ)·s(ω), ladění na 50% potřebného momentu, max 60 Nm, váha roviny 1.

## Moment kolene člověka

| Úsek | Varianta | Extenzní špička [Nm] | Flexní špička [Nm] | RMS [Nm] | Práce + [J/cyklus] | Práce − [J/cyklus] |
|---|---|---|---|---|---|---|
| Rovina | bez exo | 32.2 | -67.1 | 28.9 | 16.4 | -22.0 |
| Rovina | pasivní exo | 31.9 | -67.2 | 28.8 | 16.2 | -22.2 |
| Rovina | řízení: úhel | 29.6 | -67.2 | 30.5 | 24.8 | -30.8 |
| Rovina | řízení: úhel + směr | 31.4 | -67.2 | 30.7 | 15.5 | -39.7 |
| Schody | bez exo | 161.8 | -56.8 | 59.6 | 71.6 | -22.0 |
| Schody | pasivní exo | 162.0 | -56.4 | 59.6 | 71.9 | -22.3 |
| Schody | řízení: úhel | 122.7 | -62.7 | 47.4 | 71.7 | -22.2 |
| Schody | řízení: úhel + směr | 112.5 | -56.4 | 42.2 | 54.9 | -23.7 |

## Pohon exoskeletu

| Úsek | Řízení | Moment max [Nm] | Výkon max [W] | Práce + [J/cyklus] | Práce − [J/cyklus] | Max. rychlost při momentu > 5 Nm [rad/s] |
|---|---|---|---|---|---|---|
| Rovina | řízení: úhel | 46.5 | 182 | 12.9 | -12.9 | 6.7 |
| Rovina | řízení: úhel + směr | 59.0 | 235 | 18.3 | -0.0 | 7.1 |
| Schody | řízení: úhel | 46.9 | 61 | 13.5 | -13.5 | 7.5 |
| Schody | řízení: úhel + směr | 59.7 | 80 | 18.3 | -0.0 | 2.8 |

## Křivka asistence G(θ) [Nm]

| θ [deg] | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| řízení: úhel | 0.0 | 0.0 | 1.6 | 4.8 | 2.6 | 16.9 | 47.1 | 0.0 | 0.0 | 0.0 | 0.9 | 7.0 | 13.2 |
| řízení: úhel + směr | 0.0 | 0.0 | 3.0 | 1.4 | 7.7 | 31.1 | 60.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.9 | 11.7 |

## Kompromis rovina × schody (váha roviny při ladění)

| Řízení | Váha roviny | RMS rovina [Nm] | RMS schody [Nm] | Extenzní špička schody [Nm] |
|---|---|---|---|---|
| pasivní exo | – | 28.8 | 59.6 | 162.0 |
| řízení: úhel | 1 | 30.5 | 47.4 | 122.7 |
| řízení: úhel | 3 | 29.0 | 51.9 | 137.2 |
| řízení: úhel | 10 | 28.4 | 56.1 | 149.8 |
| řízení: úhel + směr | 1 | 30.7 | 42.2 | 112.5 |
| řízení: úhel + směr | 3 | 29.7 | 47.0 | 125.2 |
| řízení: úhel + směr | 10 | 28.8 | 53.6 | 143.5 |

## Osa kloubu

Pevná osa exo ve femur_r (-12.3, -468.9) mm; nesouosost s kolenem v rozsahu 0–110° max 2.0 mm, RMS 0.8 mm (osa v počátku tibie by měla max 28.0 mm).
