# test4 – dopředná simulace v uzavřené smyčce s CNN: souhrn

Průměr přes 6 cyklů pravé nohy. Moment kolene člověka = aktuátor `clovek_knee_angle_r` (regulátor s vypočteným momentem, InverseDynamicsSolver). Shoda ON/OFF = podíl času, kdy se rozhodnutí CNN v uzavřené smyčce shoduje s otevřenou smyčkou (test3).

| Scénář | Úsek | Ext. špička kolene [Nm] | RMS [Nm] | Kladná práce člověka [J] | Odchylka kolene RMS / max [°] | Motor ON [%] | Shoda ON/OFF [%] | Práce motoru [J] | Reziduum pánve RMS [N] |
|---|---|---|---|---|---|---|---|---|---|
| bez exo | Rovina | 32.0 | 28.8 | 16.1 | 0.02 / 0.07 | 0.0 | – | 0.0 | 26.4 |
| bez exo | Schody | 162.1 | 59.7 | 72.2 | 0.06 / 0.12 | 0.0 | – | 0.0 | 0.1 |
| CNN, člověk kompenzuje ihned | Rovina | 32.0 | 28.8 | 16.1 | 0.02 / 0.07 | 0.0 | 100.0 | 0.0 | 26.4 |
| CNN, člověk kompenzuje ihned | Schody | 122.3 | 44.5 | 53.1 | 0.06 / 0.12 | 23.1 | 99.7 | 19.7 | 0.1 |
| CNN, člověk kompenzuje za 100 ms | Rovina | 32.0 | 28.8 | 16.1 | 0.02 / 0.07 | 0.0 | 100.0 | 0.0 | 26.4 |
| CNN, člověk kompenzuje za 100 ms | Schody | 112.0 | 42.3 | 51.3 | 2.67 / 8.67 | 23.1 | 99.7 | 18.6 | 5.0 |
| CNN, člověk nekompenzuje | Rovina | 32.0 | 28.8 | 16.1 | 0.02 / 0.07 | 0.0 | 100.0 | 0.0 | 26.4 |
| CNN, člověk nekompenzuje | Schody | 112.1 | 41.8 | 64.5 | 3.38 / 9.37 | 19.2 | 96.2 | 20.1 | 0.8 |
| CNN + šum snímačů (100 ms) | Rovina | 32.0 | 28.8 | 16.1 | 0.02 / 0.07 | 0.0 | 100.0 | 0.0 | 26.4 |
| CNN + šum snímačů (100 ms) | Schody | 112.0 | 42.1 | 48.6 | 2.76 / 9.49 | 25.0 | 97.8 | 21.7 | 5.0 |

## Kontroly

| Kontrola | Hodnota |
|---|---|
| bez exo vs ID (test3): RMS [Nm] | 1.62 |
| bez exo vs ID (test3): 99. percentil absolutního rozdílu [Nm] | 6.70 |
| ideální kompenzace vs ID s CNN (test3): RMS [Nm] | 1.80 |
| ideální kompenzace vs ID s CNN (test3): 99. percentil absolutního rozdílu [Nm] | 7.44 |
| ideální kompenzace: shoda ON/OFF s otevřenou smyčkou [%] | 99.84 |

Ojedinělé špičky rozdílu (až ~30 Nm po dobu 1-3 ms) vznikají v okamžiku odrazu chodidla, kdy reakční síla země skokově klesne: simulace běží po 1 ms, výsledek ID je po 5 ms.
