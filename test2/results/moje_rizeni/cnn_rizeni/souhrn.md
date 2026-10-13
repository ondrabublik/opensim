# Vlastní řízení `cnn_rizeni`

Motor: max 60 Nm, časová konstanta 20 ms, ON = 40 Nm. Průměr přes 6 cyklů; moment + = extenzní, práce + = generace.

| Úsek | Varianta | Extenzní špička člověka [Nm] | Flexní špička [Nm] | RMS [Nm] | Práce člověka + [J/cyk] | Moment motoru max [Nm] | Výkon max [W] | Práce motoru + / − [J/cyk] |
|---|---|---|---|---|---|---|---|---|
| Rovina | bez exo | 32.2 | -67.1 | 28.9 | 16.4 | 0.0 | 0 | 0.0 / 0.0 |
| Rovina | pasivní exo | 31.9 | -67.2 | 28.8 | 16.2 | 0.0 | 0 | 0.0 / 0.0 |
| Rovina | vestavěné úhel + směr | 31.4 | -67.2 | 30.7 | 15.5 | 59.0 | 235 | 18.3 / -0.0 |
| Rovina | moje: cnn_rizeni | 31.9 | -67.2 | 28.8 | 16.2 | 0.0 | 0 | 0.0 / 0.0 |
| Schody | bez exo | 161.8 | -56.8 | 59.6 | 71.6 | 0.0 | 0 | 0.0 / 0.0 |
| Schody | pasivní exo | 162.0 | -56.4 | 59.6 | 71.9 | 0.0 | 0 | 0.0 / 0.0 |
| Schody | vestavěné úhel + směr | 112.5 | -56.4 | 42.2 | 54.9 | 59.7 | 80 | 18.3 / -0.0 |
| Schody | moje: cnn_rizeni | 122.1 | -56.4 | 44.2 | 53.1 | 40.0 | 101 | 19.2 / -0.0 |

Motor zapnut: rovina 0 % času, schody 23 % času.
