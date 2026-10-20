# test3 – exoskelet kolene přes OpenSim API: souhrn

Moment člověka = výstup InverseDynamicsTool s motorem v modelu; moment a výkon motoru z AnalyzeTool (ForceReporter, Kinematics). Průměr přes 6 cyklů, + = extenzní / generace.

| Úsek | Varianta | Ext. špička [Nm] | Flex. špička [Nm] | RMS [Nm] | Ext. impuls [Nm·s] | Práce člověka + [J] | Motor max [Nm] | Výkon motoru max [W] | Práce motoru [J] |
|---|---|---|---|---|---|---|---|---|---|
| Rovina | bez exo | 32.2 | -67.1 | 28.9 | 5.4 | 16.4 | 0.0 | 0.0 | 0.0 |
| Rovina | pasivní exo | 31.9 | -67.2 | 28.8 | 5.3 | 16.2 | 0.0 | 0.0 | 0.0 |
| Rovina | úhel + směr | 31.4 | -67.2 | 30.7 | 4.8 | 15.5 | 59.0 | 234.6 | 18.3 |
| Rovina | CNN ON/OFF | 31.9 | -67.2 | 28.8 | 5.3 | 16.2 | 0.0 | 0.0 | 0.0 |
| Schody | bez exo | 161.8 | -56.8 | 59.6 | 37.9 | 71.6 | 0.0 | 0.0 | 0.0 |
| Schody | pasivní exo | 162.0 | -56.4 | 59.6 | 37.9 | 71.9 | 0.0 | 0.0 | 0.0 |
| Schody | úhel + směr | 112.5 | -56.4 | 42.2 | 25.4 | 54.9 | 59.7 | 79.5 | 18.3 |
| Schody | CNN ON/OFF | 122.1 | -56.4 | 44.2 | 26.1 | 53.1 | 40.0 | 100.8 | 19.2 |

## Shoda s test2 (moment kolene člověka, celých 15 m)

| Varianta | max rozdíl [Nm] |
|---|---|
| bez exo | 0.0000 |
| pasivní exo | 0.0000 |
| úhel + směr | 0.0000 |
| CNN ON/OFF | 0.0001 |
