"""Krok 2: reakční síly země pro model s exoskeletem (OpenSim API).

1. InverseDynamicsTool bez vnějších sil pro model bez exo a s exo: rozdíl zbytkových sil
   na pánvi = síla, kterou navíc musí dodat země kvůli hmotě exoskeletu.
2. Přičtení k GRF z test1 (TimeSeriesTable), rozdělení mezi chodidla podle svislé síly.
3. Posun působiště tak, aby zbytkový moment na pánvi zůstal jako bez exo (druhý průchod ID).
4. ExternalLoads sestavené přes API.
"""
from t3common import *  # noqa: F401,F403

PELVIS = ["pelvis_tx_force", "pelvis_ty_force", "pelvis_tilt_moment"]


def main():
    print("ID bez vnějších sil ...")
    t, db = run_id("ID_noGRF_bez_exo", BASE_MODEL)
    _, de = run_id("ID_noGRF_exo", MODELS["pasivni"])
    dQ = {c: de[c] - db[c] for c in PELVIS}

    tg, G = table_np(GRF_BASE)
    assert np.allclose(tg, t)
    fy_r, fy_l = np.clip(G["ground_force_vy"], 0, None), np.clip(G["1_ground_force_vy"], 0, None)
    share = {"": fy_r / np.maximum(fy_r + fy_l, 1e-6)}
    share["1_"] = 1 - share[""]
    for p in ("", "1_"):
        G[p + "ground_force_vx"] = G[p + "ground_force_vx"] + share[p] * dQ["pelvis_tx_force"]
        G[p + "ground_force_vy"] = G[p + "ground_force_vy"] + share[p] * dQ["pelvis_ty_force"]
    write_sto(np_table(t, G), GRF_EXO)
    external_loads(LOADS_BASE, GRF_BASE)
    external_loads(LOADS_EXO, GRF_EXO)

    # vyrovnání momentu na pánvi posunem působiště
    _, d0 = run_id("ID_bez_exo", BASE_MODEL, LOADS_BASE)
    _, dp = run_id("ID_exo_pasivni", MODELS["pasivni"], LOADS_EXO)
    fy = sum(np.clip(G[p + "ground_force_vy"], 0, None) for p in ("", "1_"))
    dx = np.where(fy > 50, (dp["pelvis_tilt_moment"] - d0["pelvis_tilt_moment"]) / np.maximum(fy, 50), 0)
    for p in ("", "1_"):
        G[p + "ground_force_px"] = G[p + "ground_force_px"] + dx
    write_sto(np_table(t, G), GRF_EXO)
    _, dp = run_id("ID_exo_pasivni", MODELS["pasivni"], LOADS_EXO)
    print(f"  posun působiště max {np.abs(dx).max() * 1000:.1f} mm")
    for name, d in (("bez exo", d0), ("pasivní exo", dp)):
        print(f"  zbytkové síly pánve ({name}): " +
              ", ".join(f"{c} RMS {np.sqrt(np.mean(d[c] ** 2)):.1f}" for c in PELVIS))
    t1, d1 = table_np(os.path.join(TEST1, "results", "ID_walk_stairs.sto"))
    print(f"  kontrola bez exo proti test1: max rozdíl momentu kolene "
          f"{np.abs(d0[KNEE + '_moment'] - d1[KNEE + '_moment']).max():.1e} Nm")


if __name__ == "__main__":
    main()
