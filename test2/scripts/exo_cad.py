"""Krok 1: parametrický CAD model exoskeletu pravého kolene.

Výstupy:
  model/Geometry/exo_*.stl      díly pro OpenSim (metry, souřadný systém kolene K)
  cad/exo_*_mm.stl              tytéž díly v mm pro CAD / 3D tisk
  cad/exo_koleno_sestava_mm.stl celá sestava v natažené poloze
  cad/exo_hmotnosti.json        hmotnost, těžiště a setrvačnost obou článků (pro OpenSim)
  cad/kusovnik.md               seznam dílů
  cad/exo_nahled.png            náhled
"""
import json
import os

import numpy as np

import cadlib as C
import exo_params as E
from render import draw_meshes, new_axes, save_fig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAD = os.path.join(ROOT, "cad")
GEOM = os.path.join(ROOT, "model", "Geometry")
LINKS = {"stehno": "exo_thigh_r", "berce": "exo_shank_r"}


def build_parts():
    D, col = E.DENSITY, E.COLORS
    parts = {"stehno": [], "berce": []}

    def add(link, name, VF, mat, mass=None, color=None):
        p = C.Part(name, *VF, mat, density=D.get(mat), mass=mass, color=color or col[mat])
        parts[link].append(p)

    t2 = E.ARM_T / 2
    # ---------------- článek stehna
    xc, zc = E.THIGH_CENTER
    y_top = E.THIGH_CUFF_Y[1] - 0.010
    r_arm = E.THIGH_R + E.PAD_T + E.SHELL_T + t2          # střed ramene u objímky
    add("stehno", "pohon", C.cylinder((0, 0), E.MOTOR_R, *E.MOTOR_Z), "pohon",
        mass=E.MOTOR_MASS, color=col["pohon"])
    add("stehno", "elektronika", C.cylinder((0, 0), E.CAP_R, *E.CAP_Z), "elektronika",
        mass=E.CAP_MASS, color=col["elektronika"])
    add("stehno", "rameno_stehna_lat",
        C.bar((0, E.MOTOR_R - 0.010, E.Z_THIGH_ARM), (xc, y_top, zc + r_arm), E.THIGH_ARM_W, E.ARM_T),
        "hlinik_7075")
    add("stehno", "rameno_stehna_med",
        C.bar((0, 0, -E.Z_THIGH_ARM), (xc, y_top, zc - r_arm), E.THIGH_ARM_W, E.ARM_T),
        "hlinik_7075")
    add("stehno", "kloub_med_stehno",
        C.cylinder((0, 0), E.MED_HINGE_R, -E.Z_THIGH_ARM - t2 - 0.006, -E.Z_THIGH_ARM + t2),
        "hlinik_7075")
    w = E.CUFF_WRAP_DEG
    add("stehno", "objimka_stehna",
        C.arc_shell(E.THIGH_CENTER, E.THIGH_R + E.PAD_T, E.THIGH_R + E.PAD_T + E.SHELL_T,
                    *E.THIGH_CUFF_Y, -w, 180 + w), "kompozit_CF")
    add("stehno", "vystelka_stehna",
        C.arc_shell(E.THIGH_CENTER, E.THIGH_R, E.THIGH_R + E.PAD_T, *E.THIGH_CUFF_Y, -w, 180 + w),
        "pena_EVA")
    ym = np.mean(E.THIGH_CUFF_Y)
    add("stehno", "pasek_stehna",
        C.arc_shell(E.THIGH_CENTER, E.THIGH_R, E.THIGH_R + E.STRAP_T,
                    ym - E.STRAP_H / 2, ym + E.STRAP_H / 2, 180, 360), "textil")

    # ---------------- článek bérce
    xs, zs = E.SHANK_CENTER
    y_bot = E.SHANK_CUFF_Y[0] + 0.010
    r_arm = E.SHANK_R + E.PAD_T + E.SHELL_T + t2
    add("berce", "vystupni_priruba", C.cylinder((0, 0), E.FLANGE_R, *E.FLANGE_Z), "hlinik_7075")
    add("berce", "rameno_berce_lat",
        C.bar((0, -E.FLANGE_R + 0.010, E.Z_SHANK_ARM), (xs, y_bot, zs + r_arm), E.SHANK_ARM_W, E.ARM_T),
        "hlinik_7075")
    add("berce", "rameno_berce_med",
        C.bar((0, 0, -E.Z_SHANK_ARM), (xs, y_bot, zs - r_arm), E.SHANK_ARM_W, E.ARM_T),
        "hlinik_7075")
    add("berce", "kloub_med_berce",
        C.cylinder((0, 0), E.MED_HINGE_R, -E.Z_SHANK_ARM - t2, -E.Z_SHANK_ARM + t2), "hlinik_7075")
    add("berce", "objimka_berce",
        C.arc_shell(E.SHANK_CENTER, E.SHANK_R + E.PAD_T, E.SHANK_R + E.PAD_T + E.SHELL_T,
                    *E.SHANK_CUFF_Y, 180 - w, 360 + w), "kompozit_CF")
    add("berce", "vystelka_berce",
        C.arc_shell(E.SHANK_CENTER, E.SHANK_R, E.SHANK_R + E.PAD_T, *E.SHANK_CUFF_Y, 180 - w, 360 + w),
        "pena_EVA")
    ym = np.mean(E.SHANK_CUFF_Y)
    add("berce", "pasek_berce",
        C.arc_shell(E.SHANK_CENTER, E.SHANK_R, E.SHANK_R + E.STRAP_T,
                    ym - E.STRAP_H / 2, ym + E.STRAP_H / 2, 0, 180), "textil")
    return parts


def link_props(parts):
    m = sum(p.mass for p in parts)
    com = sum(p.mass * p.com for p in parts) / m
    I = np.zeros((3, 3))
    for p in parts:
        d = p.com - com
        I += p.inertia + p.mass * (d @ d * np.eye(3) - np.outer(d, d))
    return m, com, I


def main():
    for d in (CAD, GEOM):
        os.makedirs(d, exist_ok=True)
    parts = build_parts()
    info, rows = {}, []
    allV, allF = [], []
    for link, plist in parts.items():
        m, com, I = link_props(plist)
        info[LINKS[link]] = {
            "mass": m, "mass_center": com.tolist(),
            "inertia": [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]],
            "parts": [{"name": p.name, "file": f"exo_{p.name}.stl", "material": p.material,
                       "color": list(p.color), "mass": p.mass} for p in plist]}
        for p in plist:
            C.write_stl(os.path.join(GEOM, f"exo_{p.name}.stl"), p.V, p.F, 1.0, p.name)
            C.write_stl(os.path.join(CAD, f"exo_{p.name}_mm.stl"), p.V, p.F, 1000.0, p.name)
            ext = (p.V.max(0) - p.V.min(0)) * 1000
            rows.append(f"| {link} | {p.name} | {p.material} | {p.mass * 1000:7.1f} | "
                        f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} |")
            allF.append(p.F + sum(len(v) for v in allV))
            allV.append(p.V)
        print(f"{link:7s}: {m:.3f} kg, těžiště K [mm] {np.round(com * 1000, 1)}")
    C.write_stl(os.path.join(CAD, "exo_koleno_sestava_mm.stl"), np.vstack(allV), np.vstack(allF),
                1000.0, "exo_koleno_sestava")
    json.dump(info, open(os.path.join(CAD, "exo_hmotnosti.json"), "w"), indent=1)

    tot = sum(v["mass"] for v in info.values())
    with open(os.path.join(CAD, "kusovnik.md"), "w", encoding="utf-8") as f:
        f.write("# Kusovník – exoskelet pravého kolene\n\n"
                "Souřadný systém K: počátek na ose kloubu, x vpřed, y nahoru, z laterálně. "
                "Rozměry jsou obalové [mm] (x × y × z).\n\n"
                "| Článek | Díl | Materiál | Hmotnost [g] | Rozměry [mm] |\n|---|---|---|---|---|\n")
        f.write("\n".join(rows) + "\n\n")
        for k, v in info.items():
            f.write(f"* **{k}**: {v['mass']:.3f} kg, těžiště v K = "
                    f"({', '.join(f'{c * 1000:.1f}' for c in v['mass_center'])}) mm, "
                    f"Ixx, Iyy, Izz = {', '.join(f'{c * 1e4:.1f}' for c in v['inertia'][:3])} kg·cm²\n")
        f.write(f"* **celkem**: {tot:.3f} kg\n\n"
                "Pohon (motor + vlnový převod) a elektronika mají zadanou hmotnost, ostatní díly "
                "hmotnost z objemu a hustoty materiálu.\n")

    # náhled: tři pohledy
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 5.2))
    meshes = [(p.V, p.F, p.color, 1.0) for pl in parts.values() for p in pl]
    for k, (title, el, az) in enumerate((("Izometrie (laterálně zepředu)", 18, -35),
                                         ("Zepředu", 0, 0), ("Z boku (laterálně)", 0, -90))):
        ax = new_axes(fig, 131 + k)
        draw_meshes(ax, meshes, elev=el, azim=az)
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Exoskelet pravého kolene – parametrický CAD ({tot:.2f} kg)", fontsize=11)
    save_fig(fig, os.path.join(CAD, "exo_nahled.png"))
    print("CAD hotov:", CAD)


if __name__ == "__main__":
    main()
