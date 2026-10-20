"""Krok 7: animace, část B (Python 3.11, matplotlib + OpenCV + imageio-ffmpeg).

Kamera sleduje pánev modelu (dopředu, nahoru i dolů po schodech, pohled z pravé strany mírně
zezadu). Vpravo běží grafy: flexe kolene, úhlová rychlost stehna a holeně, moment kolene člověka
s vypnutým / zapnutým exoskeletem a moment motoru (CNN).

    py -3.11 s7_anim_render.py [--test]      (--test = jen několik snímků do PNG; osoba z OSOBA / osoba.json)
Výstup: results/<jmeno>/animace/vstavani_schody_exo.mp4 (H.264), snimek_*.png
"""
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "test2", "scripts"))
import cadlib as C  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSOBA = json.load(open(os.path.abspath(os.environ.get("OSOBA", os.path.join(ROOT, "osoba.json"))), encoding="utf-8"))
OUT = os.path.join(ROOT, "results", OSOBA["jmeno"], "animace")
W_PX, H_PX, DPI = 1280, 720, 100
HALF = 1.1
AZIM, ELEV = -118, 16
LIGHT = np.array([0.35, -0.55, 0.75]) / np.linalg.norm([0.35, -0.55, 0.75])
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
C_ON, C_OFF, C_SHADE, C_TH = "#eb6834", "#52514e", "#f6d3c4", "#2a78d6"
STAIR, CHAIR = np.array([0.84, 0.81, 0.76]), np.array([0.55, 0.42, 0.30])
S = {}


def decimate(V, F, cell):
    key = np.floor(V / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vn = np.zeros((inv.max() + 1, 3))
    np.add.at(Vn, inv, V)
    Vn /= np.bincount(inv)[:, None]
    Fn = inv[F]
    ok = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
    return Vn, Fn[ok]


def block(cx, cy, hx, hy, z0=-0.6, z1=0.30, nz=8, color=STAIR):
    """Kvádr stupně: nášlapná plocha a obě čela (dělené po z), boční stěna na straně kamery."""
    x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
    out = []
    zs = np.linspace(z0, z1, nz + 1)
    for za, zb in zip(zs[:-1], zs[1:]):
        out.append(([[x0, y1, za], [x1, y1, za], [x1, y1, zb], [x0, y1, zb]], color))
        out.append(([[x0, y0, za], [x0, y1, za], [x0, y1, zb], [x0, y0, zb]], color * 0.86))
        out.append(([[x1, y0, za], [x1, y1, za], [x1, y1, zb], [x1, y0, zb]], color * 0.86))
    out.append(([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], color * 0.72))
    return out


def cuboid(c, h, color):
    """Plný kvádr (6 stěn) pro židli."""
    (cx, cy, cz), (hx, hy, hz) = c, h
    v = np.array([[cx + sx * hx, cy + sy * hy, cz + sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    shade = [0.8, 0.8, 0.7, 1.0, 0.85, 0.9]
    return [(v[list(f)], color * s) for f, s in zip(faces, shade)]


def to_plot(V):
    V = np.asarray(V)
    return np.column_stack([V[..., 0].ravel(), -V[..., 2].ravel(), V[..., 1].ravel()]).reshape(V.shape)


def init_worker():
    sc = json.load(open(os.path.join(OUT, "scena.json")))
    d = np.load(os.path.join(OUT, "snimky.npz"))
    S["d"] = {k: d[k] for k in d.files}
    S["sc"] = sc
    meshes = []
    for m in sc["meshes"]:
        V, F = (C.read_stl(m["file"]) if m["file"].endswith(".stl") else C.read_vtp_points_polys(m["file"]))
        V = V * np.array(m["scale"])
        if len(F) > 1500:
            V, F = decimate(V, F, 0.006)
        meshes.append((sc["bodies"].index(m["body"]), V, F, np.array(m["color"])))
    S["meshes"] = meshes
    scene = []
    for k, (name, (cx, cy), (hx, hy)) in enumerate(sc["terrain"]["blocks"]):
        scene += block(cx, cy, hx, hy, color=STAIR * (1.0 if k % 2 == 0 else 0.95))
    ch = sc["chair"]
    x0 = ch["x_front"] - ch["depth"]
    scene += cuboid(((x0 + ch["x_front"]) / 2, ch["seat_h"] - 0.025, 0.0), (ch["depth"] / 2, 0.025, 0.24), CHAIR)
    scene += cuboid((x0 + 0.02, ch["seat_h"] + 0.25, 0.0), (0.02, 0.25, 0.24), CHAIR)
    for dx in (0.03, ch["depth"] - 0.03):
        for dz in (-0.2, 0.2):
            scene += cuboid((x0 + dx, (ch["seat_h"] - 0.05) / 2, dz), (0.02, (ch["seat_h"] - 0.05) / 2, 0.02), CHAIR)
    S["scene"] = [(np.array(q, float), np.clip(c, 0, 1)) for q, c in scene]
    tr = sc["terrain"]
    S["x_up"], S["x_end"] = tr["XS"], tr["X_END"]


def shade(tri, color):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    return np.clip(color[None, :] * (0.45 + 0.55 * np.abs(n @ LIGHT))[:, None], 0, 1)


def render(i):
    d, sc = S["d"], S["sc"]
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(SURF)
    ax = fig.add_axes([0.0, 0.0, 0.58, 0.92], projection="3d")
    ax.set_facecolor(SURF)
    ax.set_axis_off()
    pel = d["pelvis"][i]
    cx, cy = pel[0], pel[1] - 0.2
    polys, cols = [], []
    for gx in np.arange(np.floor((cx - 1.8) * 4) / 4, cx + 1.8, 0.25):       # podlaha (šachovnice)
        if S["x_up"] <= gx < S["x_end"]:
            continue
        for gz in np.arange(-0.75, 0.50, 0.25):
            q = np.array([[gx, 0, gz], [gx + 0.25, 0, gz], [gx + 0.25, 0, gz + 0.25], [gx, 0, gz + 0.25]])
            polys.append(to_plot(q))
            c = 0.88 if (int(round(gx * 4)) + int(round(gz * 4))) % 2 == 0 else 0.81
            cols.append([c, c, c * 0.97, 1.0])
    for q, c in S["scene"]:
        if abs(q[:, 0].mean() - cx) < 2.0:
            polys.append(to_plot(q))
            cols.append([*c, 1.0])
    for bi, V, F, col in S["meshes"]:
        tri = (V @ d["R"][i, bi].T + d["P"][i, bi])[F]
        polys.extend(to_plot(tri))
        cols.extend(np.column_stack([shade(tri, col), np.ones(len(F))]))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none", linewidths=0))
    for g, colr in ((d["grf_r"][i], "#1baf7a"), (d["grf_l"][i], "#2a78d6"), (d["grf_ch"][i], "#8a5a2b")):
        if g[1] > 20:
            P = to_plot(np.array([g[3:6], g[3:6] + g[0:3] / 1000.0]))
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=colr, lw=2.5, zorder=10)
    ax.set_xlim(cx - HALF, cx + HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_zlim(cy - HALF, cy + HALF)
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.view_init(elev=ELEV, azim=AZIM)
    t = d["t"][i]
    usek = next((nm for nm, a, b in sc["phases"] if a <= t < b), sc["phases"][-1][0])
    ts = d["sig_t"]
    k = min(np.searchsorted(ts, t), len(ts) - 1)
    on = d["sig_on"][k] > 0.5
    fig.text(0.02, 0.95, f"t = {t:5.2f} s   ·   {usek}", fontsize=13, color=INK)
    fig.text(0.02, 0.91, "motor exo: " + ("ZAPNUT" if on else "vypnut"), fontsize=13,
             color=C_ON if on else INK2, weight="bold" if on else "normal")
    fig.text(0.02, 0.03, "šipky: reakční síla pravé (zelená) / levé (modrá) nohy a sedáku (hnědá), 1 m = 1000 N",
             fontsize=8.5, color=INK2)
    w0, w1 = t - 3.0, t + 1.0
    sel = (ts >= w0) & (ts <= w1)
    axs = [fig.add_axes([0.64, 0.765 - 0.235 * j, 0.34, 0.17]) for j in range(4)]
    for a in axs:
        a.set_facecolor(SURF)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(True, color=GRID, lw=0.6)
        a.set_xlim(w0, w1)
        a.tick_params(labelsize=7.5, colors=INK2)
        a.fill_between(ts[sel], 0, 1, where=d["sig_on"][sel] > 0.5, color=C_SHADE, lw=0,
                       transform=a.get_xaxis_transform(), zorder=0)
        a.axvline(t, color=INK, lw=1.0)
    axs[0].plot(ts[sel], d["sig_flexe"][sel], color=INK, lw=1.5)
    axs[0].set_ylim(-5, 125)
    axs[0].set_title("Flexe pravého kolene [°]", fontsize=9.5, color=INK, loc="left")
    axs[1].plot(ts[sel], d["sig_w_st"][sel], color=C_TH, lw=1.4, label="stehno")
    axs[1].plot(ts[sel], d["sig_w_ho"][sel], color=INK, lw=1.2, label="holeň")
    axs[1].set_ylim(-320, 460)
    axs[1].set_title("Úhlová rychlost vůči zemi [°/s]", fontsize=9.5, color=INK, loc="left")
    axs[1].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    axs[2].plot(ts[sel], d["sig_M_off"][sel], color=C_OFF, lw=1.2, label="exo vypnutý")
    axs[2].plot(ts[sel], d["sig_M_on"][sel], color=C_ON, lw=1.6, label="exo zapnutý")
    axs[2].set_ylim(-80, max(200, 1.1 * S["d"]["sig_M_off"].max()))
    axs[2].set_title("Moment kolene člověka [N m] (+ extenzní)", fontsize=9.5, color=INK, loc="left")
    axs[2].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    axs[3].plot(ts[sel], d["sig_tau"][sel], color=C_ON, lw=1.6)
    axs[3].set_ylim(-5, max(50, 1.1 * S["d"]["sig_tau"].max()))
    axs[3].set_title("Moment motoru exo [N m] (řízení sítí)", fontsize=9.5, color=INK, loc="left")
    axs[3].set_xlabel("čas [s]", fontsize=8.5, color=INK2)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def main():
    import cv2
    d = np.load(os.path.join(OUT, "snimky.npz"))
    n = len(d["t"])
    sc = json.load(open(os.path.join(OUT, "scena.json")))
    fps = sc["fps"]
    ph = {nm: (a, b) for nm, a, b in sc["phases"]}
    picks = {"vstavani": int(1.6 * fps)}
    for key, nm in (("rovina", "rovina"), ("schody_nahoru", "schody nahoru"), ("podesta", "podesta"),
                    ("schody_dolu", "schody dolů")):
        picks[key] = min(n - 1, int(0.5 * sum(ph[nm]) * fps))
    if "--test" in sys.argv:
        init_worker()
        for name, i in picks.items():
            plt.imsave(os.path.join(OUT, f"snimek_{name}.png"), render(i))
        print("testovací snímky uloženy")
        return
    raw = os.path.join(OUT, "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_PX, H_PX))
    with Pool(max(1, os.cpu_count() - 2), initializer=init_worker) as pool:
        for i, img in enumerate(pool.imap(render, range(n), chunksize=4)):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            for name, j in picks.items():
                if i == j:
                    plt.imsave(os.path.join(OUT, f"snimek_{name}.png"), img)
            if i % 100 == 0:
                print(f"  {i}/{n}", flush=True)
    vw.release()
    import imageio_ffmpeg
    out = os.path.join(OUT, "vstavani_schody_exo.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264",
                    "-preset", "slow", "-crf", "23", "-tune", "animation", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", out], check=True)
    os.remove(raw)
    print("Video:", out, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
