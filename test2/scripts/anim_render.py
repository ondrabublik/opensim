"""Animace, krok B (Python 3.11, matplotlib + OpenCV): vykreslení snímků a MP4.

Kamera sleduje pánev modelu (střed scény se posouvá s chůzí i nahoru po schodech),
vpravo běží grafy flexe kolene, momentu kolene (bez exo / člověk s exo) a motoru (CNN).

    py -3.11 anim_render.py [--test]      (--test = jen několik snímků do PNG)
Výstup: results/animace/chuze_exo.mp4 (velké), chuze_exo_email.mp4 (H.264), snimek_*.png
"""
import json
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cadlib as C  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "animace")
W_PX, H_PX, DPI = 1280, 720, 100
HALF = 1.05                 # polovina velikosti krychle zobrazené kolem pánve [m]
AZIM, ELEV = -118, 16
LIGHT = np.array([0.35, -0.55, 0.75]) / np.linalg.norm([0.35, -0.55, 0.75])
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
C_HUMAN, C_BASE, C_ON = "#eb6834", "#52514e", "#f6d3c4"

S = {}   # scéna pro worker procesy


def decimate(V, F, cell):
    """Zjednodušení sítě shlukováním vrcholů do mřížky (jen pro rychlejší kreslení)."""
    key = np.floor(V / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    Vn = np.zeros((inv.max() + 1, 3))
    np.add.at(Vn, inv, V)
    Vn /= np.bincount(inv)[:, None]
    Fn = inv[F]
    ok = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
    return Vn, Fn[ok]


def box(center, half, nz=8, z_near=0.30):
    """Stupeň jako seznam čtyřúhelníků: nášlapná plocha a čelo (dělené po z kvůli řazení
    ploch) a boční stěna. Na straně ke kameře je schodiště oříznuté na z_near, aby
    nezakrývalo nohy."""
    c, h = np.asarray(center), np.asarray(half)
    x0, x1, y0, y1, z0 = c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], c[2] - h[2]
    z1 = min(c[2] + h[2], z_near)
    quads = []
    zs = np.linspace(z0, z1, nz + 1)
    for za, zb in zip(zs[:-1], zs[1:]):
        quads.append(([[x0, y1, za], [x1, y1, za], [x1, y1, zb], [x0, y1, zb]], 1.00))   # nášlap
        quads.append(([[x0, y0, za], [x0, y1, za], [x0, y1, zb], [x0, y0, zb]], 0.86))   # čelo
    quads.append(([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], 0.72))       # bok
    return quads


def to_plot(V):
    return np.column_stack([V[..., 0].ravel(), -V[..., 2].ravel(), V[..., 1].ravel()]).reshape(V.shape)


def init_worker():
    scene = json.load(open(os.path.join(OUT, "scena.json")))
    d = np.load(os.path.join(OUT, "snimky.npz"))
    S["d"] = {k: d[k] for k in d.files}
    S["bodies"] = scene["bodies"]
    S["scene"] = scene
    meshes = []
    for m in scene["meshes"]:
        if m["file"].endswith(".stl"):
            V, F = C.read_stl(m["file"])
        else:
            V, F = C.read_vtp_points_polys(m["file"])
        V = V * np.array(m["scale"])
        if len(F) > 1500:
            V, F = decimate(V, F, 0.006)
        meshes.append((scene["bodies"].index(m["body"]), V, F, np.array(m["color"])))
    S["meshes"] = meshes
    stairs = []
    for st in scene["stairs"]:
        base = 0.86 if len(S.get("_n", [])) % 2 == 0 else 0.80
        S.setdefault("_n", []).append(1)
        for q, f in box(st["center"], st["half"]):
            stairs.append((np.array(q), np.array([base, base * 0.97, base * 0.92]) * f))
    S["stairs"] = stairs
    S["x_stairs"] = min(st["center"][0] - st["half"][0] for st in scene["stairs"])


def shade(tri, color):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    s = 0.45 + 0.55 * np.abs(n @ LIGHT)
    return np.clip(color[None, :] * s[:, None], 0, 1)


def render(i):
    d = S["d"]
    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    fig.patch.set_facecolor(SURF)
    ax = fig.add_axes([0.0, 0.0, 0.60, 0.95], projection="3d")
    ax.set_facecolor(SURF)
    ax.set_axis_off()
    pel = d["pelvis"][i]
    cx, cy = pel[0], pel[1] - 0.15
    polys, cols = [], []
    # podlaha (šachovnice 0.25 m) jen v okolí kamery
    for gx in np.arange(np.floor((cx - 1.6) * 4) / 4, cx + 1.6, 0.25):
        if gx >= S["x_stairs"]:
            continue
        for gz in np.arange(-0.75, 0.50, 0.25):
            q = np.array([[gx, 0, gz], [gx + 0.25, 0, gz], [gx + 0.25, 0, gz + 0.25], [gx, 0, gz + 0.25]])
            polys.append(to_plot(q))
            c = 0.88 if (int(round(gx * 4)) + int(round(gz * 4))) % 2 == 0 else 0.81
            cols.append([c, c, c * 0.97, 1.0])
    for q, c in S["stairs"]:
        if abs(q[:, 0].mean() - cx) < 1.8:
            polys.append(to_plot(q))
            cols.append([*c, 1.0])
    for bi, V, F, col in S["meshes"]:
        Vw = V @ d["R"][i, bi].T + d["P"][i, bi]
        tri = Vw[F]
        polys.extend(to_plot(tri))
        cols.extend(np.column_stack([shade(tri, col), np.ones(len(F))]))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none", linewidths=0))
    for g, colr in ((d["grf_r"][i], "#1baf7a"), (d["grf_l"][i], "#2a78d6")):
        if g[1] > 20:
            p0 = g[3:6]
            p1 = p0 + g[0:3] / 1000.0
            P = to_plot(np.array([p0, p1]))
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=colr, lw=2.5, zorder=10)
    ax.set_xlim(cx - HALF, cx + HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_zlim(cy - HALF, cy + HALF)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.view_init(elev=ELEV, azim=AZIM)
    t = d["t"][i]
    sc = S["scene"]
    usek = "rovina" if t < sc["t_rovina_konec"] else ("přechod" if t < sc["t_schody"] else "schody")
    ts = d["sig_t"]
    k = min(np.searchsorted(ts, t), len(ts) - 1)
    on = d["sig_on"][k] > 0.5
    fig.text(0.02, 0.95, f"t = {t:5.2f} s   ·   {usek}   ·   x = {pel[0]:.1f} m", fontsize=13, color=INK)
    fig.text(0.02, 0.91, "motor exo: " + ("ZAPNUT" if on else "vypnut"), fontsize=13,
             color=C_HUMAN if on else INK2, weight="bold" if on else "normal")
    fig.text(0.02, 0.03, "zelená/modrá šipka = reakční síla země pravé/levé nohy (1 m = 1000 N)",
             fontsize=9, color=INK2)

    # grafy vpravo – posuvné okno
    w0, w1 = t - 3.0, t + 1.0
    sel = (ts >= w0) & (ts <= w1)
    axs = [fig.add_axes([0.66, 0.70 - 0.29 * j, 0.32, 0.22]) for j in range(3)]
    for a in axs:
        a.set_facecolor(SURF)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(True, color=GRID, lw=0.6)
        a.set_xlim(w0, w1)
        a.tick_params(labelsize=8, colors=INK2)
        a.fill_between(ts[sel], 0, 1, where=d["sig_on"][sel] > 0.5, color=C_ON, lw=0,
                       transform=a.get_xaxis_transform(), zorder=0)
        a.axvline(t, color=INK, lw=1.0)
    axs[0].plot(ts[sel], d["sig_flexe"][sel], color=INK, lw=1.6)
    axs[0].set_ylim(-10, 120)
    axs[0].set_title("Flexe pravého kolene [°]", fontsize=10, color=INK, loc="left")
    axs[1].plot(ts[sel], d["sig_M_bez"][sel], color=C_BASE, lw=1.3, label="bez exo")
    axs[1].plot(ts[sel], d["sig_M_clovek"][sel], color=C_HUMAN, lw=1.8, label="člověk s exo")
    axs[1].set_ylim(-80, 175)
    axs[1].set_title("Moment kolene [Nm] (+ extenzní)", fontsize=10, color=INK, loc="left")
    axs[1].legend(frameon=False, fontsize=8, loc="upper left")
    axs[2].plot(ts[sel], d["sig_tau"][sel], color=C_HUMAN, lw=1.8)
    axs[2].set_ylim(-5, 50)
    axs[2].set_title("Moment motoru exo [Nm] (řízení CNN)", fontsize=10, color=INK, loc="left")
    axs[2].set_xlabel("čas [s]", fontsize=9, color=INK2)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def main():
    import cv2
    test = "--test" in sys.argv
    d = np.load(os.path.join(OUT, "snimky.npz"))
    n = len(d["t"])
    fps = json.load(open(os.path.join(OUT, "scena.json")))["fps"]
    picks = {"rovina": int(4.4 * fps), "prechod": int(9.4 * fps), "schody_stoj": int(14.6 * fps),
             "schody_svih": int(15.2 * fps)}
    if test:
        init_worker()
        for name, i in picks.items():
            plt.imsave(os.path.join(OUT, f"snimek_{name}.png"), render(i))
        print("testovací snímky uloženy")
        return
    path = os.path.join(OUT, "chuze_exo.mp4")
    vw = None
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (W_PX, H_PX))
        if vw.isOpened():
            print("kodek", fourcc)
            break
    with Pool(max(1, os.cpu_count() - 2), initializer=init_worker) as pool:
        for i, img in enumerate(pool.imap(render, range(n), chunksize=4)):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if i in picks.values():
                name = [k for k, v in picks.items() if v == i][0]
                plt.imsave(os.path.join(OUT, f"snimek_{name}.png"), img)
            if i % 50 == 0:
                print(f"  {i}/{n}", flush=True)
    vw.release()
    print("Video:", path)
    # komprese do H.264 (malý soubor, přehraje ho cokoli); vyžaduje balíček imageio-ffmpeg
    try:
        import subprocess
        import imageio_ffmpeg
        small = os.path.join(OUT, "chuze_exo_email.mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", path,
                        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-tune", "animation",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", small], check=True)
        print("H.264:", small)
    except ImportError:
        print("imageio-ffmpeg není nainstalované, komprese H.264 vynechána")


if __name__ == "__main__":
    main()
