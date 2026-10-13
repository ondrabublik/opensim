"""Vykreslení sítí (matplotlib 3D) se stínováním. Souřadnice OpenSim (x vpřed, y nahoru,
z laterálně) se kreslí jako (x, -z, y), aby osa y modelu mířila v grafu nahoru."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

SURF = "#fcfcfb"


def to_plot(V):
    V = np.asarray(V)
    return np.column_stack([V[:, 0], -V[:, 2], V[:, 1]])


def new_axes(fig, pos):
    ax = fig.add_subplot(pos, projection="3d")
    ax.set_facecolor(SURF)
    ax.set_axis_off()
    return ax


def draw_meshes(ax, meshes, elev=15, azim=-35, light=(0.4, -0.6, 0.7), pad=0.01, zoom=1.45):
    """meshes: [(V (n,3) v OpenSim souř., F (m,3), barva rgb, alfa)]."""
    polys, cols = [], []
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for V, F, color, alpha in meshes:
        P = to_plot(V)
        tri = P[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
        shade = 0.45 + 0.55 * np.abs(n @ L)
        rgb = np.clip(np.asarray(color)[None, :] * shade[:, None], 0, 1)
        polys.extend(tri)
        cols.extend(np.column_stack([rgb, np.full(len(F), alpha)]))
        lo, hi = np.minimum(lo, P.min(0)), np.maximum(hi, P.max(0))
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 + pad
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    return ax


def save_fig(fig, path, dpi=150):
    fig.patch.set_facecolor(SURF)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
