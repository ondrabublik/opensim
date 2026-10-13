"""Jednoduché parametrické CAD: uzavřené trojúhelníkové sítě z tažených průřezů,
hmotnostní vlastnosti (objem, těžiště, tenzor setrvačnosti) a zápis STL."""
import struct

import numpy as np


class Part:
    """Uzavřené těleso: vrcholy V (n,3) [m], trojúhelníky F (m,3), materiál, barva."""

    def __init__(self, name, V, F, material, density=None, mass=None, color=None):
        self.name, self.V, self.F = name, np.asarray(V, float), np.asarray(F, int)
        self.material, self.color = material, color
        vol = self.props(1.0)[0]
        if vol < 0:                       # sjednotit orientaci (normály ven)
            self.F = self.F[:, ::-1]
            vol = -vol
        self.volume = vol
        self.density = density if mass is None else mass / vol
        self.mass, self.com, self.inertia = self.props(self.density)

    def props(self, rho):
        """Hmotnost, těžiště a tenzor setrvačnosti k těžišti (Gaussova věta, tetraedry)."""
        a, b, c = (self.V[self.F[:, k]] for k in range(3))
        v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        vol = v.sum()
        com = (v[:, None] * (a + b + c)).sum(0) / 4.0 / vol
        s = a + b + c
        C = (v[:, None, None] / 20.0 * (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b)
                                        + np.einsum("ni,nj->nij", c, c)
                                        + np.einsum("ni,nj->nij", s, s))).sum(0) * rho
        m = vol * rho
        I0 = np.trace(C) * np.eye(3) - C
        Ic = I0 - m * (com @ com * np.eye(3) - np.outer(com, com))
        return m, com, Ic

    def transformed(self, R, p):
        return (self.V @ np.asarray(R).T) + p


def loft(rings):
    """Síť z posloupnosti uzavřených průřezů (každý se stejným počtem bodů), konce zavíčkované.
    Víčka se triangulují vějířem ze středu => průřezy musí být konvexní."""
    rings = [np.asarray(r, float) for r in rings]
    k, n = len(rings), len(rings[0])
    V = np.vstack(rings + [rings[0].mean(0), rings[-1].mean(0)])
    F = []
    for j in range(k - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            c, d = (j + 1) * n + (i + 1) % n, (j + 1) * n + i
            F += [(a, b, c), (a, c, d)]
    c0, c1 = k * n, k * n + 1
    for i in range(n):
        F.append((c0, (i + 1) % n, i))
        F.append((c1, (k - 1) * n + i, (k - 1) * n + (i + 1) % n))
    return V, np.array(F)


def cylinder(center, r, z0, z1, n=64):
    """Válec s osou rovnoběžnou se z."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([center[0] + r * np.cos(a), center[1] + r * np.sin(a)])
    return loft([np.column_stack([ring, np.full(n, z)]) for z in (z0, z1)])


def bar(p0, p1, w, t, wdir=(1, 0, 0)):
    """Hranol obdélníkového průřezu w x t mezi body p0 a p1; šířka w ve směru ~wdir."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    e1 = np.asarray(wdir, float) - u * (u @ wdir)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.5
    ring = sq[:, :1] * w * e1 + sq[:, 1:] * t * e2
    return loft([p0 + ring, p1 + ring])


def arc_shell(center_xz, r_in, r_out, y0, y1, phi0, phi1, n=72):
    """Část válcové skořepiny s osou ve směru y (objímka).
    Úhel phi [deg] se měří od +z (laterálně) k +x (vpřed)."""
    rings = []
    for phi in np.radians(np.linspace(phi0, phi1, n)):
        d = np.array([np.sin(phi), 0.0, np.cos(phi)])
        o = np.array([center_xz[0], 0.0, center_xz[1]])
        rings.append([o + r_in * d + [0, y0, 0], o + r_out * d + [0, y0, 0],
                      o + r_out * d + [0, y1, 0], o + r_in * d + [0, y1, 0]])
    return loft(rings)


def write_stl(path, V, F, scale=1.0, name="part"):
    """Binární STL."""
    V = np.asarray(V) * scale
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        for nn, t in zip(nrm, tri):
            f.write(struct.pack("<12fH", *nn, *t[0], *t[1], *t[2], 0))


def read_vtp_points_polys(path):
    """Načte body a polygony z ASCII .vtp (kosti OpenSim) pro vykreslení."""
    import re
    s = open(path).read()

    def arr(tag, name):
        m = re.search(rf"<{tag}>.*?<DataArray[^>]*Name=\"{name}\"[^>]*>(.*?)</DataArray>", s, re.S)
        if m is None:
            m = re.search(rf"<{tag}>.*?<DataArray[^>]*>(.*?)</DataArray>", s, re.S)
        return np.array(m.group(1).split(), float)
    P = arr("Points", "Points").reshape(-1, 3)
    conn = arr("Polys", "connectivity").astype(int)
    off = arr("Polys", "offsets").astype(int)
    polys, start = [], 0
    for o in off:
        polys.append(conn[start:o])
        start = o
    tris = [(p[0], p[i], p[i + 1]) for p in polys for i in range(1, len(p) - 1)]
    return P, np.array(tris)


def read_stl(path):
    """Binární STL -> (V, F) (vrcholy se neslučují)."""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        d = np.frombuffer(f.read(50 * n), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    V = d["v"].reshape(-1, 3).astype(float)
    return V, np.arange(len(V)).reshape(-1, 3)
