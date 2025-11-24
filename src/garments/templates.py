"""
Parametric garment templates.

This module provides small, deterministic template meshes for common garment types
and simple parameter-to-vertex mappings suitable for UI previews and fitting.

Templates included: tee, shirt, jeans, dress

Each template generator accepts a parameter dict with keys:
  chest, waist, hip, length, rise, sleeve

The output is a (N,3) numpy array of vertices forming a simple surface mesh.
"""
from typing import Dict, Tuple
import numpy as np

LIST_TEMPLATES = ["tee", "shirt", "jeans", "dress"]


def _param_grid(width: float, length: float, cols: int = 40, rows: int = 12):
    xs = np.linspace(-width / 2, width / 2, cols)
    ys = np.linspace(0.0, length, rows)
    xv, yv = np.meshgrid(xs, ys)
    verts2d = np.stack([xv.ravel(), yv.ravel()], axis=-1)
    z = np.zeros((verts2d.shape[0], 1), dtype=float)
    verts = np.hstack([verts2d, z])
    # faces (triangulate grid) and simple UVs
    faces = []
    uvs = []
    for r in range(rows):
        for c in range(cols):
            u = c / max(1, cols - 1)
            v = r / max(1, rows - 1)
            uvs.append([u, 1.0 - v])
    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            faces.append([i, i + 1, i + cols])
            faces.append([i + 1, i + cols + 1, i + cols])
    return verts, np.array(faces, dtype=int), np.array(uvs, dtype=float)


def generate_template_mesh(template: str, params: Dict[str, float]):
    """Return a simple vertex array for the requested template.

    This legacy function returns only vertices for compatibility. New code
    should call `generate_template_mesh_full` to obtain faces and uvs.
    """
    chest = float(params.get("chest", 92.0))
    waist = float(params.get("waist", 78.0))
    hip = float(params.get("hip", 98.0))
    length = float(params.get("length", 65.0))
    rise = float(params.get("rise", 25.0))
    sleeve = float(params.get("sleeve", 20.0))

    if template == "tee" or template == "shirt":
        # taper from chest at top to waist and maybe hip
        top_w = chest / 2.0
        mid_w = waist / 2.0
        bottom_w = max(hip / 2.0, mid_w)
        cols = 48
        rows = 14
        xs = np.linspace(-1.0, 1.0, cols)
        ys = np.linspace(0.0, length, rows)
        verts = []
        for yi, y in enumerate(ys):
            if yi < rows * 0.35:
                w = top_w
            elif yi < rows * 0.7:
                w = (top_w * (rows * 0.7 - yi) + mid_w * (yi - rows * 0.35)) / (rows * 0.35)
            else:
                w = (mid_w * (rows - yi) + bottom_w * (yi - rows * 0.7)) / (rows * 0.3)
            for x in xs:
                verts.append([x * w, y, 0.0])
    return np.array(verts, dtype=float)

    if template == "dress":
        # wider towards bottom (skirt flare)
        top_w = chest / 2.0
        bottom_w = hip / 2.0 + 0.2 * top_w
        cols = 60
        rows = 20
        xs = np.linspace(-1.0, 1.0, cols)
        ys = np.linspace(0.0, length, rows)
        verts = []
        for yi, y in enumerate(ys):
            t = yi / max(1, rows - 1)
            w = top_w * (1 - t) + bottom_w * t
            for x in xs:
                verts.append([x * w, y, 0.0])
    return np.array(verts, dtype=float)

    if template == "jeans":
        # two pant legs approximate using mirrored panels
        waist_w = waist / 2.0
        hip_w = hip / 2.0
        cols = 36
        rows = 22
        xs = np.linspace(0.0, 1.0, cols)
        ys = np.linspace(0.0, length, rows)
        verts = []
        for yi, y in enumerate(ys):
            # leg width reduces from hip to ankle
            w = hip_w * (1 - 0.6 * (yi / (rows - 1)))
            for x in xs:
                # left leg
                verts.append([-0.2 * waist_w + x * w, y, 0.0])
                # right leg
                verts.append([0.2 * waist_w + x * w, y, 0.0])
    return np.array(verts, dtype=float)

    # unknown template: fallback rectangular panel (legacy returns verts only)
    verts, faces, uvs = _param_grid(chest / 2.0, length)
    return verts


def generate_template_mesh_full(template: str, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices, faces, uvs) for the requested template.

    Faces are indices into the vertex array (triangles). UVs are per-vertex
    (u,v) coordinates in [0,1].
    """
    chest = float(params.get("chest", 92.0))
    waist = float(params.get("waist", 78.0))
    hip = float(params.get("hip", 98.0))
    length = float(params.get("length", 65.0))
    rise = float(params.get("rise", 25.0))
    sleeve = float(params.get("sleeve", 20.0))

    # For now use the grid generator and adjust widths per row similar to legacy
    cols = 48
    rows = 14
    xs = np.linspace(-1.0, 1.0, cols)
    ys = np.linspace(0.0, length, rows)
    verts = []
    uvs = []
    for yi, y in enumerate(ys):
        t = yi / max(1, rows - 1)
        # linear blend top->mid->bottom widths
        top_w = chest / 2.0
        mid_w = waist / 2.0
        bottom_w = max(hip / 2.0, mid_w)
        if yi < rows * 0.35:
            w = top_w
        elif yi < rows * 0.7:
            w = (top_w * (rows * 0.7 - yi) + mid_w * (yi - rows * 0.35)) / (rows * 0.35)
        else:
            w = (mid_w * (rows - yi) + bottom_w * (yi - rows * 0.7)) / (rows * 0.3)
        for xi, x in enumerate(xs):
            verts.append([x * w, y, 0.0])
            u = xi / max(1, cols - 1)
            uvs.append([u, 1.0 - t])

    verts = np.array(verts, dtype=float)
    uvs = np.array(uvs, dtype=float)
    # build faces
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i = r * cols + c
            faces.append([i, i + 1, i + cols])
            faces.append([i + 1, i + cols + 1, i + cols])
    faces = np.array(faces, dtype=int)
    return verts, faces, uvs
