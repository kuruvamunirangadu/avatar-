"""Small runner to demonstrate the drape pipeline.

This script creates minimal OBJ fixtures (a sphere-like avatar and a plane
as cloth), then calls the drape pipeline. It will use Blender if available
otherwise fall back to the stub.
"""
from pathlib import Path
import tempfile
import textwrap
import shutil
import sys

import sys
from pathlib import Path
# Ensure src is on sys.path so we can import local packages
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from draping.drape import drape_garment


def write_plane_obj(path: Path, size=1.0, divisions=1):
    # Very small plane OBJ
    verts = [(-size, 0.0, -size), (size, 0.0, -size), (size, 0.0, size), (-size, 0.0, size)]
    faces = [(1,2,3,4)]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# plane\n')
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write('f ' + ' '.join(str(i) for i in face) + '\n')


def write_sphere_like_obj(path: Path):
    # Minimal placeholder sphere (icosphere would be better but keep tiny)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# minimal avatar placeholder\n')
        f.write('v 0 0 0\n')
        f.write('v 0 1 0\n')
        f.write('v 1 0 0\n')
        f.write('f 1 2 3\n')


def main(out_dir: str = None, engine: str = 'blender'):
    tmp = Path(out_dir or tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    avatar = tmp / 'avatar.obj'
    cloth = tmp / 'cloth.obj'
    write_sphere_like_obj(avatar)
    write_plane_obj(cloth)

    print('Running drape pipeline with engine=', engine)
    meta = drape_garment(str(avatar), str(cloth), engine=engine, out_dir=str(tmp))
    print('Result:', meta)


if __name__ == '__main__':
    engine = sys.argv[1] if len(sys.argv) > 1 else 'blender'
    out = sys.argv[2] if len(sys.argv) > 2 else None
    main(out_dir=out, engine=engine)
