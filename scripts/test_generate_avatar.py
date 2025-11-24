"""Lightweight test runner for generate_avatar_mesh.

This script avoids importing the full `src` package (and therefore heavy
dependencies like mediapipe/tensorflow) by loading `generate_mesh.py` via its
file path. It constructs a minimal dummy object with a `gender` attribute and
prints results or errors.
"""
import os
import importlib.util
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(ROOT, 'src', 'avatar', 'generate_mesh.py')

spec = importlib.util.spec_from_file_location('generate_mesh', SRC_PATH)
gen_mod = importlib.util.module_from_spec(spec)
sys.modules['generate_mesh'] = gen_mod
spec.loader.exec_module(gen_mod)

def run():
    class D:
        pass

    d = D()
    d.gender = 'neutral'

    try:
        mesh, meas = gen_mod.generate_avatar_mesh(d)
        print('MEASUREMENTS:', meas)
        if hasattr(mesh, 'vertices'):
            print('VERTEX_COUNT:', len(mesh.vertices))
        else:
            print('NO_VERTS')
    except Exception as e:
        print('ERROR:', e)

if __name__ == '__main__':
    run()
