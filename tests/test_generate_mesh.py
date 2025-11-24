import os
import importlib.util
import sys
import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_ROOT = os.path.join(ROOT, 'src', 'models', 'smplx')


def _model_exists_for_gender(gender):
    # Prefer gender subfolder model.npz
    path1 = os.path.join(MODEL_ROOT, gender, 'model.npz')
    # Fallback to top-level SMPLX_<GENDER>.npz naming
    top = os.path.join(MODEL_ROOT, f'SMPLX_{gender.upper()}.npz')
    return os.path.exists(path1) or os.path.exists(top)


@pytest.mark.parametrize('gender', ['neutral', 'male', 'female'])
def test_generate_avatar_loads_mesh(gender):
    """Integration smoke-test: load a mesh for each gender if models are present.

    If models are not present, the test is skipped so CI doesn't fail when
    proprietary SMPL-X files are omitted.
    """
    if not _model_exists_for_gender(gender):
        pytest.skip(f'Model files for gender {gender} not found under {MODEL_ROOT}')

    # Load generate_mesh module directly by path to avoid importing the full app
    gen_path = os.path.join(ROOT, 'src', 'avatar', 'generate_mesh.py')
    spec = importlib.util.spec_from_file_location('generate_mesh', gen_path)
    gen_mod = importlib.util.module_from_spec(spec)
    sys.modules['generate_mesh'] = gen_mod
    spec.loader.exec_module(gen_mod)

    class Dummy:
        def __init__(self, gender):
            self.gender = gender

    dummy = Dummy(gender)

    mesh, meas = gen_mod.generate_avatar_mesh(dummy)
    assert mesh is not None
    assert hasattr(mesh, 'vertices') or hasattr(mesh, 'faces')
    assert isinstance(meas, dict)
