import uuid
import numpy as np
from pathlib import Path
import importlib.util


# Import module by path to avoid relying on PYTHONPATH during tests
def _load_generate_mesh_module():
    repo_root = Path(__file__).resolve().parents[1]
    mod_path = repo_root / 'src' / 'avatar' / 'generate_mesh.py'
    spec = importlib.util.spec_from_file_location('generate_mesh', str(mod_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_mesh = _load_generate_mesh_module()


def test_pose_preset_shapes():
    presets = ['T-pose', 'A-pose', 'hands-down', 'walking']
    for p in presets:
        body_pose, global_orient, lhand, rhand = generate_mesh._pose_preset_arrays(p)
        assert isinstance(body_pose, np.ndarray)
        assert body_pose.shape == (1, 21 * 3)
        assert global_orient.shape == (1, 3)
        assert lhand.shape == (1, 15 * 3)
        assert rhand.shape == (1, 15 * 3)


def test_cache_save_load_roundtrip(tmp_path):
    key = f"test_{uuid.uuid4().hex}"
    betas = np.zeros((1, 10), dtype=float)
    body_pose = np.zeros((1, 21 * 3), dtype=float)
    global_orient = np.zeros((1, 3), dtype=float)
    # Save
    generate_mesh._save_cache(key, betas, body_pose, global_orient)
    # Load
    out = generate_mesh._load_cache(key)
    assert out is not None
    b2, bp2, go2 = out
    assert np.allclose(b2, betas)
    assert np.allclose(bp2, body_pose)
    assert np.allclose(go2, global_orient)
