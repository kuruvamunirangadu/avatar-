"""
Optional PIFuHD integration stub.

This module provides a thin wrapper around an optional PIFuHD-style
single-image reconstruction codepath. It's intentionally a graceful
stub: if the heavy dependency isn't installed it raises an informative
ImportError and the app can fall back to template-based meshes.
"""
from typing import Any
import os
import subprocess
import tempfile
from pathlib import Path
import threading
import uuid
import json
import shutil
import time

from pathlib import Path

# Persistent job storage directory
ROOT = Path(__file__).resolve().parents[2]
JOB_DIR = ROOT / 'data' / 'pifuhd_jobs'
JOB_DIR.mkdir(parents=True, exist_ok=True)


def reconstruct_image_to_mesh(image: Any, device: str = 'cpu', pifuhd_root: str = None):
    """Reconstruct a 3D mesh from a single image using PIFuHD-like toolchains.

    Strategy (in order):
    1) If a Python package named `pifuhd` is importable and exposes `reconstruct`, call it.
    2) If env var `PIFUHD_ROOT` or the `pifuhd_root` argument is set to a local PIFuHD repo,
       attempt to run a CLI inference script (e.g., `run_monocular.py` or `inference.py`) via subprocess.

    Returns: path to a generated OBJ (or mesh-like object) on success.

    Raises ImportError or RuntimeError with guidance when not available.
    """
    # Option A: Python package
    try:
        import pifuhd as _pifuhd  # type: ignore
        if hasattr(_pifuhd, 'reconstruct'):
            return _pifuhd.reconstruct(image, device=device)
    except Exception:
        pass

    # Option B: CLI invocation of a local PIFuHD repo
    root = pifuhd_root or os.environ.get('PIFUHD_ROOT')
    if not root:
        raise ImportError('PIFuHD integration not available: no `pifuhd` package and PIFUHD_ROOT not set.')

    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f'PIFUHD_ROOT set to {root} but path does not exist.')

    # Find candidate inference scripts
    candidates = ['run_monocular.py', 'inference.py', 'test.py']
    script = None
    for c in candidates:
        p = root_path / c
        if p.exists():
            script = p
            break
    if script is None:
        # fallback: try `scripts/inference.py`
        for p in root_path.rglob('*.py'):
            if p.name in candidates:
                script = p
                break

    if script is None:
        raise RuntimeError('Could not find a PIFuHD inference script under PIFUHD_ROOT. Please set PIFUHD_ROOT to your PIFuHD repo path.')

    # Write the input image to a temporary file and call the script
    with tempfile.TemporaryDirectory() as td:
        input_path = Path(td) / 'input.png'
        output_dir = Path(td) / 'out'
        output_dir.mkdir(parents=True, exist_ok=True)
        # `image` may be a PIL Image or numpy array; attempt to save
        try:
            from PIL import Image
            if hasattr(image, 'save'):
                image.save(str(input_path))
            else:
                Image.fromarray(image).save(str(input_path))
        except Exception:
            raise RuntimeError('Failed to write input image for PIFuHD.')

        # Build a CLI command. Many PIFuHD forks accept: python run_monocular.py --img input --out output_dir
        cmd = ['python', str(script), '--img', str(input_path), '--out', str(output_dir)]
        try:
            subprocess.check_call(cmd, cwd=str(root_path))
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'PIFuHD subprocess failed: {e}')

        # Search for an OBJ in output_dir
        objs = list(output_dir.rglob('*.obj'))
        if objs:
            return str(objs[0])
        else:
            raise RuntimeError('PIFuHD completed but no OBJ was found in the output directory.')


_JOB_STORE = {}


def submit_pifu_job(image: Any, device: str = 'cpu', pifuhd_root: str = None):
    """Submit an asynchronous PIFuHD job. Returns a job_id string.

    Call `check_pifu_job(job_id)` to poll for status/result. Result will be
    a dict with keys: status ('running'|'finished'|'error') and result (obj path) or error message.
    """
    jid = str(uuid.uuid4())
    _JOB_STORE[jid] = {'status': 'queued', 'result': None}
    # persistent job dir
    job_path = JOB_DIR / jid
    job_path.mkdir(parents=True, exist_ok=True)
    meta = {'job_id': jid, 'status': 'queued', 'result': None, 'created_at': time.time()}
    with open(job_path / 'job.json', 'w', encoding='utf-8') as jf:
        json.dump(meta, jf)

    def _run():
        _JOB_STORE[jid]['status'] = 'running'
        # update persistent metadata
        meta['status'] = 'running'
        with open(job_path / 'job.json', 'w', encoding='utf-8') as jf:
            json.dump(meta, jf)
        try:
            res = reconstruct_image_to_mesh(image, device=device, pifuhd_root=pifuhd_root)
            _JOB_STORE[jid]['status'] = 'finished'
            _JOB_STORE[jid]['result'] = str(res)
            meta['status'] = 'finished'
            meta['result'] = str(res)
            meta['finished_at'] = time.time()
            with open(job_path / 'job.json', 'w', encoding='utf-8') as jf:
                json.dump(meta, jf)
        except Exception as e:
            _JOB_STORE[jid]['status'] = 'error'
            _JOB_STORE[jid]['result'] = str(e)
            meta['status'] = 'error'
            meta['result'] = str(e)
            meta['finished_at'] = time.time()
            with open(job_path / 'job.json', 'w', encoding='utf-8') as jf:
                json.dump(meta, jf)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jid


def check_pifu_job(job_id: str):
    # try in-memory first
    if job_id in _JOB_STORE:
        return _JOB_STORE.get(job_id)
    # fallback: try persistent job file
    job_path = JOB_DIR / job_id / 'job.json'
    if job_path.exists():
        try:
            with open(job_path, 'r', encoding='utf-8') as jf:
                return json.load(jf)
        except Exception:
            return None
    return None
