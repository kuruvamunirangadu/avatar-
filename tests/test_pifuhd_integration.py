import time
from pathlib import Path
import sys

here = Path(__file__).resolve().parent.parent / 'src' / 'garments'
sys.path.insert(0, str(here.parent))

import importlib.util
spec = importlib.util.spec_from_file_location('pifuhd_integration', str(here / 'pifuhd_integration.py'))
pifumod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pifumod)


def test_submit_and_check_job(monkeypatch, tmp_path):
    # Monkeypatch reconstruct_image_to_mesh to simulate work
    def fake_reconstruct(image, device='cpu', pifuhd_root=None):
        out = tmp_path / 'out.obj'
        out.write_text('o fake\nv 0 0 0\n')
        return str(out)

    monkeypatch.setattr(pifumod, 'reconstruct_image_to_mesh', fake_reconstruct)

    job_id = pifumod.submit_pifu_job(image=None)
    assert job_id is not None

    # Poll for completion
    for _ in range(20):
        status = pifumod.check_pifu_job(job_id)
        if status and status['status'] == 'finished':
            break
        time.sleep(0.1)

    status = pifumod.check_pifu_job(job_id)
    assert status is not None
    assert status['status'] in ('finished', 'error')
