import json
import tempfile
import sys
from pathlib import Path


def test_stub_adapter_runs(tmp_path, monkeypatch):
    # Ensure the local src is on sys.path so tests can import package modules
    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / 'src'
    sys.path.insert(0, str(SRC))
    from draping.drape import drape_garment

    avatar = tmp_path / 'avatar.obj'
    cloth = tmp_path / 'cloth.obj'
    avatar.write_text('# avatar\nv 0 0 0\n')
    cloth.write_text('# cloth\nv 0 0 0\n')

    meta = drape_garment(str(avatar), str(cloth), engine='stub', out_dir=str(tmp_path))
    assert meta['status'] == 'finished'
    assert 'result' in meta
    assert Path(meta['result']).exists()
