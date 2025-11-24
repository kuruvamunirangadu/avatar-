import numpy as np
from pathlib import Path
import sys

# import the templates module by path
here = Path(__file__).resolve().parent.parent / 'src' / 'garments'
sys.path.insert(0, str(here.parent))

from garments import templates


def test_templates_list():
    assert 'tee' in templates.LIST_TEMPLATES


def test_generate_tee_mesh_shape():
    params = {'chest': 100, 'waist': 80, 'hip': 100, 'length': 70}
    mesh = templates.generate_template_mesh('tee', params)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 2 and mesh.shape[1] == 3
