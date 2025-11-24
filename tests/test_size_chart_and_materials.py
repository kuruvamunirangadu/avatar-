import sys
from pathlib import Path

here = Path(__file__).resolve().parent.parent / 'src' / 'garments'
sys.path.insert(0, str(here.parent))

from garments import size_chart, materials


def test_parse_dict_size_chart():
    d = {'chest': 95, 'waist': 80}
    parsed = size_chart.parse_size_chart(d)
    assert parsed['chest'] == 95
    assert parsed['waist'] == 80


def test_parse_html_snippet():
    html = "<div>Chest: 92 cm<br>Waist: 76 cm<br>Hip: 100 cm</div>"
    parsed = size_chart.parse_size_chart(html)
    assert parsed['chest'] == 92
    assert parsed['waist'] == 76
    assert parsed['hip'] == 100


def test_material_inference_cotton():
    props = materials.infer_material_properties('100% cotton t-shirt')
    assert 'bend' in props and 'stretch' in props
    # cotton should have mid-range stretch lower than knit
    knit = materials.infer_material_properties('knit sweater')
    assert knit['stretch'] > props['stretch']
