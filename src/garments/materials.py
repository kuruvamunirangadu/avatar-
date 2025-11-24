"""
Material inference utilities.

Given a product description or category, infer a small set of material parameters
used by simple fitting/draping heuristics: bend, shear, stretch, density, thickness.

This module intentionally provides conservative defaults and category overrides.
"""
from typing import Dict, Optional
import re


DEFAULTS = {
    'bend': 0.5,    # 0..1, lower=stiffer
    'shear': 0.5,
    'stretch': 0.5,
    'density': 0.5,  # relative
    'thickness': 0.5,
}

CATEGORY_OVERRIDES = {
    'cotton': {'bend': 0.6, 'shear': 0.5, 'stretch': 0.4, 'density': 0.6, 'thickness': 0.45},
    'denim': {'bend': 0.3, 'shear': 0.4, 'stretch': 0.2, 'density': 0.9, 'thickness': 0.8},
    'knit': {'bend': 0.8, 'shear': 0.7, 'stretch': 0.9, 'density': 0.4, 'thickness': 0.35},
    'silk': {'bend': 0.9, 'shear': 0.9, 'stretch': 0.3, 'density': 0.35, 'thickness': 0.12},
    'synthetic': {'bend': 0.6, 'shear': 0.6, 'stretch': 0.7, 'density': 0.6, 'thickness': 0.4},
}


def infer_material_properties(text: Optional[str], category: Optional[str] = None) -> Dict[str, float]:
    """Infer material properties from textual description and optional category.

    Args:
        text: product description or fabric listing (e.g., "100% cotton")
        category: optional broad category hint ("jeans", "sweater", ...)

    Returns:
        dict with keys bend, shear, stretch, density, thickness (values 0..1)
    """
    props = DEFAULTS.copy()
    if category:
        cat = category.lower()
        for k in CATEGORY_OVERRIDES:
            if k in cat:
                props.update(CATEGORY_OVERRIDES[k])
                return props

    if not text:
        return props

    t = text.lower()

    # First, check for composition percentages like '95% cotton, 5% elastane'
    # and blend category overrides by their percentages.
    comps = {}
    for match in re.finditer(r'([0-9]{1,3})%\s*([a-zA-Z]+)', t):
        pct = float(match.group(1)) / 100.0
        mat = match.group(2).lower()
        comps[mat] = comps.get(mat, 0.0) + pct

    if comps:
        # normalize to sum 1
        total = sum(comps.values())
        if total == 0:
            total = 1.0
        blended = DEFAULTS.copy()
        for mat, pct in comps.items():
            w = pct / total
            # find best matching category override key
            matched = None
            for key in CATEGORY_OVERRIDES:
                if key in mat or mat in key:
                    matched = key
                    break
            if matched is None:
                # map some synonyms
                if 'elast' in mat or 'spandex' in mat or 'lycra' in mat:
                    matched = 'synthetic'
            if matched:
                override = CATEGORY_OVERRIDES[matched]
                for k in blended:
                    blended[k] = blended[k] * (1 - w) + override[k] * w
        return blended

    # crude keyword matching fallback
    for k in CATEGORY_OVERRIDES:
        if k in t:
            props.update(CATEGORY_OVERRIDES[k])
            return props

    # heuristic: look for common fabric words
    if 'cotton' in t:
        props.update(CATEGORY_OVERRIDES['cotton'])
    elif 'denim' in t or 'jean' in t:
        props.update(CATEGORY_OVERRIDES['denim'])
    elif 'knit' in t or 'wool' in t or 'sweater' in t:
        props.update(CATEGORY_OVERRIDES['knit'])
    elif 'silk' in t or 'satin' in t:
        props.update(CATEGORY_OVERRIDES['silk'])
    elif 'poly' in t or 'nylon' in t or 'polyester' in t or 'elast' in t or 'spandex' in t:
        props.update(CATEGORY_OVERRIDES['synthetic'])

    return props
