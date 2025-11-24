"""
Size chart parsing utilities.

Supports:
 - dict input (passes through and normalizes)
 - HTML snippet or text (heuristic regex-based extraction)
 - CSV file path (simple header matching)

Returns canonical dict with keys: chest, waist, hip, length, rise, sleeve (values in cm)
"""
from typing import Any, Dict
import re
import os

# Try to import pandas for CSV support, but keep this optional to avoid
# adding a hard dependency for lightweight installs.
try:
    import pandas as pd
except Exception:
    pd = None


def _to_cm(value: str) -> float:
    # Basic conversion: if value in mm or inches, try to convert; otherwise assume cm.
    s = value.strip().lower()
    if s.endswith('cm'):
        return float(s[:-2])
    if s.endswith('mm'):
        return float(s[:-2]) / 10.0
    if s.endswith('in') or s.endswith('"'):
        try:
            return float(s.rstrip('in"')) * 2.54
        except Exception:
            pass
    # if plain number, assume cm
    try:
        return float(re.sub('[^0-9.]', '', s))
    except Exception:
        raise ValueError(f"Cannot parse measurement: {value}")


def parse_size_chart(input_data: Any) -> Dict[str, float]:
    """
    Parse size chart from multiple kinds of inputs and return canonical measurements.

    Accepts dict, HTML/text, or a path to a CSV/HTML file.
    """
    keys = ['chest', 'waist', 'hip', 'length', 'rise', 'sleeve']
    result = {}

    if isinstance(input_data, dict):
        for k in keys:
            if k in input_data and input_data[k] not in (None, ''):
                result[k] = float(input_data[k])
        return result

    # If input looks like a path to a file
    if isinstance(input_data, str) and os.path.exists(input_data):
        # CSV handling if pandas is available
        if pd is not None and input_data.lower().endswith('.csv'):
            try:
                df = pd.read_csv(input_data)
                # Heuristic: look for columns matching keys
                columns = {c.lower(): c for c in df.columns}
                for k in keys:
                    for col_lower, col in columns.items():
                        if k in col_lower or ('bust' in col_lower and k == 'chest'):
                            try:
                                val = float(df.iloc[0][col])
                                result[k] = val
                            except Exception:
                                continue
                return result
            except Exception:
                pass
        try:
            text = open(input_data, 'r', encoding='utf-8', errors='ignore').read()
        except Exception:
            text = input_data
    else:
        text = str(input_data)

    # Heuristic regex: look for measurement words near numbers
    patterns = {
        'chest': r'(?:chest|bust)\s*[:\-]?\s*([0-9]+\.?[0-9]*\s*(?:cm|mm|in|\"|))',
        'waist': r'waist\s*[:\-]?\s*([0-9]+\.?[0-9]*\s*(?:cm|mm|in|\"|))',
        'hip': r'hip\s*[:\-]?\s*([0-9]+\.?[0-9]*\s*(?:cm|mm|in|\"|))',
        'length': r'(?:length|body length|dress length|shirt length)\s*[:\-]?\s*([0-9]+\.?[0-9]*\s*(?:cm|mm|in|\"|))',
        'rise': r'rise\s*[:\-]?\s*([0-9]+\.?[0-9]*\s*(?:cm|mm|in|\"|))',
        'sleeve': r'sleeve\s*[:\-]?\s*([0-9]+\.?[0-9]*\s*(?:cm|mm|in|\"|))',
    }

    found = {}
    lower_text = text.lower()
    # Vendor-specific heuristics
    def detect_vendor(s: str):
        if 'asos' in s:
            return 'asos'
        if 'zara' in s:
            return 'zara'
        if 'hm' in s or 'h&m' in s:
            return 'hm'
        return None

    vendor = detect_vendor(lower_text)
    # vendor-specific parsing shortcuts
    if vendor == 'asos' and pd is not None:
        # ASOS CSVs often have headers like 'Chest (cm)' or 'Bust (cm)'
        try:
            df = pd.read_csv(input_data) if isinstance(input_data, str) and os.path.exists(input_data) else pd.read_csv(pd.compat.StringIO(text))
            cols = {c.lower(): c for c in df.columns}
            for k in ['chest', 'waist', 'hip']:
                for col_lower, col in cols.items():
                    if k in col_lower or ('bust' in col_lower and k == 'chest'):
                        try:
                            found[k] = float(df.iloc[0][col])
                        except Exception:
                            continue
            if found:
                return found
        except Exception:
            pass
    for field, pat in patterns.items():
        m = re.search(pat, lower_text)
        if m:
            try:
                found[field] = _to_cm(m.group(1))
            except Exception:
                continue

    # If nothing found, try to extract numbers in order (fallback heuristics)
    if not found:
        numbers = re.findall(r'([0-9]+\.?[0-9]*)\s*(?:cm|mm|in|\"|)?', lower_text)
        nums = [float(n) for n in numbers]
        if len(nums) >= 3:
            # assume chest, waist, hip
            found['chest'] = nums[0]
            found['waist'] = nums[1]
            found['hip'] = nums[2]

    # If we found values, normalize to cm (some may already be cm)
    for k, v in list(found.items()):
        try:
            found[k] = float(v)
        except Exception:
            try:
                found[k] = _to_cm(str(v))
            except Exception:
                del found[k]

    return found
