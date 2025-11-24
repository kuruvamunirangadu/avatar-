import importlib, sys

try:
    mp = importlib.import_module('mediapipe')
    print('mediapipe_version:', getattr(mp, '__version__', 'unknown'))
except Exception as e:
    print('IMPORT_ERROR:', repr(e))
    sys.exit(1)
