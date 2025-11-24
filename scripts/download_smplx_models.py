"""
Try to download official SMPL-X .npz model files into the repo.

Usage:
  - Run from repository root (virtual-tryon) with your project venv active.
  - The script will attempt several known candidate URLs. If the host requires an interactive license acceptance or login, the download will fail and the script will print instructions for manual download and placement.

Example:
    Activate your project's venv and run the script from the `virtual-tryon` folder, for example:
        - activate the venv (Windows): `.venv310\\Scripts\\activate.bat`
        - run the script: `python scripts/download_smplx_models.py --out src/models/smplx`

Notes:
  - This script does not bypass any license or authentication protection. If the official SMPL-X host requires you to accept a license or login, please download the files manually and place them in the output folder.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except Exception:
    print('This script requires the requests package. Install it in your venv: pip install requests')
    sys.exit(2)

CANDIDATE_URLS = {
    'SMPLX_NEUTRAL.npz': [
        # Community/author repo raw locations (may exist)
        'https://github.com/vchoutas/smplx/raw/master/models/smplx/SMPLX_NEUTRAL.npz',
        'https://github.com/facebookresearch/smplx/raw/master/models/SMPLX_NEUTRAL.npz',
        # Some mirrors (may or may not exist)
        'https://smpl-x.is.tue.mpg.de/models/SMPLX_NEUTRAL.npz',
    ],
    'SMPLX_MALE.npz': [
        'https://github.com/vchoutas/smplx/raw/master/models/smplx/SMPLX_MALE.npz',
        'https://smpl-x.is.tue.mpg.de/models/SMPLX_MALE.npz',
    ],
    'SMPLX_FEMALE.npz': [
        'https://github.com/vchoutas/smplx/raw/master/models/smplx/SMPLX_FEMALE.npz',
        'https://smpl-x.is.tue.mpg.de/models/SMPLX_FEMALE.npz',
    ],
}


def download_file(url, out_path, chunk_size=4096):
    r = requests.get(url, stream=True, allow_redirects=True, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f'HTTP {r.status_code} for {url}')
    total = int(r.headers.get('content-length', 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
    return out_path


def try_candidates(name, urls, out_dir):
    out_path = Path(out_dir) / name
    for url in urls:
        print(f'Trying {url}...')
        try:
            download_file(url, out_path)
            print(f'  downloaded to {out_path}')
            return out_path
        except Exception as e:
            print(f'  failed: {e}')
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='src/models/smplx', help='Output folder for model files (default: src/models/smplx)')
    p.add_argument('--force', action='store_true', help='Overwrite existing files')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, urls in CANDIDATE_URLS.items():
        target = out_dir / name
        if target.exists() and not args.force:
            print(f'{target} already exists, skipping (use --force to overwrite)')
            results[name] = str(target)
            continue
        got = try_candidates(name, urls, out_dir)
        if got is None:
            print('\nCould not download', name)
            print('If the official SMPL-X site requires license acceptance or login, please:')
            print('  1) Visit the official SMPL-X site, accept the license, and download the files:')
            print('     - SMPLX_NEUTRAL.npz, SMPLX_MALE.npz, SMPLX_FEMALE.npz')
            print('  2) Place them under', out_dir)
            print('  3) Run this script again')
            results[name] = None
        else:
            # Also create gender subfolders with model.npz expected by code
            gender_map = {
                'SMPLX_NEUTRAL.npz': ('neutral', 'model.npz'),
                'SMPLX_MALE.npz': ('male', 'model.npz'),
                'SMPLX_FEMALE.npz': ('female', 'model.npz'),
            }
            if name in gender_map:
                g, fname = gender_map[name]
                dest = out_dir / g / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                # copy
                with open(got, 'rb') as srcf, open(dest, 'wb') as dstf:
                    dstf.write(srcf.read())
                print(f'  also copied to {dest}')
            results[name] = str(got)

    print('\nSummary:')
    for k,v in results.items():
        print(' ', k, '->', v)

    missing = [k for k,v in results.items() if v is None]
    if missing:
        print('\nOne or more files could not be downloaded. Please download them manually and re-run this script.')
        sys.exit(3)
    else:
        print('\nAll model files present. Next: install a matching smplx package in your venv and run scripts/test_generate_avatar.py')

if __name__ == '__main__':
    main()
