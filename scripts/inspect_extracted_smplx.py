import numpy as np
import os
paths=[
    os.path.join('src','models','smplx','smplx','neutral','SMPLX_NEUTRAL.npz'),
    os.path.join('src','models','smplx','smplx','male','SMPLX_MALE.npz'),
    os.path.join('src','models','smplx','smplx','female','SMPLX_FEMALE.npz'),
    os.path.join('src','models','smplx','smplx','models','smplx','SMPLX_NEUTRAL.npz'),
]
for p in paths:
    print(p)
    if not os.path.exists(p):
        print('  MISSING')
        print('-'*40)
        continue
    try:
        npz=np.load(p, allow_pickle=True)
        print('  keys:', npz.files)
        print('  types:', {k:type(npz[k]).__name__ for k in npz.files})
        # print shapes for large arrays (limit to small set)
        for k in npz.files[:10]:
            v = npz[k]
            try:
                print(f'    {k}: shape={getattr(v, "shape", None)}, dtype={getattr(v, "dtype", None)}')
            except Exception:
                pass
    except Exception as e:
        print('  ERROR loading:', e)
    print('-'*40)
