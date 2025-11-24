import numpy as np
import os
paths=[os.path.join('src','models','smplx','smplx','SMPLX_NEUTRAL.npz'), os.path.join('src','models','smplx','smplx','SMPLX_MALE.npz'), os.path.join('src','models','smplx','smplx','SMPLX_FEMALE.npz')]
for p in paths:
    print(p)
    try:
        npz = np.load(p, allow_pickle=True)
        print('  keys:', npz.files)
        print('  sample types:', {k:type(npz[k]).__name__ for k in npz.files})
    except Exception as e:
        print('  ERROR:', e)
    print('-'*60)
