from pygltflib import GLTF2
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print('Usage: inspect_glb.py <path_to_glb>')
    sys.exit(2)

path = Path(sys.argv[1])
if not path.exists():
    print('MISSING', path)
    sys.exit(1)

try:
    g = GLTF2().load(str(path))
except Exception as e:
    print('ERROR_LOADING', e)
    sys.exit(1)

print('Loaded GLB:', path)
print('Scenes:', len(g.scenes) if g.scenes else 0)
print('Nodes:', len(g.nodes) if g.nodes else 0)
print('Meshes:', len(g.meshes) if g.meshes else 0)

for mi, mesh in enumerate(g.meshes or []):
    print(f'Mesh[{mi}] name={mesh.name} primitives={len(mesh.primitives)}')
    for pi, prim in enumerate(mesh.primitives):
        attrs = prim.attributes
        present = {k: v for k, v in attrs.__dict__.items() if v is not None}
        keys = list(present.keys())
        print(f'  Prim[{pi}] attributes={keys}')
        # Common attribute checks
        has_color = 'COLOR_0' in keys or 'COLOR0' in keys
        has_tangent = 'TANGENT' in keys
        has_normal = 'NORMAL' in keys
        has_texcoord = any(k.startswith('TEXCOORD') for k in keys)
        print(f'    has_normal={has_normal} has_tangent={has_tangent} has_texcoord={has_texcoord} has_vertex_color={has_color}')

print('Accessors:', len(g.accessors) if g.accessors else 0)
print('BufferViews:', len(g.bufferViews) if g.bufferViews else 0)
print('Buffers:', len(g.buffers) if g.buffers else 0)
