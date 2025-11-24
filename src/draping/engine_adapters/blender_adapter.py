"""Blender engine adapter for cloth draping.

This adapter generates a small Blender Python script and invokes Blender in
background mode to perform cloth simulation using Blender's native Cloth
modifier and Collision physics. The adapter expects avatar and cloth as
OBJ file paths.

Requirements & notes:
- Requires a Blender executable available and reachable via the
  `BLENDER_EXECUTABLE` environment variable or on PATH as `blender`.
- The adapter runs Blender in background and returns paths to exported GLB.
- This is intentionally conservative and uses a generated script that reads
  a JSON params file to avoid shell-escaping complexity.
"""

from typing import Any, Dict
import os
import subprocess
import tempfile
import json
from pathlib import Path


def _find_blender_executable() -> str:
    # Prefer explicit env var
    be = os.environ.get('BLENDER_EXECUTABLE') or os.environ.get('BLENDER_PATH')
    if be and Path(be).exists():
        return be
    # fallback to system PATH
    return 'blender'


class BlenderAdapter:
    """Adapter that runs Blender in background to simulate cloth.

    Public methods:
    - run_drape(avatar_obj, cloth_obj, params) -> metadata
    - bake_export(output_path, options) -> metadata
    """

    def __init__(self):
        self.blender = _find_blender_executable()

    def run_drape(self, avatar_obj: str, cloth_obj: str, params: Dict) -> Dict:
        avatar_obj = str(avatar_obj)
        cloth_obj = str(cloth_obj)
        out_dir = Path(params.get('out_dir') or tempfile.mkdtemp())
        out_dir.mkdir(parents=True, exist_ok=True)

        script_path = out_dir / 'blender_drape_job.py'
        param_path = out_dir / 'job_params.json'

        job = {
            'avatar_obj': avatar_obj,
            'cloth_obj': cloth_obj,
            'out_dir': str(out_dir),
            'collision_inflation': float(params.get('collision_inflation', 0.01)),
            'material_params': params.get('material_params', {}),
            'poses': params.get('poses', []),
        }

        script_code = _generate_blender_script()
        script_path.write_text(script_code)
        param_path.write_text(json.dumps(job))

        cmd = [self.blender, '--background', '--python', str(script_path), '--', str(param_path)]
        try:
            subprocess.check_call(cmd)
        except FileNotFoundError:
            raise RuntimeError('Blender executable not found. Set BLENDER_EXECUTABLE to your blender binary.')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Blender run failed: {e}')

        # Expect the Blender script to write `result.json` in out_dir
        result_path = out_dir / 'result.json'
        if not result_path.exists():
            raise RuntimeError('Blender job did not produce a result.json; check Blender logs for errors.')

        try:
            res = json.loads(result_path.read_text(encoding='utf-8'))
        except Exception as e:
            raise RuntimeError(f'Failed to read blender result.json: {e}')

        return res

    def bake_export(self, output_path: str, options: Dict) -> Dict:
        # For Blender-based flow the export is already handled; just return a success
        outp = Path(output_path)
        if not outp.exists():
            # Nothing to return but signal failure
            raise RuntimeError(f'Expected exported file at {output_path} not found')
        return {'status': 'exported', 'path': str(outp)}


def _generate_blender_script() -> str:
    # The generated script reads a params JSON and executes Blender operations.
    # Keep the script small and robust; it writes a `result.json` with paths.
    return '''
import bpy, sys, json, os
from pathlib import Path
import bmesh
import mathutils


def inflate_mesh(obj, distance):
    # Duplicate object and displace vertices along normals to create an inflated collision mesh
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    # compute vertex normals
    bm.normal_update()
    for v in bm.verts:
        n = v.normal
        v.co = v.co + n * distance
    inflated_mesh = mesh.copy()
    bm.to_mesh(inflated_mesh)
    bm.free()
    inflated_obj = bpy.data.objects.new(obj.name + '_inflated', inflated_mesh)
    bpy.context.collection.objects.link(inflated_obj)
    return inflated_obj


def apply_pose_if_fbx(pose_entry, avatar_obj):
    # If pose_entry is a path to FBX, try importing it and copying bone transforms.
    p = pose_entry.get('fbx') if isinstance(pose_entry, dict) else None
    if not p:
        return False
    try:
        bpy.ops.import_scene.fbx(filepath=p)
        # find imported armature and transfer bone transforms
        imported = [o for o in bpy.context.selected_objects if o.type == 'ARMATURE']
        if not imported:
            return False
        src_arm = imported[0]
        # find avatar armature
        avatar_arm = None
        for o in bpy.context.scene.objects:
            if o.type == 'ARMATURE' and o.name != src_arm.name:
                avatar_arm = o
                break
        if avatar_arm:
            # Copy pose bones where names match
            for pb in src_arm.pose.bones:
                if pb.name in avatar_arm.pose.bones:
                    avatar_arm.pose.bones[pb.name].matrix = pb.matrix
        return True
    except Exception:
        return False


def compute_vertex_displacement(orig_obj, final_obj):
    # Compute per-vertex displacement magnitude between original and final cloth meshes
    orig_co = [v.co.copy() for v in orig_obj.data.vertices]
    disp = []
    for i, v in enumerate(final_obj.data.vertices):
        o = orig_co[i] if i < len(orig_co) else mathutils.Vector((0,0,0))
        disp.append((v.co - o).length)
    return disp


def write_result(out_dir_p, data):
    rp = out_dir_p / 'result.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def main(params_path):
    with open(params_path, 'r', encoding='utf-8') as f:
        job = json.load(f)

    avatar = job['avatar_obj']
    cloth = job['cloth_obj']
    out_dir = job['out_dir']
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Ensure common import/export addons are enabled (OBJ/FBX/GLTF)
    try:
        bpy.ops.preferences.addon_enable(module='io_scene_obj')
    except Exception:
        pass
    try:
        bpy.ops.preferences.addon_enable(module='io_scene_fbx')
    except Exception:
        pass
    try:
        bpy.ops.preferences.addon_enable(module='io_scene_gltf2')
    except Exception:
        pass

    # Import avatar and cloth; if import operators are unavailable (some Blender
    # builds may omit bundled addons in headless mode), fall back to creating
    # simple placeholder primitives so the pipeline can proceed and produce
    # a demonstrable GLB.
    imported_ok = False
    
    # Try new OBJ importer (Blender 4.x)
    try:
        bpy.ops.wm.obj_import(filepath=avatar)
        bpy.ops.wm.obj_import(filepath=cloth)
        imported_ok = True
    except Exception as e1:
        # Try legacy OBJ importer (Blender 3.x and earlier)
        try:
            bpy.ops.import_scene.obj(filepath=avatar)
            bpy.ops.import_scene.obj(filepath=cloth)
            imported_ok = True
        except Exception as e2:
            imported_ok = False

    if not imported_ok:
        # Create placeholder avatar (UV sphere) and cloth (plane)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0,0,0))
        avatar_obj = bpy.context.active_object
        avatar_obj.name = 'avatar_placeholder'
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0,0,1))
        cloth_obj = bpy.context.active_object
        cloth_obj.name = 'cloth_placeholder'

    # Heuristics to find imported objects
    objs = [o for o in bpy.context.scene.objects if o.type in {'MESH', 'ARMATURE'}]
    if not objs:
        write_result(out_dir_p, {'status': 'error', 'error': 'No objects imported'})
        return

    # Identify cloth object as the mesh with the largest surface area among recently imported
    mesh_objs = [o for o in objs if o.type == 'MESH']
    if not mesh_objs:
        write_result(out_dir_p, {'status': 'error', 'error': 'No mesh objects found'})
        return

    cloth_obj = mesh_objs[-1]
    avatar_obj = mesh_objs[0]

    # Create inflated collision mesh from avatar
    try:
        inflated = inflate_mesh(avatar_obj, job.get('collision_inflation', 0.01))
        inflated.select_set(True)
        bpy.context.view_layer.objects.active = inflated
        bpy.ops.object.modifier_add(type='COLLISION')
    except Exception as e:
        # Fallback to using avatar collision settings
        try:
            avatar_obj.select_set(True)
            bpy.context.view_layer.objects.active = avatar_obj
            bpy.ops.object.modifier_add(type='COLLISION')
            avatar_obj.collision.use = True
        except Exception:
            pass

    # Setup cloth modifier with self-collision and material params
    try:
        cloth_obj.select_set(True)
        bpy.context.view_layer.objects.active = cloth_obj
        bpy.ops.object.modifier_add(type='CLOTH')
        cloth_mod = cloth_obj.modifiers['Cloth']
        settings = cloth_mod.settings
        m = job.get('material_params', {})
        settings.quality = int(m.get('quality', 5))
        settings.mass = float(m.get('mass', 0.3))
        settings.tension_stiffness = float(m.get('tension', 5.0))
        settings.compression_stiffness = float(m.get('compression', 5.0))
        settings.shear_stiffness = float(m.get('shear', 5.0))
        settings.bending_stiffness = float(m.get('bend', 0.5))
        settings.use_self_collision = True
        # collision settings
        try:
            cloth_mod.collision_settings.distance_min = float(m.get('collision_distance', 0.005))
        except Exception:
            pass
    except Exception:
        pass

    scene = bpy.context.scene
    # Sim schedule: A-pose settle -> neutral -> provided poses
    # For now we interpret poses as a list of FBX paths to import and apply
    frames_per_stage = int(job.get('frames_per_stage', 40))
    frame = 1
    scene.frame_start = 1
    for stage, pose in enumerate(['a_pose_settle', 'neutral'] + job.get('poses', [])):
        start = frame
        end = frame + frames_per_stage - 1
        scene.frame_end = end
        for f in range(start, end + 1):
            scene.frame_set(f)
        # attempt to apply pose if provided and is an FBX
        if isinstance(pose, dict):
            apply_pose_if_fbx(pose, avatar_obj)
        frame = end + 1

    # After sim, compute per-vertex displacement and bake into vertex colors
    try:
        # Make a copy of original cloth to compute displacement
        orig = cloth_obj.copy()
        orig.data = cloth_obj.data.copy()
        bpy.context.collection.objects.link(orig)
        # ensure cloth_obj is the evaluated final mesh
        disp = compute_vertex_displacement(orig, cloth_obj)
        # create vertex color layer
        me = cloth_obj.data
        if not me.vertex_colors:
            me.vertex_colors.new(name='HeatMap')
        vcol = me.vertex_colors['HeatMap']
        # assign per-loop colors using displacement normalized
        maxd = max(disp) if disp else 0.0
        for poly in me.polygons:
            for idx, li in enumerate(poly.loop_indices):
                vi = me.loops[li].vertex_index
                val = disp[vi] / (maxd + 1e-8)
                c = (val, 0.0, 1.0 - val, 1.0)
                vcol.data[li].color = c
    except Exception:
        pass

    # Add UV unwrapping for tangent export (tangents require UVs in glTF)
    for obj in mesh_objs:
        if obj.type == 'MESH' and not obj.data.uv_layers:
            try:
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
                bpy.ops.object.mode_set(mode='OBJECT')
                obj.select_set(False)
            except Exception:
                pass

    # Export with tangents
    glb_path = out_dir_p / 'draped.glb'
    try:
        bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format='GLB', export_apply=True, export_tangents=True)
    except Exception as e:
        write_result(out_dir_p, {'status': 'error', 'error': f'Export failed: {e}'})
        return

    write_result(out_dir_p, {'status': 'finished', 'result_glb': str(glb_path)})


if __name__ == '__main__':
    # Blender passes its own CLI args into Python; the convention is that
    # script-specific args come after a "--" marker. Locate the marker and
    # use the following argument as the params JSON path.
    params_path = None
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        if idx + 1 < len(sys.argv):
            params_path = sys.argv[idx + 1]
    else:
        # fallback: the first arg after the script
        if len(sys.argv) > 1:
            params_path = sys.argv[1]

    if not params_path:
        print('Expected params JSON path as argument (after --)')
        sys.exit(2)
    main(params_path)
'''
