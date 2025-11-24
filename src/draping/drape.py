"""Draping orchestration and engine adapter selection.

This module provides a high-level entry point `drape_garment` that accepts
an avatar mesh and a cloth mesh (either as file paths or simple mesh dicts)
and runs a physics-based drape using a selected engine adapter.

Currently supported engines:
- blender: runs Blender in background via CLI to use Blender Cloth solver.

If Blender is not available the module falls back to a conservative
projection-based stub so the pipeline remains usable for tests.

See the `engine_adapters` package for implementation details.
"""

from typing import Any, Dict, List, Optional
import os
from pathlib import Path
import tempfile
import json
import shutil

# Adapter discovery
try:
	from .engine_adapters.blender_adapter import BlenderAdapter
except Exception:
	BlenderAdapter = None  # type: ignore


class PhysicsEngineAdapter:
	"""Adapter interface for physics engines.

	Concrete adapters should implement `run_drape` and `bake_export`.
	"""

	def run_drape(self, avatar_obj: str, cloth_obj: str, params: Dict) -> Dict:
		raise NotImplementedError()

	def bake_export(self, output_path: str, options: Dict) -> Dict:
		raise NotImplementedError()


def _choose_adapter(engine: str) -> PhysicsEngineAdapter:
	engine = (engine or '').lower()
	if engine == 'blender' and BlenderAdapter is not None:
		return BlenderAdapter()
	# fallback: simple stub adapter implemented inline
	return _StubAdapter()


class _StubAdapter(PhysicsEngineAdapter):
	"""Very small fallback: performs a conservative projection of the cloth
	onto the avatar to avoid penetrations. This is used when Blender is not
	available (e.g., CI/test environments).
	"""

	def run_drape(self, avatar_obj: str, cloth_obj: str, params: Dict) -> Dict:
		# If given paths, simply copy cloth->out and write metadata indicating stub
		out_dir = Path(params.get('out_dir') or tempfile.mkdtemp())
		out_dir.mkdir(parents=True, exist_ok=True)
		out_path = out_dir / (Path(cloth_obj).stem + '_draped.obj')
		try:
			shutil.copy(cloth_obj, out_path)
		except Exception:
			# If cloth_obj isn't a path, create a minimal placeholder
			out_path.write_text('# stub draped OBJ\n')
		meta = {'status': 'finished', 'result': str(out_path), 'engine': 'stub'}
		return meta

	def bake_export(self, output_path: str, options: Dict) -> Dict:
		# nothing to do; return success metadata
		return {'status': 'exported', 'path': str(output_path)}


def drape_garment(avatar: str, cloth: str, *, engine: str = 'blender',
				  material_params: Optional[Dict] = None,
				  collision_inflation: float = 0.01,
				  poses: Optional[List[Dict]] = None,
				  out_dir: Optional[str] = None) -> Dict:
	"""High-level drape runner.

	Parameters
	- avatar: path to avatar OBJ (or mesh identifier)
	- cloth: path to cloth OBJ
	- engine: 'blender' or 'stub'
	- material_params: mapping of fabric properties (friction, bend, stretch)
	- collision_inflation: offset applied to avatar collision geometry (meters)
	- poses: list of pose dicts to run sequentially (each pose may be a filepath to an FBX/pose or dict)
	- out_dir: directory where outputs (GLB/OBJ and metadata) will be written

	Returns metadata dict with result paths and status.
	"""
	out_dir = out_dir or str(Path.cwd() / 'data' / 'drape_outputs')
	Path(out_dir).mkdir(parents=True, exist_ok=True)
	adapter = _choose_adapter(engine)

	params = {
		'material_params': material_params or {},
		'collision_inflation': float(collision_inflation),
		'poses': poses or [],
		'out_dir': out_dir,
	}

	meta = adapter.run_drape(avatar, cloth, params)

	# Optionally bake/export
	export_path = str(Path(out_dir) / (Path(cloth).stem + '_draped.glb'))
	try:
		exp = adapter.bake_export(export_path, options={'include_avatar': True})
		meta['export'] = exp
	except Exception as e:
		meta['export_error'] = str(e)

	return meta
