
"""
Avatar mesh generation from 2D image
Placeholder for SMPL/PIFuHD or other 3D body reconstruction integration.
"""

import numpy as np
import os
import io
import hashlib
from pathlib import Path


def _image_hash(image):
	try:
		from PIL import Image
		buf = io.BytesIO()
		if isinstance(image, Image.Image):
			image.save(buf, format='PNG')
			data = buf.getvalue()
		elif hasattr(image, 'img') and isinstance(image.img, Image.Image):
			image.img.save(buf, format='PNG')
			data = buf.getvalue()
		else:
			# fallback: attempt to get bytes attribute
			data = getattr(image, 'tobytes', lambda: b'')()
	except Exception:
		data = b''
	return hashlib.sha256(data).hexdigest()


def _cache_dir():
	root = Path(__file__).resolve().parents[2]
	cd = root / '.cache' / 'avatars'
	cd.mkdir(parents=True, exist_ok=True)
	return cd


def _save_cache(key, betas, body_pose, global_orient):
	path = _cache_dir() / f"{key}.npz"
	np.savez(path, betas=betas, body_pose=body_pose, global_orient=global_orient)


def _load_cache(key):
	path = _cache_dir() / f"{key}.npz"
	if path.exists():
		data = np.load(path)
		return data['betas'], data['body_pose'], data['global_orient']
	return None


def _apply_height_scaling(vertices, desired_height_cm):
	"""Scale mesh vertices so the mesh height matches desired_height_cm.

	Uses the vertex axis with the largest extent as 'height' axis.
	"""
	if desired_height_cm is None:
		return vertices, 1.0
	desired_m = float(desired_height_cm) / 100.0
	verts = np.asarray(vertices)
	ranges = verts.max(axis=0) - verts.min(axis=0)
	axis = int(np.argmax(ranges))
	current_height = ranges[axis]
	if current_height <= 1e-6:
		return vertices, 1.0
	scale = desired_m / float(current_height)
	verts = verts * scale
	return verts, scale


def _apply_weight_to_betas(betas, desired_weight_kg, height_m):
	"""Simple heuristic: nudge betas magnitude according to BMI deviation.

	This is not a substitute for a learned mapping; it provides a small
	shape adjustment when weight is supplied.
	"""
	if desired_weight_kg is None or height_m is None or height_m <= 0:
		return betas
	bmi = desired_weight_kg / (height_m ** 2)
	# Reference BMI ~21. Scale betas by small factor per BMI point delta
	delta = (bmi - 21.0) * 0.02
	return betas * (1.0 + delta)


def _pose_preset_arrays(preset):
	"""Return (body_pose, global_orient, left_hand, right_hand) arrays for presets.

	These are approximate, lightweight presets to produce reproducible poses for
	draping tests. They are not replacement for full pose regressors.
	"""
	# default zeros
	body_pose = np.zeros((1, 21 * 3), dtype=float)
	global_orient = np.zeros((1, 3), dtype=float)
	left_hand = np.zeros((1, 15 * 3), dtype=float)
	right_hand = np.zeros((1, 15 * 3), dtype=float)

	if preset == 'T-pose' or preset == 't-pose' or preset == 'tpose':
		# approximate: rotate shoulders so arms extend horizontally
		# shoulder joint indices depend on SMPL ordering; we approximate by
		# setting small rotations in the first few body_pose entries.
		# This is a heuristic.
		body_pose[0, :3] = 0.0
		# set some shoulder rotation values (approx)
		body_pose[0, 3:6] = [0.0, 0.8, 0.0]
		body_pose[0, 6:9] = [0.0, -0.8, 0.0]
	elif preset == 'A-pose' or preset == 'a-pose' or preset == 'apose':
		body_pose[0, 3:6] = [0.0, 0.4, 0.0]
		body_pose[0, 6:9] = [0.0, -0.4, 0.0]
	elif preset == 'hands-down' or preset == 'hands_down' or preset == 'hands-down':
		# arms by side — small negative rotation so hands point down
		body_pose[0, 3:6] = [0.0, 0.1, 0.0]
		body_pose[0, 6:9] = [0.0, -0.1, 0.0]
	elif preset == 'walking' or preset == 'walk':
		# simple alternating leg swing
		body_pose[0, 9:12] = [0.2, 0.0, 0.0]
		body_pose[0, 12:15] = [-0.2, 0.0, 0.0]

	return body_pose, global_orient, left_hand, right_hand


def _try_adapter(adapter_name, image):
	"""Attempt to run an external adapter like ROMP/PARE/PIXIE to get betas/theta.

	Returns None if adapter not available. Adapters are optional and the code
	will fall back to smplx-based generation.
	"""
	try:
		if adapter_name == 'romp':
			import romp
			# assuming romp has an API that returns betas and pose
			out = romp.estimate(image)
			return out.get('betas'), out.get('body_pose'), out.get('global_orient')
		if adapter_name == 'pare':
			import pare
			out = pare.estimate(image)
			return out.get('betas'), out.get('body_pose'), out.get('global_orient')
		if adapter_name == 'pixie':
			import pixie
			out = pixie.estimate(image)
			return out.get('betas'), out.get('body_pose'), out.get('global_orient')
	except Exception:
		return None
	return None



def generate_avatar_mesh(user_image, height_cm=None, weight_kg=None, pose_preset=None, adapter=None, cache_key=None):
	"""
	Generate a basic SMPL-X mesh (neutral pose) for visualization using the smplx package.
	Args:
		user_image (PIL.Image): Input user image (frontal). (Not used for now)
		gender (str): 'neutral', 'male', or 'female'.
	Returns:
		mesh (object): Mesh object with .vertices and .faces (numpy arrays)
		measurements (dict): Dummy body measurements
	"""
	# Default gender is 'neutral' if not provided
	gender = getattr(user_image, 'gender', 'neutral') if hasattr(user_image, 'gender') else 'neutral'
	if gender not in ['neutral', 'male', 'female']:
		gender = 'neutral'
	import smplx
	import torch
	import os
	# Use absolute path for model_folder
	current_dir = os.path.dirname(os.path.abspath(__file__))
	model_folder = os.path.abspath(os.path.join(current_dir, '..', 'models', 'smplx'))
	# Support model.npz in gender subfolders (neutral, male, female)
	model_path = os.path.join(model_folder, gender, 'model.npz')
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"SMPL-X model file for gender '{gender}' not found at {model_path}. Please ensure model.npz exists in the '{gender}' subfolder.")
	if os.path.isdir(model_path):
		raise FileNotFoundError(f"A directory named 'model.npz' exists at {model_path}, but a file is required. Please delete the directory and place the correct SMPL-X model file here.")
	# Try to detect whether the provided model.npz is SMPL (older) or SMPL-X and call smplx.create accordingly.
	# If model files are SMPL (lack SMPL-X-specific fields like hand/face components), use model_type='smpl'.
	# Try adapter first (ROMP/PARE/PIXIE) if requested
	adapter_res = None
	if adapter is not None:
		try:
			adapter_res = _try_adapter(adapter, user_image)
		except Exception:
			adapter_res = None

	# If no cache_key provided, try to derive from image
	if cache_key is None:
		try:
			cache_key = _image_hash(user_image)
		except Exception:
			cache_key = None

	# If adapter returned betas/pose use them; else try cache; else build from zeros
	betas_t = None
	body_pose_t = None
	global_orient_t = None

	if adapter_res:
		betas_t, body_pose_t, global_orient_t = adapter_res

	if betas_t is None and cache_key is not None:
		cached = _load_cache(cache_key)
		if cached is not None:
			betas_t, body_pose_t, global_orient_t = cached

	try:
		npz = np.load(model_path, allow_pickle=True)
		keys = set(npz.files)
		if 'hands_componentsl' in keys or 'hands_components' in keys or 'expr_mean' in keys or 'expression_mean' in keys:
			model_type = 'smplx'
		else:
			model_type = 'smpl'
		model_root = model_folder
		model = smplx.create(model_root, model_type=model_type, gender=gender, use_pca=False)
	except Exception as e:
		raise RuntimeError(f"Failed to load SMPL(-X) model for gender '{gender}' from {model_folder} (detected type '{locals().get('model_type','unknown')}'): {e}")

	# Default parameters
	if betas_t is None:
		betas = torch.zeros([1, 10])
	else:
		betas = torch.tensor(betas_t, dtype=torch.float32)
	if body_pose_t is None:
		body_pose = torch.zeros([1, 21 * 3])
	else:
		body_pose = torch.tensor(body_pose_t, dtype=torch.float32)
	if global_orient_t is None:
		global_orient = torch.zeros([1, 3])
	else:
		global_orient = torch.tensor(global_orient_t, dtype=torch.float32)

	# hands and face placeholders
	left_hand_pose = torch.zeros([1, 15 * 3])
	right_hand_pose = torch.zeros([1, 15 * 3])
	expression = torch.zeros([1, 10])
	jaw_pose = torch.zeros([1, 3])
	leye_pose = torch.zeros([1, 3])
	reye_pose = torch.zeros([1, 3])

	# Apply pose preset overrides if requested
	if pose_preset is not None:
		bpose, gorient, lhand, rhand = _pose_preset_arrays(pose_preset)
		try:
			body_pose = torch.tensor(bpose, dtype=torch.float32)
			global_orient = torch.tensor(gorient, dtype=torch.float32)
			left_hand_pose = torch.tensor(lhand, dtype=torch.float32)
			right_hand_pose = torch.tensor(rhand, dtype=torch.float32)
		except Exception:
			pass

	# Convert to tensors and call model
	output = model(
		betas=betas,
		body_pose=body_pose,
		global_orient=global_orient,
		left_hand_pose=left_hand_pose,
		right_hand_pose=right_hand_pose,
		expression=expression,
		jaw_pose=jaw_pose,
		leye_pose=leye_pose,
		reye_pose=reye_pose,
		return_verts=True
	)
	vertices = output.vertices.detach().cpu().numpy().squeeze()
	faces = model.faces

	# Height scaling
	vertices_scaled, scale_factor = _apply_height_scaling(vertices, height_cm)

	# Simple weight->betas adjustment (post-hoc): apply small shape scaling if weight provided
	if weight_kg is not None and height_cm is not None:
		height_m = float(height_cm) / 100.0
		# adjust betas using heuristic then re-run model for consistent vertices
		try:
			betas_np = betas.detach().cpu().numpy() if hasattr(betas, 'detach') else np.asarray(betas)
			betas_np_adj = _apply_weight_to_betas(betas_np, weight_kg, height_m)
			# ensure shape matches model expectation
			n_betas = betas_np_adj.shape[1] if betas_np_adj.ndim == 2 else betas_np_adj.shape[0]
			# create torch tensor with correct shape [1, n_betas]
			try:
				new_betas = torch.tensor(betas_np_adj, dtype=torch.float32).reshape(1, -1)
			except Exception:
				new_betas = torch.tensor(betas_np_adj, dtype=torch.float32)
			# pad or trim to model expected betas length if available
			try:
				expected = getattr(model, 'num_betas', None)
			except Exception:
				expected = None
			if expected is not None:
				cur = new_betas.shape[1]
				if cur < expected:
					# pad with zeros
					pad = np.zeros((1, expected - cur), dtype=np.float32)
					new_betas = torch.cat([new_betas, torch.from_numpy(pad)], dim=1)
				elif cur > expected:
					new_betas = new_betas[:, :expected]
			# Re-run model with updated betas to get consistent verts
			try:
				output2 = model(
					betas=new_betas,
					body_pose=body_pose,
					global_orient=global_orient,
					left_hand_pose=left_hand_pose,
					right_hand_pose=right_hand_pose,
					expression=expression,
					jaw_pose=jaw_pose,
					leye_pose=leye_pose,
					reye_pose=reye_pose,
					return_verts=True
				)
				vertices2 = output2.vertices.detach().cpu().numpy().squeeze()
				# apply height scaling to the new vertices
				vertices_scaled, scale_factor = _apply_height_scaling(vertices2, height_cm)
			except Exception:
				# fallback to previous visual tweak if re-eval fails
				try:
					vertices_scaled = vertices_scaled * (1.0 + (np.mean(betas_np_adj) * 0.01))
				except Exception:
					pass
		except Exception:
			pass

	mesh = type('Mesh', (), {})()
	mesh.vertices = vertices_scaled
	mesh.faces = faces

	# Save cache for later reuse
	if cache_key is not None:
		try:
			b_np = betas.detach().cpu().numpy() if hasattr(betas, 'detach') else np.asarray(betas)
			bp_np = body_pose.detach().cpu().numpy() if hasattr(body_pose, 'detach') else np.asarray(body_pose)
			go_np = global_orient.detach().cpu().numpy() if hasattr(global_orient, 'detach') else np.asarray(global_orient)
			_save_cache(cache_key, b_np, bp_np, go_np)
		except Exception:
			pass

	measurements = {
		'chest': 90 * scale_factor,
		'waist': 75 * scale_factor,
		'hips': 95 * scale_factor,
		'shoulders': 45 * scale_factor
	}
	return mesh, measurements
