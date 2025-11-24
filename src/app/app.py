import os
import sys
import numpy as np
import importlib.util
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os as _os
import numpy as _np
import traceback
import json
from datetime import datetime

# Ensure the project root is on sys.path so `from src...` imports work
# This is needed when Streamlit runs the app file directly because it
# inserts the app directory into sys.path and the sibling `src` package
# is not found unless the project root is added.
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from src.preprocessing.preprocess import (
	load_obj_mesh,
	remove_background,
	detect_keypoints,
	segment_garment,
	detect_garment_edges,
)
from src.avatar.generate_mesh import generate_avatar_mesh
from src.garments.garment_loader import generate_garment_mesh
from src.fit_model.fit_metrics import compute_fit_percentage
# Optional templates list for UI
try:
	from src.garments.templates import LIST_TEMPLATES
except Exception:
	LIST_TEMPLATES = ['tee', 'shirt', 'jeans', 'dress']

# PIFuHD integration (optional)
try:
	from src.garments.pifuhd_integration import reconstruct_image_to_mesh
except Exception:
	reconstruct_image_to_mesh = None


def extract_real_measurements(keypoints, ref_length_px, ref_length_cm):
	"""Estimate real-world body measurements from keypoints using a reference object.

	Returns a dict of measurements in cm. Minimal implementation kept for demo/tests.
	"""
	if not keypoints or ref_length_px == 0 or ref_length_cm == 0:
		return {}
	px_to_cm = ref_length_cm / ref_length_px
	shoulder_width_cm = None
	if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
		ls = np.array(keypoints['left_shoulder'])
		rs = np.array(keypoints['right_shoulder'])
		shoulder_width_px = np.linalg.norm(ls - rs)
		shoulder_width_cm = shoulder_width_px * px_to_cm
	return {'shoulder_width_cm': shoulder_width_cm}


def main():
	st.title("AI-Driven 3D Virtual Try-On")
	st.write("Upload your photo and garment to see the virtual try-on!")

	# Model detection status (SMPL vs SMPL-X)
	def _detect_model_type():
		"""Return one of: 'SMPL-X', 'SMPL', 'missing', or 'error:<msg>'"""
		try:
			current_dir = _os.path.dirname(_os.path.abspath(__file__))
			model_folder = _os.path.abspath(_os.path.join(current_dir, '..', 'models', 'smplx'))
			# Check top-level SMPLX NPZ
			top = _os.path.join(model_folder, 'SMPLX_NEUTRAL.npz')
			if _os.path.exists(top):
				npz = _np.load(top, allow_pickle=True)
				keys = set(npz.files)
				if 'hands_componentsl' in keys or 'hands_componentsr' in keys or 'hands_meanl' in keys:
					return 'SMPL-X'
			# Check gender subfolder model.npz
			neutral = _os.path.join(model_folder, 'neutral', 'model.npz')
			if _os.path.exists(neutral):
				npz = _np.load(neutral, allow_pickle=True)
				keys = set(npz.files)
				if 'hands_componentsl' in keys or 'hands_componentsr' in keys:
					return 'SMPL-X'
				# If no SMPL-X-specific fields, assume plain SMPL
				return 'SMPL'
			# Nothing found
			return 'missing'
		except Exception as e:
			# Store the full traceback for debugging in the Streamlit session state
			tb = traceback.format_exc()
			try:
				st.session_state['model_error_trace'] = tb
			except Exception:
				# If session_state isn't available in this context, ignore storing the trace
				pass
			return f'error:{e}'

	model_status = _detect_model_type()
	# Allow user to re-run detection from the sidebar and surface detailed traces
	if st.sidebar.button("Re-check models"):
		model_status = _detect_model_type()
		# refresh displayed status immediately (no full rerun)
	st.sidebar.subheader('Model status')
	if model_status == 'SMPL-X':
		st.sidebar.success('SMPL-X models detected — full hands & expression support available')
	elif model_status == 'SMPL':
		st.sidebar.info('SMPL models detected — limited (no hands/expressions)')
	elif model_status == 'missing':
		st.sidebar.warning('No SMPL/SMPL-X models found. App will use fallback avatar.')
	else:
		st.sidebar.error(f'Model detection error: {model_status}')
		# Show an expander with the stored traceback when available to help debugging
		if 'model_error_trace' in st.session_state:
			with st.sidebar.expander('Show model detection traceback'):
				st.text(st.session_state.get('model_error_trace'))

	# Minimal UI: no hardcoded API keys, no SMPLify-X automation included here.

	# User image upload
	user_image = st.file_uploader("Upload your photo (front view)", type=["jpg", "jpeg", "png"])

	# Simple gender selection (user-controlled). Automatic detection removed for privacy and simplicity.
	gender = st.selectbox("Select avatar gender", options=['neutral', 'male', 'female'], index=0)

	# Reference object input (for measurement scaling)
	ref_length_cm = st.number_input("Reference object real length (cm)", min_value=1.0, max_value=200.0, value=8.5)
	ref_length_px = st.number_input("Reference object length in image (pixels)", min_value=1, value=100)

	# Garment input
	garment_file = st.file_uploader("Upload garment (OBJ, STL, image)", type=["obj", "stl", "png", "jpg", "jpeg"])

	# Garment reconstruction UI (template + size chart + material)
	st.subheader("Garment Reconstruction (templates)")
	col1, col2 = st.columns([2, 1])
	with col1:
		template_choice = st.selectbox('Template', options=LIST_TEMPLATES, index=0)
		garment_color = st.color_picker('Garment Color', value='#F5DEB3')  # Wheat/beige default
		size_chart_paste = st.text_area('Paste size chart HTML/text (optional)', height=80)
		size_chart_csv = st.file_uploader('Or upload size chart CSV (optional)', type=['csv'])
		material_text = st.text_input('Product/material description (optional)', value='')
	with col2:
		st.markdown('Template preview')
		# Show a small SVG preview if available
		try:
			import streamlit.components.v1 as components
			svg_path = os.path.join(project_root, 'src', 'garments', 'fixtures', f"{template_choice}.svg")
			if os.path.exists(svg_path):
				svg_content = open(svg_path, 'r', encoding='utf-8').read()
				components.html(svg_content, height=180)
			else:
				st.text('No preview available')
		except Exception:
			st.text('Preview unavailable')

	generate_garment_btn = st.button('Generate garment from template')

	# default size chart fallback to dataset CSV
	default_size_chart_path = os.path.join("data", "raw", "size_chart.csv")
	size_chart = None
	if size_chart_csv is not None:
		try:
			# prefer to pass the uploaded file object (generate_garment_mesh/parse will handle CSV via pandas if available)
			size_chart = size_chart_csv
		except Exception:
			size_chart = None
	elif os.path.exists(default_size_chart_path):
		try:
			size_chart = pd.read_csv(default_size_chart_path)
		except Exception:
			size_chart = None


	# Preprocessing & keypoints
	if user_image is not None:
		from PIL import Image

		image_np = Image.open(user_image).convert('RGB')
		st.image(image_np, caption="Original Image", use_column_width=True)
		bg_removed = remove_background(image_np)
		st.image(bg_removed, caption="Background Removed", use_column_width=True)

		st.subheader("Garment preprocessing (placeholders)")
		segmented = segment_garment(bg_removed)
		st.image(segmented, caption="Garment Segmentation (placeholder)", use_column_width=True)
		edges = detect_garment_edges(bg_removed)
		st.image(edges, caption="Garment Edges (placeholder)", use_column_width=True)

		keypoints = detect_keypoints(bg_removed)
		st.write("Detected Keypoints:", keypoints)
	else:
		image_np = None
		bg_removed = None
		keypoints = None

	# Measurement estimation
	real_measurements = None
	if keypoints is not None and ref_length_px > 0 and ref_length_cm > 0:
		real_measurements = extract_real_measurements(keypoints, ref_length_px, ref_length_cm)
		st.write("Estimated Real-World Measurements (cm):", real_measurements)

	# Avatar generation (SMPL-X placeholder integration)
	avatar_mesh = None
	avatar_measurements = None
	if image_np is not None:
		# generate_avatar_mesh currently expects an image-like object; attach gender attribute
		try:
			image_np.gender = gender
		except Exception:
			# If PIL Image is immutable in this environment, wrap into a simple object
			class ImgWrapper:
				def __init__(self, img, gender):
					self.img = img
					self.gender = gender

			wrapped = ImgWrapper(image_np, gender)
			image_for_avatar = wrapped
		else:
			image_for_avatar = image_np

		try:
			# UI: allow users to provide height/weight, pose preset and adapter selection
			st.sidebar.subheader('Avatar controls')
			user_height = st.sidebar.number_input('User height (cm)', min_value=50, max_value=250, value=170)
			user_weight = st.sidebar.number_input('User weight (kg, optional)', min_value=0.0, value=0.0)
			if user_weight <= 0.0:
				user_weight = None
			pose_options = ['None', 'T-pose', 'A-pose', 'hands-down', 'walking']
			pose_choice = st.sidebar.selectbox('Pose preset', options=pose_options, index=0)
			pose_preset = None if pose_choice == 'None' else pose_choice
			# Detect installed adapters
			adapters = ['none']
			for mod in ('romp', 'pare', 'pixie'):
				if importlib.util.find_spec(mod) is not None:
					adapters.append(mod)
			adapter_choice = st.sidebar.selectbox('Adapter (optional)', options=adapters, index=0)
			adapter = None if adapter_choice == 'none' else adapter_choice
			avatar_mesh, avatar_measurements = generate_avatar_mesh(image_for_avatar, height_cm=user_height, weight_kg=user_weight, pose_preset=pose_preset, adapter=adapter)
			st.write("Avatar Measurements (model units):", avatar_measurements)
		except FileNotFoundError as fe:
			st.error(f"Avatar generation failed: {fe}")
			st.info("To enable 3D avatar generation, place SMPL-X model.npz files under src/models/smplx/<gender>/model.npz or follow README instructions.")
		except Exception as e:
			st.error(f"Avatar generation failed: {e}")
			# Fallback: create a simple cube mesh for demo purposes so visualization and fit flow continue
			class Mesh:
				pass

			cube_vertices = np.array([
				[-0.5, -0.5, -0.5],
				[0.5, -0.5, -0.5],
				[0.5, 0.5, -0.5],
				[-0.5, 0.5, -0.5],
				[-0.5, -0.5, 0.5],
				[0.5, -0.5, 0.5],
				[0.5, 0.5, 0.5],
				[-0.5, 0.5, 0.5],
			])
			# 12 triangles (two per face)
			cube_faces = np.array([
				[0, 1, 2], [0, 2, 3],
				[4, 5, 6], [4, 6, 7],
				[0, 1, 5], [0, 5, 4],
				[2, 3, 7], [2, 7, 6],
				[1, 2, 6], [1, 6, 5],
				[0, 3, 7], [0, 7, 4]
			])
			mesh = Mesh()
			mesh.vertices = cube_vertices
			mesh.faces = cube_faces
			avatar_mesh = mesh
			avatar_measurements = {'chest': 90, 'waist': 75, 'hips': 95}
			st.info("Using a fallback demo avatar (simple cube). Replace with SMPL-X models for realistic avatars.")

	# Garment mesh generation (from uploaded mesh OR template-based reconstruction)
	garment_mesh = None
	garment_dims = None
	garment_material = None
	garment_meta = None
	if garment_file is not None and not generate_garment_btn:
		# If a real garment mesh was uploaded (OBJ/STL), prefer using it directly
		try:
			garment_mesh, garment_dims, garment_material, garment_meta = generate_garment_mesh(garment_file)
			st.write("Garment dimensions:", garment_dims)
		except Exception as e:
			st.warning(f"Failed to load uploaded garment: {e}")

	if generate_garment_btn:
		# Prepare size_chart_input: prefer uploaded CSV, then pasted text, then dataframe
		size_input = None
		if size_chart_csv is not None:
			size_input = size_chart_csv
		elif size_chart_paste and len(size_chart_paste.strip())>0:
			size_input = size_chart_paste
		elif isinstance(size_chart, pd.DataFrame):
			# pass first row as dict
			size_input = size_chart.iloc[0].to_dict()
		else:
			size_input = None

		# Optional PIFuHD path (experimental)
		use_pifu = st.sidebar.checkbox('Use PIFuHD (R&D - requires GPU and large deps)', value=False)
		
		garment_mesh = None
		garment_dims = None
		garment_material = {}
		garment_meta = {}
		
		if use_pifu and reconstruct_image_to_mesh is not None and image_np is not None:
			try:
				st.info('Running PIFuHD reconstruction (this may take a long time). Check logs for progress...')
				# reconstruct_image_to_mesh returns an object or path; handle both
				pifu_result = reconstruct_image_to_mesh(image_np, device='cpu')
				if isinstance(pifu_result, str) and pifu_result.lower().endswith('.obj'):
					# load the OBJ via existing loader if available
					try:
						from src.preprocessing.preprocess import load_obj_mesh
						pifu_mesh = load_obj_mesh(pifu_result)
						# pifu mesh becomes the garment source (experimental)
						garment_mesh = pifu_mesh
						garment_dims = None
						garment_material = {}
						garment_meta = {'source': 'pifuhd', 'obj_path': pifu_result}
					except Exception:
						st.warning('PIFuHD returned OBJ but failed to load it; falling back to template.')
						garment_mesh, garment_dims, garment_material, garment_meta = generate_garment_mesh(None, template=template_choice, size_chart_input=size_input, material_text=material_text)
				else:
					# If result is a mesh-like object, accept it directly
					garment_mesh = pifu_result
					garment_dims = None
					garment_material = {}
					garment_meta = {'source': 'pifuhd'}
			except Exception as e:
				st.warning(f'PIFuHD reconstruction unavailable or failed: {e}')
				garment_mesh, garment_dims, garment_material, garment_meta = generate_garment_mesh(None, template=template_choice, size_chart_input=size_input, material_text=material_text)
		
		if garment_mesh is None:
			# Template-based generation
			try:
				# Show sliders to adjust template parameters live
				st.subheader('Template parameters')
				# Defaults
				defaults = {'chest': 92.0, 'waist': 78.0, 'hip': 98.0, 'length': 65.0, 'rise': 25.0, 'sleeve': 20.0}
				# If the size_input has values, prefer them
				if isinstance(size_input, dict):
					for k in defaults:
						if k in size_input:
							try:
								defaults[k] = float(size_input[k])
							except Exception:
								pass

				# Create sliders
				param_values = {}
				cols = st.columns(2)
				with cols[0]:
					param_values['chest'] = st.slider('Chest (cm)', min_value=50.0, max_value=160.0, value=float(defaults['chest']))
					param_values['waist'] = st.slider('Waist (cm)', min_value=40.0, max_value=140.0, value=float(defaults['waist']))
					param_values['hip'] = st.slider('Hip (cm)', min_value=60.0, max_value=170.0, value=float(defaults['hip']))
				with cols[1]:
					param_values['length'] = st.slider('Length (cm)', min_value=20.0, max_value=120.0, value=float(defaults['length']))
					param_values['rise'] = st.slider('Rise (cm)', min_value=10.0, max_value=40.0, value=float(defaults['rise']))
					param_values['sleeve'] = st.slider('Sleeve (cm)', min_value=0.0, max_value=80.0, value=float(defaults['sleeve']))

				# Live generate the template mesh with current slider values
				garment_mesh, garment_dims, garment_material, garment_meta = generate_garment_mesh(None, template=template_choice, size_chart_input=param_values, material_text=material_text, params=param_values)
			except Exception as e:
				st.error(f'Could not generate garment: {e}')

	if garment_dims is not None:
		st.write("Garment dimensions:", garment_dims)
	if garment_material is not None and garment_material:
		st.write("Inferred material properties:", garment_material)

	# Physics-based draping (optional)
	use_physics_draping = st.sidebar.checkbox('Use Physics-Based Draping (requires Blender)', value=False)
	draped_garment_mesh = None
	draped_glb_path = None
	
	if use_physics_draping and avatar_mesh is not None and garment_mesh is not None:
		try:
			import tempfile
			from src.draping.drape import drape_garment
			
			# Set Blender executable path
			os.environ['BLENDER_EXECUTABLE'] = r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe'
			
			st.info('🎨 Running physics-based cloth simulation with Blender... This may take 30-60 seconds.')
			
			# Create temporary directory for draping
			with tempfile.TemporaryDirectory() as tmpdir:
				# Export avatar and garment to OBJ files
				avatar_obj_path = os.path.join(tmpdir, 'avatar.obj')
				garment_obj_path = os.path.join(tmpdir, 'garment.obj')
				drape_out_dir = os.path.join(tmpdir, 'draped')
				
				# Write OBJ files
				def write_mesh_obj(mesh, path):
					with open(path, 'w') as f:
						if hasattr(mesh, 'vertices'):
							verts = mesh.vertices
						elif isinstance(mesh, tuple) and len(mesh) > 0:
							verts = mesh[0]
						else:
							verts = mesh
						
						for v in verts:
							f.write(f"v {v[0]} {v[1]} {v[2]}\n")
						
						if hasattr(mesh, 'faces'):
							faces = mesh.faces
						elif isinstance(mesh, tuple) and len(mesh) > 1:
							faces = mesh[1]
						else:
							faces = None
						
						if faces is not None:
							for face in faces:
								f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
				
				write_mesh_obj(avatar_mesh, avatar_obj_path)
				write_mesh_obj(garment_mesh, garment_obj_path)
				
				# Run draping with material parameters
				material_params = {
					'mass': garment_material.get('density', 0.3),
					'tension': 15.0,
					'compression': 15.0,
					'shear': garment_material.get('shear', 5.0),
					'bend': garment_material.get('bend', 0.5),
					'quality': 5,
					'collision_distance': 0.015
				}
				
				result = drape_garment(
					avatar=avatar_obj_path,
					cloth=garment_obj_path,
					engine='blender',
					material_params=material_params,
					collision_inflation=0.01,
					poses=[
						{'frames': 50},  # A-pose settle
						{'frames': 30}   # Neutral pose
					],
					out_dir=drape_out_dir
				)
				
				if result.get('status') == 'finished' and result.get('result_glb'):
					draped_glb_path = result['result_glb']
					st.success('✅ Physics-based draping completed!')
					
					# Load the draped GLB and extract cloth mesh
					try:
						import pygltflib
						gltf = pygltflib.GLTF2().load(draped_glb_path)
						# For now, show download link - full GLB parsing would require more work
						with open(draped_glb_path, 'rb') as f:
							glb_bytes = f.read()
						st.download_button(
							label="📥 Download Draped Garment (GLB)",
							data=glb_bytes,
							file_name="draped_garment.glb",
							mime="model/gltf-binary"
						)
						
						# Show vertex color heatmap info
						st.info('💡 The GLB file contains vertex color heatmap showing cloth displacement.')
						
						# Embed three.js GLB viewer
						st.subheader('🎨 Physics-Based Draped Result')
						import base64
						from streamlit.components.v1 import html
						
						glb_b64 = base64.b64encode(glb_bytes).decode('ascii')
						viewer_html = f"""
						<div id='glb-viewer' style='width:100%; height:600px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'></div>
						<script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script>
						<script src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js'></script>
						<script src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'></script>
						<script>
						const container = document.getElementById('glb-viewer');
						const scene = new THREE.Scene();
						scene.background = null;
						const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
						const renderer = new THREE.WebGLRenderer({{alpha: true, antialias: true}});
						renderer.setSize(container.clientWidth, container.clientHeight);
						renderer.shadowMap.enabled = true;
						container.appendChild(renderer.domElement);
						
						// Lighting
						const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
						scene.add(ambientLight);
						const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
						dirLight.position.set(5, 10, 7.5);
						dirLight.castShadow = true;
						scene.add(dirLight);
						
						// Load GLB
						const loader = new THREE.GLTFLoader();
						const glbData = atob('{glb_b64}');
						const glbArray = new Uint8Array(glbData.length);
						for (let i = 0; i < glbData.length; i++) {{
							glbArray[i] = glbData.charCodeAt(i);
						}}
						const glbBlob = new Blob([glbArray], {{ type: 'model/gltf-binary' }});
						const glbUrl = URL.createObjectURL(glbBlob);
						
						loader.load(glbUrl, function(gltf) {{
							const model = gltf.scene;
							scene.add(model);
							
							// Center and scale model
							const box = new THREE.Box3().setFromObject(model);
							const center = box.getCenter(new THREE.Vector3());
							const size = box.getSize(new THREE.Vector3());
							const maxDim = Math.max(size.x, size.y, size.z);
							const scale = 2.0 / maxDim;
							model.scale.setScalar(scale);
							model.position.sub(center.multiplyScalar(scale));
							
							// Enable shadows
							model.traverse(function(child) {{
								if (child.isMesh) {{
									child.castShadow = true;
									child.receiveShadow = true;
								}}
							}});
						}}, undefined, function(error) {{
							console.error('Error loading GLB:', error);
						}});
						
						camera.position.set(0, 1, 3);
						camera.lookAt(0, 0, 0);
						
						// Controls
						const controls = new THREE.OrbitControls(camera, renderer.domElement);
						controls.enableDamping = true;
						controls.dampingFactor = 0.05;
						controls.minDistance = 1;
						controls.maxDistance = 10;
						
						// Animate
						function animate() {{
							requestAnimationFrame(animate);
							controls.update();
							renderer.render(scene, camera);
						}}
						animate();
						
						// Responsive
						window.addEventListener('resize', function() {{
							const w = container.clientWidth;
							const h = container.clientHeight;
							renderer.setSize(w, h);
							camera.aspect = w / h;
							camera.updateProjectionMatrix();
						}});
						</script>
						"""
						html(viewer_html, height=620, scrolling=False)
						
					except Exception as e:
						st.warning(f'Draping completed but could not load result: {e}')
				else:
					st.error(f'Draping failed: {result.get("error", "Unknown error")}')
					
		except ImportError:
			st.error('Physics-based draping requires: pip install pygltflib')
		except Exception as e:
			st.error(f'Draping error: {e}')
			import traceback
			st.code(traceback.format_exc())

	# Fit evaluation
	show_visualization = False
	if avatar_mesh is not None and garment_mesh is not None and avatar_measurements is not None and garment_dims is not None:
		fit_percentage, fit_details = compute_fit_percentage(avatar_measurements, garment_dims)
		st.metric("Fit Percentage", f"{fit_percentage:.2f}%")
		st.write("Fit Details:", fit_details)
		show_visualization = True

	# 3D Visualization (realistic view)
	if show_visualization:
		st.subheader("👔 3D Virtual Try-On")
		
		st.info("💡 **Rendering Mode**: Current view shows 3D geometry with PBR materials. For cartoon-style rendering with hair, facial features, and detailed textures (like the reference image), consider integrating texture generation models or pre-made texture assets.")
		
		# Realistic 3D viewer with avatar + garment
		def render_realistic_tryon(avatar_mesh, garment_mesh, garment_color='#F5F5DC'):
			"""Create a realistic 3D viewer with textured avatar and garment using three.js"""
			import json, base64
			from streamlit.components.v1 import html
			
			# Prepare avatar mesh data
			if hasattr(avatar_mesh, 'vertices'):
				avatar_verts = avatar_mesh.vertices
				avatar_faces = avatar_mesh.faces if hasattr(avatar_mesh, 'faces') else []
			elif isinstance(avatar_mesh, tuple) and len(avatar_mesh) >= 2:
				avatar_verts = avatar_mesh[0]
				avatar_faces = avatar_mesh[1]
			else:
				avatar_verts = np.array(avatar_mesh) if not isinstance(avatar_mesh, np.ndarray) else avatar_mesh
				avatar_faces = []
			
			avatar_data = {
				'vertices': avatar_verts.tolist() if isinstance(avatar_verts, np.ndarray) else avatar_verts,
				'faces': avatar_faces.tolist() if isinstance(avatar_faces, np.ndarray) else (avatar_faces if isinstance(avatar_faces, list) else [])
			}
			
			# Prepare garment mesh data
			if hasattr(garment_mesh, 'vertices'):
				garment_verts = garment_mesh.vertices
				garment_faces = garment_mesh.faces if hasattr(garment_mesh, 'faces') else []
			elif isinstance(garment_mesh, tuple) and len(garment_mesh) >= 2:
				garment_verts = garment_mesh[0]
				garment_faces = garment_mesh[1]
			else:
				garment_verts = np.array(garment_mesh) if not isinstance(garment_mesh, np.ndarray) else garment_mesh
				garment_faces = []
			
			garment_data = {
				'vertices': garment_verts.tolist() if isinstance(garment_verts, np.ndarray) else garment_verts,
				'faces': garment_faces.tolist() if isinstance(garment_faces, np.ndarray) else (garment_faces if isinstance(garment_faces, list) else [])
			}
			
			# Debug info
			st.write(f"🔍 Debug: Avatar has {len(avatar_data['vertices'])} vertices, {len(avatar_data['faces'])} faces")
			st.write(f"🔍 Debug: Garment has {len(garment_data['vertices'])} vertices, {len(garment_data['faces'])} faces")
			
			viewer_html = f"""
			<div id='realistic-viewer' style='width:100%; height:700px; background: linear-gradient(to bottom, #e0e0e0 0%, #ffffff 100%); border-radius: 10px;'></div>
			<script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script>
			<script src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'></script>
			<script>
			const avatarData = {json.dumps(avatar_data)};
			const garmentData = {json.dumps(garment_data)};
			
			const container = document.getElementById('realistic-viewer');
			const scene = new THREE.Scene();
			scene.background = new THREE.Color(0xf0f0f0);
			
			const camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
			const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
			renderer.setSize(container.clientWidth, container.clientHeight);
			renderer.shadowMap.enabled = true;
			renderer.shadowMap.type = THREE.PCFSoftShadowMap;
			renderer.outputEncoding = THREE.sRGBEncoding;
			renderer.toneMapping = THREE.ACESFilmicToneMapping;
			renderer.toneMappingExposure = 1.2;
			container.appendChild(renderer.domElement);
			
			// Enhanced studio lighting for realistic rendering
			const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
			scene.add(ambientLight);
			
			const keyLight = new THREE.DirectionalLight(0xffffff, 1.0);
			keyLight.position.set(5, 10, 7);
			keyLight.castShadow = true;
			keyLight.shadow.mapSize.width = 2048;
			keyLight.shadow.mapSize.height = 2048;
			keyLight.shadow.camera.near = 0.5;
			keyLight.shadow.camera.far = 50;
			scene.add(keyLight);
			
			const fillLight = new THREE.DirectionalLight(0xfff5e6, 0.4);
			fillLight.position.set(-5, 5, -5);
			scene.add(fillLight);
			
			const backLight = new THREE.DirectionalLight(0xe6f2ff, 0.5);
			backLight.position.set(0, 5, -10);
			scene.add(backLight);
			
			const rimLight = new THREE.DirectionalLight(0xffffff, 0.3);
			rimLight.position.set(0, 3, 5);
			scene.add(rimLight);
			
			// Ground plane with shadow
			const groundGeometry = new THREE.PlaneGeometry(20, 20);
			const groundMaterial = new THREE.ShadowMaterial({{ opacity: 0.3 }});
			const ground = new THREE.Mesh(groundGeometry, groundMaterial);
			ground.rotation.x = -Math.PI / 2;
			ground.position.y = -1;
			ground.receiveShadow = true;
			scene.add(ground);
			
			// Create avatar mesh with realistic skin material
			const avatarGeometry = new THREE.BufferGeometry();
			const avatarVertices = new Float32Array(avatarData.vertices.flat());
			avatarGeometry.setAttribute('position', new THREE.BufferAttribute(avatarVertices, 3));
			
			if (avatarData.faces && avatarData.faces.length > 0) {{
				const avatarIndices = new Uint32Array(avatarData.faces.flat());
				avatarGeometry.setIndex(new THREE.BufferAttribute(avatarIndices, 1));
			}}
			avatarGeometry.computeVertexNormals();
			
			// Create procedural skin texture
			const skinCanvas = document.createElement('canvas');
			skinCanvas.width = 256;
			skinCanvas.height = 256;
			const skinCtx = skinCanvas.getContext('2d');
			
			// Base skin color
			skinCtx.fillStyle = '#ffdbac';
			skinCtx.fillRect(0, 0, 256, 256);
			
			// Add subtle skin texture variation
			const skinImageData = skinCtx.getImageData(0, 0, 256, 256);
			for (let i = 0; i < skinImageData.data.length; i += 4) {{
				const noise = (Math.random() - 0.5) * 8;
				skinImageData.data[i] = Math.max(0, Math.min(255, 255 + noise));
				skinImageData.data[i+1] = Math.max(0, Math.min(255, 219 + noise));
				skinImageData.data[i+2] = Math.max(0, Math.min(255, 172 + noise));
			}}
			skinCtx.putImageData(skinImageData, 0, 0);
			
			const skinTexture = new THREE.CanvasTexture(skinCanvas);
			skinTexture.wrapS = THREE.RepeatWrapping;
			skinTexture.wrapT = THREE.RepeatWrapping;
			
			// Realistic skin material with enhanced properties
			const skinMaterial = new THREE.MeshStandardMaterial({{
				color: 0xffdbac,
				map: skinTexture,
				roughness: 0.4,
				metalness: 0.0,
				side: THREE.FrontSide,
				flatShading: false,
				emissive: 0xffdbac,
				emissiveIntensity: 0.05
			}});
			
			const avatarMesh = new THREE.Mesh(avatarGeometry, skinMaterial);
			avatarMesh.castShadow = true;
			avatarMesh.receiveShadow = true;
			scene.add(avatarMesh);
			
			// Create garment mesh with fabric material
			const garmentGeometry = new THREE.BufferGeometry();
			const garmentVertices = new Float32Array(garmentData.vertices.flat());
			garmentGeometry.setAttribute('position', new THREE.BufferAttribute(garmentVertices, 3));
			
			if (garmentData.faces && garmentData.faces.length > 0) {{
				const garmentIndices = new Uint32Array(garmentData.faces.flat());
				garmentGeometry.setIndex(new THREE.BufferAttribute(garmentIndices, 1));
			}}
			garmentGeometry.computeVertexNormals();
			
			// Realistic fabric material with texture-like appearance
			const fabricMaterial = new THREE.MeshStandardMaterial({{
				color: '{garment_color}',
				roughness: 0.9,
				metalness: 0.0,
				side: THREE.DoubleSide,
				flatShading: false,
				emissive: new THREE.Color('{garment_color}').multiplyScalar(0.02),
				emissiveIntensity: 1.0
			}});
			
			// Create procedural fabric texture
			const canvas = document.createElement('canvas');
			canvas.width = 512;
			canvas.height = 512;
			const ctx = canvas.getContext('2d');
			
			// Parse garment color
			const tempColor = new THREE.Color('{garment_color}');
			const r = Math.floor(tempColor.r * 255);
			const g = Math.floor(tempColor.g * 255);
			const b = Math.floor(tempColor.b * 255);
			
			// Base color
			ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
			ctx.fillRect(0, 0, 512, 512);
			
			// Add fabric weave pattern
			ctx.strokeStyle = `rgba(${{r*0.8}},${{g*0.8}},${{b*0.8}}, 0.3)`;
			ctx.lineWidth = 1;
			for (let i = 0; i < 512; i += 4) {{
				ctx.beginPath();
				ctx.moveTo(i, 0);
				ctx.lineTo(i, 512);
				ctx.stroke();
				ctx.beginPath();
				ctx.moveTo(0, i);
				ctx.lineTo(512, i);
				ctx.stroke();
			}}
			
			// Add subtle noise for fabric texture
			const imageData = ctx.getImageData(0, 0, 512, 512);
			for (let i = 0; i < imageData.data.length; i += 4) {{
				const noise = (Math.random() - 0.5) * 10;
				imageData.data[i] = Math.max(0, Math.min(255, imageData.data[i] + noise));
				imageData.data[i+1] = Math.max(0, Math.min(255, imageData.data[i+1] + noise));
				imageData.data[i+2] = Math.max(0, Math.min(255, imageData.data[i+2] + noise));
			}}
			ctx.putImageData(imageData, 0, 0);
			
			// Create texture from canvas
			const fabricTexture = new THREE.CanvasTexture(canvas);
			fabricTexture.wrapS = THREE.RepeatWrapping;
			fabricTexture.wrapT = THREE.RepeatWrapping;
			fabricTexture.repeat.set(2, 2);
			
			// Update fabric material with texture and transparency
			fabricMaterial.map = fabricTexture;
			fabricMaterial.transparent = true;
			fabricMaterial.opacity = 0.95;
			fabricMaterial.depthWrite = true;
			fabricMaterial.needsUpdate = true;
			
			const garmentMesh = new THREE.Mesh(garmentGeometry, fabricMaterial);
			garmentMesh.castShadow = true;
			garmentMesh.receiveShadow = true;
			
			// Add wireframe for debugging
			const wireframeGeo = new THREE.WireframeGeometry(garmentGeometry);
			const wireframeMat = new THREE.LineBasicMaterial({{ color: 0xff0000, linewidth: 2 }});
			const wireframe = new THREE.LineSegments(wireframeGeo, wireframeMat);
			wireframe.visible = false;
			garmentMesh.add(wireframe);
			
			scene.add(garmentMesh);
			
			console.log('✅ Avatar vertices:', avatarData.vertices.length);
			console.log('✅ Garment vertices:', garmentData.vertices.length);
			console.log('✅ Garment faces:', garmentData.faces.length);
			
			// Position camera
			camera.position.set(0, 0.5, 3);
			camera.lookAt(0, 0, 0);
			
			// Orbit controls
			const controls = new THREE.OrbitControls(camera, renderer.domElement);
			controls.enableDamping = true;
			controls.dampingFactor = 0.05;
			controls.minDistance = 1.5;
			controls.maxDistance = 10;
			controls.target.set(0, 0, 0);
			controls.enablePan = true;
			controls.enableZoom = true;
			
			// Animation loop
			function animate() {{
				requestAnimationFrame(animate);
				controls.update();
				renderer.render(scene, camera);
			}}
			animate();
			
			// Keyboard controls
			window.addEventListener('keydown', function(e) {{
				if (e.key === 'w' || e.key === 'W') {{
					wireframe.visible = !wireframe.visible;
					console.log('Wireframe:', wireframe.visible ? 'ON' : 'OFF');
				}}
				if (e.key === 'g' || e.key === 'G') {{
					garmentMesh.visible = !garmentMesh.visible;
					console.log('Garment:', garmentMesh.visible ? 'ON' : 'OFF');
				}}
			}});
			
			// Responsive resize
			window.addEventListener('resize', function() {{
				const w = container.clientWidth;
				const h = container.clientHeight;
				renderer.setSize(w, h);
				camera.aspect = w / h;
				camera.updateProjectionMatrix();
			}});
			
			// UI Info overlay
			const info = document.createElement('div');
			info.style.position = 'absolute';
			info.style.top = '10px';
			info.style.left = '10px';
			info.style.color = '#333';
			info.style.fontFamily = 'Arial';
			info.style.fontSize = '14px';
			info.style.background = 'rgba(255,255,255,0.8)';
			info.style.padding = '10px';
			info.style.borderRadius = '5px';
			info.innerHTML = '🖱️ Drag: Rotate | Right: Pan | Scroll: Zoom<br>⌨️ Press W: Toggle wireframe | G: Toggle garment';
			container.appendChild(info);
			</script>
			"""
			html(viewer_html, height=720, scrolling=False)
		
		# Transform garment to fit avatar before rendering
		transformed_garment_for_view = None
		try:
			import copy
			from scipy.spatial import cKDTree
			
			# Apply transformation to fit garment on avatar
			gm = copy.deepcopy(garment_mesh)
			verts = gm.vertices.astype(float) if hasattr(gm, 'vertices') else np.array(gm[0])
			
			# Compute anisotropic scales
			sx = sy = sz = 1.0
			try:
				if garment_dims and avatar_measurements:
					if 'chest' in garment_dims and 'chest' in avatar_measurements and garment_dims['chest'] > 0:
						sx = float(avatar_measurements.get('chest', 1.0)) / float(garment_dims.get('chest', 1.0))
					if 'shoulders' in garment_dims and 'shoulders' in avatar_measurements and garment_dims['shoulders'] > 0:
						sy = float(avatar_measurements.get('shoulders', 1.0)) / float(garment_dims.get('shoulders', 1.0))
					if 'hips' in garment_dims and 'hips' in avatar_measurements and garment_dims['hips'] > 0:
						sz = float(avatar_measurements.get('hips', 1.0)) / float(garment_dims.get('hips', 1.0))
			except Exception:
				sx = sy = sz = 1.05  # Default slight scale up
			
			# Prevent extreme scaling and ensure garment is slightly larger
			sx = max(0.9, min(1.3, sx * 1.08))  # Add 8% extra scale
			sy = max(0.9, min(1.3, sy * 1.08))
			sz = max(0.9, min(1.3, sz * 1.08))
			
			# Apply scaling
			centroid = verts.mean(axis=0)
			scale_mat = np.array([sx, sy, sz], dtype=float)
			verts = (verts - centroid) * scale_mat + centroid
			
			# Translate to avatar position
			avatar_verts = avatar_mesh.vertices if hasattr(avatar_mesh, 'vertices') else np.array(avatar_mesh[0])
			avatar_centroid = avatar_verts.mean(axis=0)
			garment_centroid = verts.mean(axis=0)
			translation = avatar_centroid - garment_centroid
			verts = verts + translation
			
			# Aggressive collision push outward to make garment visible
			try:
				avatar_kd = cKDTree(avatar_verts)
				bbox = avatar_verts.max(axis=0) - avatar_verts.min(axis=0)
				threshold = 0.05 * np.linalg.norm(bbox)  # Increased from 0.015 to 0.05
				min_offset = 0.02 * np.linalg.norm(bbox)  # Guaranteed minimum offset
				
				for idx, v in enumerate(verts):
					dist, loc = avatar_kd.query(v)
					# Always push outward if too close
					if dist < threshold:
						nearest = avatar_verts[loc]
						dir_vec = v - nearest
						norm = np.linalg.norm(dir_vec)
						if norm < 1e-6:
							# Use radial direction from avatar center
							dir_vec = v - avatar_centroid
							norm = np.linalg.norm(dir_vec)
						if norm > 1e-6:
							# Push to at least min_offset distance
							push_dist = max(threshold - dist, min_offset)
							verts[idx] = nearest + (dir_vec / norm) * (dist + push_dist)
			except Exception as e:
				st.warning(f'Collision push failed: {e}')
				pass
			
			# Additional global offset along surface normals (inflate the garment)
			try:
				# Calculate vertex normals for garment
				if hasattr(garment_mesh, 'faces'):
					faces = garment_mesh.faces
					normals = np.zeros_like(verts)
					for face in faces:
						v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
						normal = np.cross(v1 - v0, v2 - v0)
						normals[face[0]] += normal
						normals[face[1]] += normal
						normals[face[2]] += normal
					# Normalize
					norms = np.linalg.norm(normals, axis=1, keepdims=True)
					norms[norms == 0] = 1
					normals = normals / norms
					# Inflate by 2% of bbox
					inflate_amount = 0.02 * np.linalg.norm(bbox)
					verts = verts + normals * inflate_amount
			except Exception:
				pass
			
			# Create transformed garment object
			if hasattr(garment_mesh, 'vertices'):
				transformed_garment_for_view = copy.deepcopy(garment_mesh)
				transformed_garment_for_view.vertices = verts
			else:
				transformed_garment_for_view = (verts, garment_mesh[1] if isinstance(garment_mesh, tuple) else None)
				
		except Exception as e:
			st.warning(f'Could not transform garment: {e}. Using original garment.')
			transformed_garment_for_view = garment_mesh
		
		# Render realistic view
		realistic_view_success = False
		try:
			render_realistic_tryon(avatar_mesh, transformed_garment_for_view, garment_color=garment_color if 'garment_color' in locals() else '#F5DEB3')
			st.success('✨ Realistic 3D view with fitted garment')
			st.info('💡 Rotate the model to see how the garment fits from all angles')
			realistic_view_success = True
		except Exception as e:
			st.error(f'Could not render realistic view: {e}')
			import traceback
			st.code(traceback.format_exc())
			st.info('Falling back to basic 3D visualization...')
		
		# Define helper functions for fallback plotly view
		def plot_dummy_mesh(fig, mesh, color, name):
			if isinstance(mesh, np.ndarray) and mesh.shape[1] == 3:
				u = np.linspace(0, 2 * np.pi, 20)
				v = np.linspace(0, np.pi, 20)
				x = 0.5 * np.outer(np.cos(u), np.sin(v))
				y = 0.5 * np.outer(np.sin(u), np.sin(v))
				z = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
				fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, color], [1, color]], opacity=0.5, name=name, showscale=False))
			elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
				x, y, z = mesh.vertices.T
				i, j, k = mesh.faces.T
				fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=0.5, name=name))

		def transform_garment_to_avatar(garment_mesh, garment_dims, avatar_meas):
			"""Apply anisotropic scaling and simple collision push to the garment mesh.
			Returns a new mesh object with transformed vertices."""
			import copy
			from scipy.spatial import cKDTree
			gm = copy.deepcopy(garment_mesh)
			verts = gm.vertices.astype(float)
			# Compute anisotropic scales (x: chest/waist/hips approx lateral, y: depth, z: vertical)
			# We'll map: x -> chest, y -> shoulders (depth proxy), z -> height/hips
			sx = sy = sz = 1.0
			try:
				if garment_dims and avatar_meas:
					# chest/waist/hips/shoulders may be present in both dicts
					if 'chest' in garment_dims and 'chest' in avatar_meas and garment_dims['chest']>0:
						sx = float(avatar_meas.get('chest', avatar_meas.get('waist', avatar_meas.get('hips', sx)))) / float(garment_dims.get('chest', garment_dims.get('waist', garment_dims.get('hips', 1.0))))
					if 'shoulders' in garment_dims and 'shoulders' in avatar_meas and garment_dims['shoulders']>0:
						sy = float(avatar_meas.get('shoulders')) / float(garment_dims.get('shoulders'))
					if 'hips' in garment_dims and 'hips' in avatar_meas and garment_dims['hips']>0:
						sz = float(avatar_meas.get('hips')) / float(garment_dims.get('hips'))
			except Exception:
				sx = sy = sz = 1.0
			# Prevent extreme scaling
			sx = max(0.5, min(2.0, sx))
			sy = max(0.5, min(2.0, sy))
			sz = max(0.5, min(2.0, sz))
			# Apply anisotropic scaling about garment centroid
			centroid = verts.mean(axis=0)
			scale_mat = np.array([sx, sy, sz], dtype=float)
			verts = (verts - centroid) * scale_mat + centroid
			# Translate garment centroid to avatar centroid
			try:
				avatar_centroid = avatar_mesh.vertices.mean(axis=0)
				garment_centroid = verts.mean(axis=0)
				translation = avatar_centroid - garment_centroid
				verts = verts + translation
			except Exception:
				pass
			# Simple collision push: if a garment vertex is too close to avatar vertex, push it outward
			try:
				avatar_kd = cKDTree(avatar_mesh.vertices)
				# set threshold as a small fraction of avatar bbox diagonal
				bbox = avatar_mesh.vertices.max(axis=0) - avatar_mesh.vertices.min(axis=0)
				threshold = 0.02 * np.linalg.norm(bbox)
				for idx, v in enumerate(verts):
					dist, loc = avatar_kd.query(v)
					if dist < threshold and dist>0:
						nearest = avatar_mesh.vertices[loc]
						dir_vec = v - nearest
						if np.linalg.norm(dir_vec) < 1e-6:
							# fallback: push along (v - avatar_centroid)
							dir_vec = v - avatar_mesh.vertices.mean(axis=0)
							if np.linalg.norm(dir_vec) < 1e-6:
								dir_vec = np.array([0.0, 0.0, 1.0])
						unit = dir_vec / np.linalg.norm(dir_vec)
						verts[idx] = nearest + unit * threshold
			except Exception:
				pass
			gm.vertices = verts
			return gm

		def export_mesh_obj(mesh, path):
			"""Write a simple OBJ file from mesh with vertices and faces."""
			with open(path, 'w') as f:
				for v in mesh.vertices:
					f.write(f"v {v[0]} {v[1]} {v[2]}\n")
				for face in mesh.faces:
					# OBJ is 1-indexed
					f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

		# Only use fallback plotly visualization if realistic view failed
		if not realistic_view_success:
			show_fitted = st.sidebar.checkbox('Show fitted garment overlay', value=True)
			fig = go.Figure()
			
			# Plot avatar first
			plot_dummy_mesh(fig, avatar_mesh, 'lightblue', 'Avatar')

		# If the garment mesh contains UVs/faces and an SVG fixture is available, render a textured preview
		def render_textured_mesh_in_browser(mesh, texture_bytes, height=400):
			"""Embed a three.js viewer that constructs geometry from position/index/uv arrays and applies a texture."""
			import json, base64
			from streamlit.components.v1 import html
			verts = mesh.vertices.tolist()
			faces = mesh.faces.tolist() if hasattr(mesh, 'faces') and getattr(mesh, 'faces') is not None else []
			uvs = mesh.uvs.tolist() if hasattr(mesh, 'uvs') and getattr(mesh, 'uvs') is not None else []
			img_b64 = base64.b64encode(texture_bytes).decode('ascii')
			# Choose mime based on SVG or PNG
			mime = 'image/svg+xml' if texture_bytes.strip().startswith(b'<') else 'image/png'
			data = {
				'verts': verts,
				'faces': faces,
				'uvs': uvs,
				'texture_b64': img_b64,
				'mime': mime,
			}
			js_data = json.dumps(data)
			tmpl = """
			<div id='viewer' style='width:100%; height:HEIGHT_PXpx;'></div>
			<script src='https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'></script>
			<script src='https://threejs.org/examples/js/controls/OrbitControls.js'></script>
			<script>
			const data = DATA_PLACEHOLDER;
			const container = document.getElementById('viewer');
			const scene = new THREE.Scene();
			const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
			const renderer = new THREE.WebGLRenderer({alpha:true, antialias:true});
			renderer.setSize(container.clientWidth, container.clientHeight);
			container.appendChild(renderer.domElement);
			const geometry = new THREE.BufferGeometry();
			const positions = new Float32Array(data.verts.flat());
			geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
			if (data.uvs && data.uvs.length){
				const uvarr = new Float32Array(data.uvs.flat());
				geometry.setAttribute('uv', new THREE.BufferAttribute(uvarr, 2));
			}
			if (data.faces && data.faces.length){
				const indices = new Uint32Array(data.faces.flat());
				geometry.setIndex(new THREE.BufferAttribute(indices, 1));
			}
			geometry.computeVertexNormals();
			const img = new Image();
			img.onload = function(){
				const tex = new THREE.Texture(img);
				tex.needsUpdate = true;
				const mat = new THREE.MeshStandardMaterial({map: tex, side: THREE.DoubleSide});
				const mesh = new THREE.Mesh(geometry, mat);
				scene.add(mesh);
				const light = new THREE.DirectionalLight(0xffffff, 1);
				light.position.set(1,1,1);
				scene.add(light);
				camera.position.set(0, -1.5, 1.0);
				camera.lookAt(new THREE.Vector3(0,0,0));

				// Orbit controls
				const controls = new THREE.OrbitControls( camera, renderer.domElement );
				controls.enableDamping = true;
				controls.dampingFactor = 0.1;
				controls.enablePan = true;
				controls.enableZoom = true;

				// responsive resize
				window.addEventListener('resize', function(){
					const w = container.clientWidth;
					const h = container.clientHeight;
					renderer.setSize(w, h);
					camera.aspect = w / h;
					camera.updateProjectionMatrix();
				});

				function animate(){
					requestAnimationFrame(animate);
					controls.update();
					renderer.render(scene, camera);
				}
				animate();
			};
			img.src = 'data:' + data.mime + ';base64,' + data.texture_b64;
			</script>
			"""
			html(tmpl.replace('DATA_PLACEHOLDER', js_data).replace('HEIGHT_PX', str(height)), height=height, scrolling=False)

			if hasattr(garment_mesh, 'uvs') and getattr(garment_mesh, 'uvs') is not None and garment_meta is not None:
				# try to find a fixture or use material image if product provided
				svg_path = os.path.join(project_root, 'src', 'garments', 'fixtures', f"{template_choice}.svg")
				if os.path.exists(svg_path):
					with open(svg_path, 'rb') as f:
						tex_bytes = f.read()
					try:
						render_textured_mesh_in_browser(garment_mesh, tex_bytes, height=360)
					except Exception:
						pass

			transformed_garment = None
			if garment_mesh is not None and avatar_mesh is not None and show_fitted:
				try:
					transformed_garment = transform_garment_to_avatar(garment_mesh, garment_dims, avatar_measurements)
				except Exception:
					transformed_garment = garment_mesh
			# Depending on toggle, plot original or transformed garment
			if show_fitted and transformed_garment is not None:
				plot_dummy_mesh(fig, transformed_garment, 'red', 'Garment (fitted)')
			else:
				plot_dummy_mesh(fig, garment_mesh, 'red', 'Garment')

			fig.update_layout(scene=dict(aspectmode='data'), width=700, height=700)

			# Do NOT save visualization files to disk automatically; show in-app only
			# (User requested the output be visible only in Streamlit and not written
			# directly to the repository output folder.)
			st.info('Visualization is shown in-app only; automatic disk saving is disabled.')

			# Provide an in-memory download for the fitted garment OBJ (no disk writes)
			if transformed_garment is not None:
				def mesh_to_obj_bytes(mesh):
					"""Serialize mesh to OBJ bytes.
					Supports:
					- objects with .vertices and .faces attributes (numpy arrays)
					- (vertices, faces) tuples or lists
					- plain numpy array of vertices (no faces)
					"""
					lines = []
					# Normalize to numpy arrays
					if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
						verts = np.asarray(mesh.vertices)
						faces = np.asarray(mesh.faces)
					elif isinstance(mesh, (list, tuple)) and len(mesh) >= 1:
						verts = np.asarray(mesh[0])
						faces = np.asarray(mesh[1]) if len(mesh) > 1 else None
					elif isinstance(mesh, np.ndarray):
						verts = mesh
						faces = None
					else:
						raise ValueError('Unsupported mesh format for OBJ export')

					# vertices
					for v in verts:
						x, y, z = float(v[0]), float(v[1]), float(v[2])
						lines.append(f"v {x} {y} {z}")

					# faces (if available)
					if faces is not None and faces.size != 0:
						faces = np.asarray(faces)
						if faces.ndim == 2 and faces.shape[1] >= 3:
							for face in faces:
								# OBJ uses 1-based indexing
								i = [int(idx) + 1 for idx in face[:3]]
								lines.append(f"f {i[0]} {i[1]} {i[2]}")
						# else: ignore non-triangular faces for now

					obj_str = "\n".join(lines) + "\n"
					return obj_str.encode('utf-8')

				# Allow user to choose a filename for download
				default_name = f"garment_fitted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
				filename = st.text_input('Download filename', value=default_name)
				try:
					obj_bytes = mesh_to_obj_bytes(transformed_garment)
					st.download_button(
						label='Download fitted garment (OBJ)',
						data=obj_bytes,
						file_name=filename,
						mime='text/plain'
					)
					# Inform the user that the download has been prepared
					st.success(f'Prepared download: {filename}')
				except Exception as e:
					st.warning(f'Could not prepare download for fitted garment: {e}')			# Second automatic save removed — keep a single in-app display
			# (No disk writes.)
			st.plotly_chart(fig)


if __name__ == "__main__":
	main()

