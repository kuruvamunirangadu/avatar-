# Utility: Load OBJ mesh (vertices, faces) from SMPLify-X output
def load_obj_mesh(obj_path):
	"""
	Load mesh vertices and faces from an OBJ file.
	Returns: mesh object with .vertices and .faces (numpy arrays)
	"""
	import numpy as np
	vertices = []
	faces = []
	with open(obj_path, 'r') as f:
		for line in f:
			if line.startswith('v '):
				vertices.append([float(x) for x in line.strip().split()[1:4]])
			elif line.startswith('f '):
				# OBJ is 1-indexed
				face = [int(x.split('/')[0]) - 1 for x in line.strip().split()[1:4]]
				faces.append(face)
	mesh = type('Mesh', (), {})()
	mesh.vertices = np.array(vertices)
	mesh.faces = np.array(faces)
	return mesh
# Export OpenPose keypoints to JSON for SMPLify-X
def export_openpose_json(openpose_keypoints, output_path, image_path):
	"""
	Save OpenPose BODY_25 keypoints to JSON in SMPLify-X compatible format.
	Args:
		openpose_keypoints: numpy array (25, 3)
		output_path: path to save JSON
		image_path: path to the corresponding image
	"""
	import json
	import os
	data = {
		"people": [
			{
				"pose_keypoints_2d": openpose_keypoints.flatten().tolist(),
				"face_keypoints_2d": [],
				"hand_left_keypoints_2d": [],
				"hand_right_keypoints_2d": []
			}
		]
	}
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, 'w') as f:
		json.dump(data, f)

# Example usage (in your app or notebook):
# from src.preprocessing.preprocess import mediapipe_to_openpose, export_openpose_json
# openpose_kps = mediapipe_to_openpose(keypoints, image_width, image_height)
# export_openpose_json(openpose_kps, 'output/keypoints.json', 'output/image.png')

# --- SMPLify-X Fitting Instructions ---
# 1. Place your image and exported keypoints JSON in a folder (e.g., output/)
# 2. Run SMPLify-X with:
#    python smplifyx/main.py --config cfg_files/fit_smplx.yaml --data_folder output/ --output_folder output/results --visualize=True --model_folder <path_to_smplx_models> --vposer_ckpt <path_to_vposer_ckpt>
# 3. The fitted mesh will be in output/results/<image_name>/meshes/
# Utility: Convert MediaPipe keypoints to OpenPose BODY_25 format
def mediapipe_to_openpose(keypoints, image_width, image_height):
	"""
	Convert MediaPipe keypoints (normalized) to OpenPose BODY_25 format (COCO order).
	Args:
		keypoints: list of (x, y, z, visibility) in normalized coordinates
		image_width, image_height: original image size
	Returns:
		openpose_keypoints: numpy array shape (25, 3) with (x, y, confidence)
	"""
	import numpy as np
	# MediaPipe Pose has 33 keypoints, OpenPose BODY_25 has 25
	# Map MediaPipe indices to OpenPose BODY_25 (COCO) indices
	# See: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
	# and https://google.github.io/mediapipe/solutions/pose.html#pose-landmark-model-blazepose-ghum-3d
	mp_to_op25 = [0, 15, 16, 17, 18, 11, 12, 23, 24, 25, 26, 27, 28, 5, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	openpose_keypoints = np.zeros((25, 3), dtype=np.float32)
	for op_idx, mp_idx in enumerate(mp_to_op25):
		if mp_idx < len(keypoints):
			x, y, z, v = keypoints[mp_idx]
			openpose_keypoints[op_idx, 0] = x * image_width
			openpose_keypoints[op_idx, 1] = y * image_height
			openpose_keypoints[op_idx, 2] = v
		else:
			openpose_keypoints[op_idx] = 0
	return openpose_keypoints

"""
Preprocessing module: background removal, keypoint detection, and data utilities
Dependencies: pip install mediapipe rembg opencv-python
"""
import numpy as np
from PIL import Image
import scipy.io
import os
import glob
import gc

# For background removal and keypoint detection
import cv2
try:
	from rembg import remove as rembg_remove
except ImportError:
	rembg_remove = None

def remove_background(image):
	"""Remove background from a PIL image using rembg."""
	try:
		 from rembg import remove as rembg_remove
	except ImportError:
		 raise ImportError("rembg is not installed. Please install with 'pip install rembg'.")
	return Image.fromarray(rembg_remove(np.array(image)))

try:
	import mediapipe as mp
except ImportError:
	mp = None


def detect_keypoints(image):
	"""Detect body keypoints using mediapipe."""
	if mp is None:
		# Don't raise here so the demo UI can still run without mediapipe installed.
		# Return an empty list to indicate no keypoints detected and let the
		# higher-level app show a helpful message to the user.
		print("WARN: mediapipe not available - keypoint detection disabled")
		return []
	mp_pose = mp.solutions.pose
	pose = mp_pose.Pose(static_image_mode=True)
	img_rgb = np.array(image.convert('RGB'))
	results = pose.process(img_rgb)
	keypoints = []
	if results.pose_landmarks:
		for lm in results.pose_landmarks.landmark:
			keypoints.append((lm.x, lm.y, lm.z, lm.visibility))
	return keypoints

# Advanced garment preprocessing
def segment_garment(image):
	"""Segment garment from image using color thresholding or ML model (placeholder)."""
	import cv2
	img_np = np.array(image.convert('RGB'))
	gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
	_, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
	segmented = cv2.bitwise_and(img_np, img_np, mask=mask)
	return Image.fromarray(segmented)

def detect_garment_edges(image):
	"""Detect garment edges using Canny edge detection (placeholder)."""
	import cv2
	img_np = np.array(image.convert('RGB'))
	gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
	edges = cv2.Canny(gray, 100, 200)
	return Image.fromarray(edges)

def read_samples(cameras_dir, read_f, num_cameras=4):
	cameras = []
	for camera_id in range(1, num_cameras+1):
		camera_path = os.path.join(cameras_dir, f'camera{camera_id:02d}')
		camera_data = read_f(camera_path)
		if camera_data is None:
			return None
		cameras.append(np.asarray(camera_data))
		gc.collect()
	return np.asarray(cameras)

def read_rgbs(files_dir):
	rgbs_path = os.path.join(files_dir, '*', '*.jpg')
	files = glob.glob(rgbs_path)
	if files:
		return [np.asarray(Image.open(file)) for file in sorted(files)]

def read_depths(files_dir):
	file = os.path.join(files_dir, 'depth.mat')
	if os.path.exists(file):
		depth = scipy.io.loadmat(file)
		return [v for k, v in sorted(depth.items()) if k.startswith('depth')]

def read_normals(files_dir):
	file = os.path.join(files_dir, 'normals.mat')
	if os.path.exists(file):
		normals = scipy.io.loadmat(file)
		return [v/2+0.5 for k, v in sorted(normals.items()) if k.startswith('normals')]

def read_optical_flow(files_dir):
	file = os.path.join(files_dir, 'optical_flow.mat')
	if os.path.exists(file):
		flow = scipy.io.loadmat(file)
		return [v for k, v in sorted(flow.items()) if k.startswith('gtflow')]

def read_segmentation(files_dir):
	segmentation_path = os.path.join(files_dir,'*.png')
	files = glob.glob(segmentation_path)
	if files:
		return [np.asarray(Image.open(file).convert('RGB')) for file in sorted(files)]

def read_skeleton(files_dir):
	skeleton_path = os.path.join(files_dir,'*.txt')
	files = glob.glob(skeleton_path)
	if files:
		files.sort()
		if os.path.basename(files[0]) == "0000.txt":
			files = files[1:]
		return [np.loadtxt(file) for file in files]
