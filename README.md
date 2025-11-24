# Virtual Try-On with Fit Percentage Evaluation

## 📌 Overview
This project is a prototype system for **AI-driven virtual try-on**.  
It generates a **3D avatar from a single 2D image**, overlays selected garments with **physics-based cloth simulation**, and computes a **fit percentage score** to evaluate how well the clothing matches the user's body dimensions.

The project is designed as a research-oriented prototype for e-commerce applications, addressing the limitations of current AR/VR try-on systems that only provide visual results without accurate size-fit evaluation.

---

## 🚀 Features
- 📷 Upload a 2D user image and generate a personalized avatar.  
- ✂️ Preprocessing: background removal & keypoint detection.  
- 🧍 Body shape & pose estimation using SMPL or heuristic methods.  
- 👕 **Physics-based cloth draping** with Blender Cloth engine (collision, self-collision, multi-pose simulation).  
- 🎨 **Displacement heatmaps** baked to vertex colors for fit visualization.  
- 📊 **Fit percentage calculation** (geometric + learned metrics).  
- 🛍️ **E-commerce Integration**:  
  - A user can select a product from an e-commerce site.  
  - The system fetches product images + size chart.  
  - Combines with the user’s avatar to simulate 3D try-on.  
  - Computes how well the product fits based on both visual draping and size chart data.  
- 💻 Streamlit demo for interactive try-on.  
- 📈 Evaluation scripts for accuracy and user testing.  

---

## 📂 Project Structure


virtual-tryon/
├── data/
│ ├── raw/ # raw images, garments, SMPL models
│ └── synthetic/ # synthetic renders for training
├── notebooks/ # Jupyter experiments
├── src/
│ ├── app/ # Streamlit/Flask demo
│ │ └── app.py
│ ├── preprocessing/ # preprocessing modules
│ │ └── preprocess.py
│ ├── pose_shape/ # body pose + shape estimation
│ │ └── smpl_estimator.py
│ ├── avatar/ # avatar generation
│ │ └── generate_mesh.py
│ ├── garments/ # garment loading utilities
│ │ ├── garment_loader.py # high-level garment generation API
│ │ ├── templates.py # parametric templates (tee/shirt/jeans/dress)
│ │ ├── size_chart.py # size chart parser (CSV/HTML)
│ │ ├── materials.py # material property inference
│ │ └── pifuhd_integration.py # optional PIFuHD async jobs
│ ├── draping/ # physics-based draping system
│ │ ├── drape.py # high-level draping API
│ │ └── engine_adapters/
│ │     └── blender_adapter.py # Blender Cloth engine integration
│ ├── fit_model/ # fit evaluation metrics & model
│ │ ├── fit_net.py
│ │ └── fit_metrics.py
│ ├── evaluation/ # evaluation and testing
│ │ └── evaluate.py
│ └── utils/ # helper functions
│ └── render.py
├── scripts/ # utility scripts
│ ├── run_blender_drape.py # demo runner for draping system
│ ├── inspect_glb.py # GLB validation tool
│ ├── blender_enable_io_scene_obj.py # Blender addon helper
│ ├── run_enable_io_scene_obj.bat # Windows batch wrapper
│ └── enable_addon_README.md # addon setup instructions
├── experiments/ # trained models, checkpoints
├── requirements.txt # dependencies
├── README.md # project documentation
└── run_demo.sh # shortcut to launch demo


---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/virtual-tryon.git
   cd virtual-tryon
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional)** Install Blender for physics-based draping:
   - Download Blender 4.x from https://www.blender.org/download/
   - Set environment variable: `BLENDER_EXECUTABLE=path/to/blender.exe`
   - See [Physics-Based Draping System](#-physics-based-draping-system) section for details

▶️ Running the Demo

Start the Streamlit app:

streamlit run src/app/app.py

Note about SMPL‑X models
------------------------
The full SMPL‑X model files are not distributed with this repository due to
licensing restrictions. To enable realistic 3D avatar generation with hands
and facial expression support, download the official SMPL‑X `.npz` files
(neutral/male/female) from the SMPL‑X model provider and place them under:

```
src/models/smplx/neutral/model.npz
src/models/smplx/male/model.npz
src/models/smplx/female/model.npz
```

Alternatively you can place top-level files named `SMPLX_NEUTRAL.npz`,
`SMPLX_MALE.npz` and `SMPLX_FEMALE.npz` in `src/models/smplx/`.

Recommended package version
---------------------------
This project was validated with `smplx==0.1.28`. Add that version to your
virtual environment before running the demo:

```bash
pip install smplx==0.1.28
```


Upload a user image and select a garment to see:

Generated avatar

Draped garment visualization

Fit percentage score

(Optional) Try-on with real product images from an e-commerce site

📊 Fit Percentage Metric

The system computes fit percentage using:

Penetration metric: garment-body intersections.

Distance metric: average garment-to-body distance.

Coverage metric: overlap ratio of garment vs body region.

Learned regressor (optional): CNN-based score predictor.

Final score is a weighted combination of these factors, normalized to 0–100%.

🛍️ E-commerce Integration

This feature allows direct integration with shopping platforms:

A user can select a clothing item directly from an e-commerce site.

The system fetches the product image and size chart details.

Combines with the uploaded user image to create a personalized 3D try-on experience:

Generates the user’s avatar.

Converts product photo into a garment mesh/texture.

Drapes garment onto avatar.

Computes fit percentage using both garment size chart and avatar body measurements.

This approach reduces size mismatches and product returns while improving customer confidence.

📚 Datasets

DeepFashion
 – clothing images

3DPeople
 – 3D body meshes

3DPW
 – human pose sequences

Synthetic dataset generation with Blender

📈 Evaluation

Quantitative:

Fit score vs. ground-truth labels

Penetration rate

Inference time

Qualitative:

Visual realism

User satisfaction surveys

Ablation:

Geometry-only vs. geometry+learned models

Try-on with and without e-commerce size chart data

## 🎨 Physics-Based Draping System

The project includes a production-ready physics-based cloth simulation system built on Blender's Cloth engine. This provides realistic garment draping with collision detection, self-collision, and multi-pose simulation.

### Features

- **Collision Pipeline**: Inflated body mesh for collision detection (configurable inflation distance)
- **Self-Collision**: Prevents cloth from intersecting with itself
- **Material Parameters**: Configurable mass, tension, compression, shear, bending stiffness, simulation quality, and collision distance
- **Multi-Pose Simulation**: Sequential pose application (A-pose settle → neutral → custom poses)
- **Displacement Heatmap**: Per-vertex displacement magnitude baked to vertex colors (COLOR_0 attribute)
- **Tangent Export**: Automatic UV unwrapping and tangent generation for normal mapping support
- **GLB Export**: Industry-standard format with all attributes (POSITION, NORMAL, TANGENT, TEXCOORD_0, COLOR_0)

### Setup

1. **Install Blender 4.x or later** (tested with 4.5.3 LTS):
   - Download from https://www.blender.org/download/
   - Install to default location or custom path

2. **Set Environment Variable**:
   ```cmd
   set BLENDER_EXECUTABLE=C:\Program Files\Blender Foundation\Blender 4.5\blender.exe
   ```
   
   Or on Linux/Mac:
   ```bash
   export BLENDER_EXECUTABLE=/usr/bin/blender
   ```

3. **Verify OBJ Import Addon** (automatic in Blender 4.x):
   - The system uses `bpy.ops.wm.obj_import` (new Blender 4.x importer)
   - Falls back to legacy `bpy.ops.import_scene.obj` for Blender 3.x
   - No manual addon installation needed for standard Blender builds

### Usage

```python
from src.draping.drape import drape_garment

# Basic usage
result = drape_garment(
    avatar='path/to/avatar.obj',
    cloth='path/to/cloth.obj',
    engine='blender',  # or 'stub' for testing without Blender
    out_dir='output/'
)

# Advanced usage with custom parameters
result = drape_garment(
    avatar='avatar.obj',
    cloth='garment.obj',
    engine='blender',
    material_params={
        'mass': 0.3,           # kg (lighter = more fluid)
        'tension': 15.0,       # structural stiffness
        'compression': 15.0,   # compression resistance
        'shear': 5.0,          # shear stiffness
        'bend': 0.5,           # bending resistance
        'quality': 5,          # simulation steps (higher = more accurate)
        'collision_distance': 0.015  # min distance from body
    },
    collision_inflation=0.01,  # inflate body mesh by 1cm
    poses=[
        {'frames': 50},                    # A-pose settle (50 frames)
        {'frames': 30},                    # Neutral pose (30 frames)
        {'fbx': 'pose1.fbx', 'frames': 40} # Custom pose from FBX
    ],
    out_dir='output/'
)

# Result structure
print(result['status'])      # 'finished' or 'error'
print(result['result_glb'])  # Path to output GLB file
```

### Output Format

The exported GLB file contains:
- **Geometry**: Simulated avatar and cloth meshes
- **Normals**: Surface normals for lighting
- **Tangents**: Tangent vectors for normal mapping
- **UVs**: Texture coordinates (auto-generated via smart projection)
- **Vertex Colors**: Displacement heatmap on cloth mesh
  - Red channel: Displacement magnitude (normalized 0-1)
  - Blue channel: Inverse displacement (1-displacement)
  - Useful for visualizing cloth stretch/compression

### Interpreting Vertex Color Heatmaps

The cloth mesh contains a `COLOR_0` vertex color layer encoding displacement:
- **Red regions**: High displacement (cloth stretched or compressed significantly)
- **Blue regions**: Low displacement (cloth maintains original shape)
- Use in shaders: `color.r * maxDisplacement` to get actual displacement in meters

### Demo Script

Run a complete demo with synthetic fixtures:
```cmd
set BLENDER_EXECUTABLE=C:\Program Files\Blender Foundation\Blender 4.5\blender.exe
python scripts\run_blender_drape.py blender
```

Inspect the output GLB:
```cmd
python scripts\inspect_glb.py output\draped.glb
```

### Architecture

The system uses an adapter pattern for extensibility:
- `PhysicsEngineAdapter`: Abstract interface
- `BlenderAdapter`: Blender Cloth engine implementation
- `_StubAdapter`: Fallback for testing without Blender

To add a new physics engine, implement `PhysicsEngineAdapter` and register in `_choose_adapter()`.

### Troubleshooting

**"Blender executable not found"**:
- Set `BLENDER_EXECUTABLE` environment variable to your `blender.exe` path
- Or ensure `blender` is available on your system PATH

**OBJ import fails**:
- Blender 4.x uses the new `wm.obj_import` operator (automatic fallback included)
- Ensure your OBJ files are valid (check with Blender GUI import)

**Simulation produces unexpected results**:
- Adjust `collision_inflation` (try 0.005 to 0.02)
- Increase `material_params.quality` (5-10 for production)
- Add more frames in pose schedule for settling

**No vertex colors in output**:
- Check that cloth mesh has vertices (not just faces)
- Verify simulation completed (check Blender terminal output)

---

🔮 Future Work

Style-based recommendations (outfit suggestions).

Real-time avatar fitting in AR.

Privacy-preserving avatar generation.

Garment reconstruction (template library, size-chart, materials)
------------------------------------------------------------
We added a lightweight garment reconstruction toolkit for the demo:

- Template library: parametric templates for `tee`, `shirt`, `jeans`, and `dress` with sliders/parameters ({chest, waist, hip, length, rise, sleeve}). These are implemented in `src/garments/templates.py` and provide deterministic meshes for UI previews and fitting pipelines.
- Size-chart parser: `src/garments/size_chart.py` accepts a dict, HTML/text snippet, or path to CSV/HTML and extracts canonical fields (`chest, waist, hip, length, rise, sleeve`) using heuristics.
- Material inference: `src/garments/materials.py` infers simple material properties (bend, shear, stretch, density, thickness) from product description text or category.

These modules are intentionally lightweight and do not perform physics-based draping; they're intended to feed the existing draping/fitting pipeline and the Streamlit demo. See `src/garments/garment_loader.py` for the high-level integration point (`generate_garment_mesh`).

Optional: single-image 3D reconstruction (PIFuHD/Geo-lifting) is planned behind an R&D feature flag and is not included by default.

PIFuHD single-image reconstruction (optional)
--------------------------------------------
If you want to use single-image 3D reconstruction (PIFuHD), follow these notes:

- PIFuHD requires a GPU and large model checkpoints. It's optional and behind a feature flag in the Streamlit UI.
- You can either install a pip-style package (e.g. a `pifuhd` Python package) or clone a PIFuHD repository and set the environment variable `PIFUHD_ROOT` to the repository root. The app will try to import a `pifuhd` package first; if not present it will attempt to run an inference script found under `PIFUHD_ROOT` (e.g., `run_monocular.py` or `inference.py`).
- The CLI-mode requires Python and the PIFuHD repo's requirements installed in your environment. Example:

```cmd
git clone https://github.com/facebookresearch/pifuhd.git C:\path\to\pifuhd
set PIFUHD_ROOT=C:\path\to\pifuhd
pip install -r C:\path\to\pifuhd\requirements.txt
```

- In the Streamlit app, enable "Use PIFuHD" (R&D) in the garment reconstruction panel. The app will attempt reconstruction and either use the resulting OBJ as the garment source or fall back to templates on error.


✍️ Authors

Your Name

Collaborators / Supervisors

📄 License

This project is intended for academic and research use.
Commercial usage may require proper licensing of datasets and pretrained models.


---

✅ This version fully includes the **E-commerce product integration workflow** as a highlighted feature.  
