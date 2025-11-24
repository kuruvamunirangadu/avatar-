"""
Blender-side script to enable the io_scene_obj addon.
Run with:
  "C:\Path\to\blender.exe" --background --python blender_enable_io_scene_obj.py -- <output_json_path>

It will try to enable the addon if present. If the addon is missing, it will print instructions and exit with code 2.
It writes a JSON object to the provided output path or prints to stdout if no path provided.
"""

import sys
import json
import os

try:
    import bpy
    import addon_utils
except Exception as e:
    # Not running inside Blender
    print(json.dumps({"status": "error", "reason": "not_running_in_blender", "error": str(e)}))
    sys.exit(3)

out_path = None
if "--" in sys.argv:
    try:
        idx = sys.argv.index("--")
        # args after -- are available to script
        script_args = sys.argv[idx+1:]
        if len(script_args) >= 1:
            out_path = script_args[0]
    except ValueError:
        script_args = []
else:
    script_args = []

result = {
    "status": "unknown",
    "addon_name": "io_scene_obj",
    "details": None,
}

addon_name = 'io_scene_obj'
try:
    # Check known addons
    available = any((mod for mod in addon_utils.modules() if getattr(mod, '__name__', '').startswith(addon_name)))
except Exception:
    available = False

# A more robust check: ask addon_utils if the module is available to register
try:
    found = addon_utils.check(addon_name)
    # addon_utils.check returns (is_enabled, is_loaded) for the named module if present
    # But it raises if the module not found, so wrap
    present = True
except Exception as e:
    present = False

try:
    if present:
        # Attempt to enable
        try:
            addon_utils.enable(addon_name, default_set=True, persistent=True)
            # Save user preferences to persist the addon enable
            try:
                bpy.ops.wm.save_userpref()
            except:
                pass  # May fail in background mode
            result['status'] = 'enabled'
            result['details'] = 'addon enabled and preferences saved'
        except Exception as e:
            # try enabling by full module name
            try:
                addon_utils.enable(addon_name)
                try:
                    bpy.ops.wm.save_userpref()
                except:
                    pass
                result['status'] = 'enabled'
                result['details'] = 'addon enabled (fallback, preferences saved)'
            except Exception as e2:
                result['status'] = 'error'
                result['details'] = f'enable_failed: {e2!r}'
    else:
        # Not present — provide helpful diagnostic info
        script_paths = bpy.utils.script_paths()
        user_scripts = bpy.utils.user_resource('SCRIPTS')
        # Typical addon directories
        addon_dirs = [os.path.join(p, 'addons') for p in script_paths if p]
        addon_dirs = list(dict.fromkeys(addon_dirs))
        result['status'] = 'missing'
        result['details'] = {
            'message': 'io_scene_obj module not found in Blender scripts/addons paths',
            'script_paths': script_paths,
            'user_scripts': user_scripts,
            'addon_dirs': addon_dirs,
            'manual_instructions': (
                'Download the "io_scene_obj" addon (it is normally bundled with Blender). '
                'Copy the folder (named io_scene_obj) into one of the addon dirs above (e.g., <user_scripts>/addons) ' 
                'or install via Blender Preferences > Add-ons > Install... with a zip containing the addon. '
                'Then re-run this script to enable it automatically.'
            )
        }
except Exception as exc:
    result['status'] = 'error'
    result['details'] = str(exc)

# Write result
if out_path:
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(json.dumps({"wrote": out_path}))
    except Exception as e:
        print(json.dumps({"status": "error_writing", "error": str(e)}))
        sys.exit(4)
else:
    print(json.dumps(result, indent=2))

# Exit codes: 0 default, 2 missing addon, 3 not running in blender, 4 write error
if result.get('status') == 'missing':
    sys.exit(2)
elif result.get('status') == 'error':
    sys.exit(1)
else:
    sys.exit(0)
