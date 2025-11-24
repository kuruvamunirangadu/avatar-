# Enabling the Blender OBJ importer (io_scene_obj)

This small helper attempts to enable the `io_scene_obj` addon inside a Blender installation. It cannot modify files outside the repo automatically from this environment; instead, it provides a Blender-side script and a Windows batch wrapper you can run locally.

Files added:

- `scripts/blender_enable_io_scene_obj.py` - Blender-side Python script. Must be executed by Blender (it imports bpy and addon_utils).
- `scripts/run_enable_io_scene_obj.bat` - Windows batch wrapper. Run it with the path to your `blender.exe` and an optional output JSON path.

How to run (Windows cmd.exe):

1. Open an Administrator cmd prompt if you plan to modify files in `C:\Program Files`.
2. From the repository root run:

```
scripts\run_enable_io_scene_obj.bat "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" "%TEMP%\\blender_enable_result.json"
```

3. The batch will run Blender in background and write a small JSON describing the result. Typical exit codes:
   - 0: addon enabled (or already enabled)
   - 1: error enabling addon
   - 2: addon missing (script prints recommended addon directories and manual instructions)
   - 3: script was not executed inside Blender (you ran with system Python accidentally)

If the script reports `missing`, follow the `manual_instructions` field in the JSON or:

- Open Blender GUI > Edit > Preferences > Add-ons.
- Search for "Wavefront OBJ" or "obj". If present, enable it.
- If not present, download the `io_scene_obj` addon (it is normally bundled with Blender). You may copy the `io_scene_obj` folder into one of the addon directories printed by the script (commonly `%APPDATA%\\Blender Foundation\\Blender\\<version>\\scripts\\addons` on Windows), then enable it from Preferences.

After enabling, re-run the batch command and it should return status `enabled`.

Notes and permissions:
- Installing or copying files into `C:\Program Files` may require Administrator privileges.
- If you prefer, enable the addon via the Blender GUI and re-run the drape job; enabling via GUI sets per-user prefs and typically does not require admin rights.

If you run the batch and paste the produced JSON here, I will continue and re-run the draping job automatically (or guide next steps).