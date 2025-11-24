@echo off
REM Usage: run_enable_io_scene_obj.bat "C:\Path\to\blender.exe" [output_json]
SET BLENDER_EXE=%1
IF "%BLENDER_EXE%"=="" (
  echo Please provide the path to blender.exe as first argument.
  echo Example: run_enable_io_scene_obj.bat "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" result.json
  exit /b 2
)
SET OUT_JSON=%2
IF "%OUT_JSON%"=="" (
  SET OUT_JSON=%TEMP%\blender_enable_result.json
)

REM Determine script path relative to this batch file
SET SCRIPT_DIR=%~dp0
SET SCRIPT_PATH=%SCRIPT_DIR%blender_enable_io_scene_obj.py

echo Running Blender to enable io_scene_obj...
"%BLENDER_EXE%" --background --python "%SCRIPT_PATH%" -- "%OUT_JSON%"
EXIT /b %ERRORLEVEL%
