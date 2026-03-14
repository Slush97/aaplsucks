"""Blender CLI script: convert FBX to GLB.

Usage:
    blender --background --python fbx_to_glb.py -- input.fbx output.glb
"""
import sys
import bpy

argv = sys.argv
args = argv[argv.index("--") + 1:]
src = args[0]
dst = args[1]

# Clear default scene.
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import FBX using the low-level loader (avoids Blender 5.0 operator changes).
from io_scene_fbx import import_fbx
import bpy.types

class _FakeOp:
    """Minimal stand-in for the operator 'self' that import_fbx.load expects."""
    def report(self, level, msg):
        print(f"[{level}] {msg}")

result = import_fbx.load(_FakeOp(), bpy.context, filepath=src)
print(f"FBX import result: {result}")

# Export GLB.
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
print(f"Converted {src} -> {dst}")
