"""Blender CLI script: convert FBX to GLB with 3DS/DS texture fixes.

Usage:
    blender --background --python fbx_to_glb.py -- input.fbx output.glb
    blender --background --python fbx_to_glb.py -- --no-flip-uv input.fbx output.glb

Handles common issues when converting ripped 3DS/DS Pokemon models:
  - UV V-axis flip (3DS uses top-left origin, glTF expects bottom-left)
  - Alpha mode detection (cutout textures for eyes, fins, etc.)
  - Vertex color preservation
  - Backface culling disabled (many Pokemon models are single-sided)
"""
import sys
import bpy

argv = sys.argv
args = argv[argv.index("--") + 1:]

# Parse flags.
flip_uv = True
if "--no-flip-uv" in args:
    flip_uv = False
    args.remove("--no-flip-uv")

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

# ── Helpers ──

def _image_has_alpha(img):
    """Check if a Blender image has any non-opaque alpha pixels.

    Samples a subset of pixels to avoid being slow on large textures.
    """
    if img.channels < 4:
        return False
    try:
        pixels = img.pixels[:]
    except Exception:
        return False
    total = len(pixels) // 4
    if total == 0:
        return False
    # Sample up to 1000 evenly spaced pixels.
    step = max(1, total // 1000)
    for i in range(0, total, step):
        alpha = pixels[i * 4 + 3]
        if alpha < 0.99:
            return True
    return False

# ── Post-import fixes ──

mesh_count = 0
uv_flipped = 0
alpha_fixed = 0
backface_fixed = 0

for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    mesh_count += 1
    mesh = obj.data

    # --- UV V-axis flip ---
    # 3DS/DS models use top-left UV origin; glTF/OpenGL uses bottom-left.
    # Flip V: v = 1.0 - v
    if flip_uv and mesh.uv_layers:
        for uv_layer in mesh.uv_layers:
            for loop in uv_layer.data:
                loop.uv[1] = 1.0 - loop.uv[1]
            uv_flipped += 1

    # --- Fix materials ---
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue

        # Enable backface rendering (many 3DS models need double-sided).
        if not mat.use_backface_culling:
            backface_fixed += 1
        mat.use_backface_culling = False

        # Detect alpha textures and set blend mode.
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    img = node.image
                    # Check if image has meaningful alpha channel.
                    if img.channels == 4 and _image_has_alpha(img):
                        mat.blend_method = 'CLIP'
                        mat.alpha_threshold = 0.5
                        alpha_fixed += 1
                        break

print(f"[fix] {mesh_count} meshes processed")
if flip_uv:
    print(f"[fix] {uv_flipped} UV layers flipped (V = 1-V)")
print(f"[fix] {alpha_fixed} materials set to alpha clip")
print(f"[fix] {backface_fixed} materials set to double-sided")

# Export GLB with vertex colors and animations.
bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format="GLB",
    export_all_vertex_colors=True,
    export_normals=True,
    export_tangents=True,
    export_animations=True,
    export_skins=True,
)
print(f"Converted {src} -> {dst}")
