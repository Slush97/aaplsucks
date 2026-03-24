"""Blender CLI script: fix UVs on an existing GLB (flip V axis, fix alpha/double-sided).

Usage:
    blender --background --python fix_glb_uvs.py -- input.glb output.glb
    blender --background --python fix_glb_uvs.py -- --no-flip-uv input.glb output.glb
"""
import sys
import bpy

argv = sys.argv
args = argv[argv.index("--") + 1:]

flip_uv = True
if "--no-flip-uv" in args:
    flip_uv = False
    args.remove("--no-flip-uv")

src = args[0]
dst = args[1]

# Clear default scene.
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import GLB.
bpy.ops.import_scene.gltf(filepath=src)
print(f"Imported {src}")

# Force-load all packed images so they survive re-export in background mode.
for img in bpy.data.images:
    if img.packed_file is not None:
        img.pack()
    # Reload to ensure pixel data is available.
    img.reload()
    print(f"  image: {img.name} {img.size[0]}x{img.size[1]} ch={img.channels}")


def _image_has_alpha(img):
    if img.channels < 4:
        return False
    try:
        pixels = img.pixels[:]
    except Exception:
        return False
    total = len(pixels) // 4
    if total == 0:
        return False
    step = max(1, total // 1000)
    for i in range(0, total, step):
        if pixels[i * 4 + 3] < 0.99:
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

    if flip_uv and mesh.uv_layers:
        for uv_layer in mesh.uv_layers:
            for loop in uv_layer.data:
                loop.uv[1] = 1.0 - loop.uv[1]
            uv_flipped += 1

    for slot in obj.material_slots:
        mat = slot.material
        if mat is None:
            continue

        if not mat.use_backface_culling:
            backface_fixed += 1
        mat.use_backface_culling = False

        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    img = node.image
                    if img.channels == 4 and _image_has_alpha(img):
                        mat.blend_method = 'CLIP'
                        mat.alpha_threshold = 0.5
                        alpha_fixed += 1
                        break

print(f"[fix] {mesh_count} meshes, {uv_flipped} UV layers flipped, {alpha_fixed} alpha, {backface_fixed} double-sided")

bpy.ops.export_scene.gltf(
    filepath=dst,
    export_format="GLB",
    export_image_format="AUTO",
    export_keep_originals=True,
    export_all_vertex_colors=True,
    export_normals=True,
    export_tangents=True,
    export_animations=True,
    export_skins=True,
)
print(f"Exported {dst}")
