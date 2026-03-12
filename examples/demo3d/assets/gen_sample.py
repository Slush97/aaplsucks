#!/usr/bin/env python3
"""Generate a sample GLB file with multiple meshes and PBR materials."""

import struct
import json
import math
import os

def pack_vec3(x, y, z):
    return struct.pack('<3f', x, y, z)

def pack_vec2(u, v):
    return struct.pack('<2f', u, v)

def pack_u16(i):
    return struct.pack('<H', i)

def make_cube():
    """Unit cube centered at origin with normals and UVs."""
    # Each face has 4 verts, 6 faces = 24 verts, 36 indices
    faces = [
        # (normal, vertices CCW, UVs)
        # +Z
        ([0,0,1],  [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]),
        # -Z
        ([0,0,-1], [[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]]),
        # +X
        ([1,0,0],  [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]]),
        # -X
        ([-1,0,0], [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]]),
        # +Y
        ([0,1,0],  [[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]]),
        # -Y
        ([0,-1,0], [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]]),
    ]
    face_uvs = [[0,0],[1,0],[1,1],[0,1]]

    positions = []
    normals = []
    uvs = []
    indices = []

    for fi, (n, verts) in enumerate(faces):
        base = fi * 4
        for vi, v in enumerate(verts):
            positions.append([v[0]*0.5, v[1]*0.5, v[2]*0.5])
            normals.append(n)
            uvs.append(face_uvs[vi])
        indices.extend([base, base+1, base+2, base, base+2, base+3])

    return positions, normals, uvs, indices

def make_uv_sphere(rings=16, sectors=32):
    """UV sphere with radius 0.5."""
    positions = []
    normals = []
    uvs = []
    indices = []

    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(sectors + 1):
            theta = 2.0 * math.pi * s / sectors
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            positions.append([x * 0.5, y * 0.5, z * 0.5])
            normals.append([x, y, z])
            uvs.append([s / sectors, r / rings])

    for r in range(rings):
        for s in range(sectors):
            a = r * (sectors + 1) + s
            b = a + sectors + 1
            indices.extend([a, b, a + 1])
            indices.extend([a + 1, b, b + 1])

    return positions, normals, uvs, indices

def make_plane(size=5.0):
    """Horizontal plane at y=0."""
    h = size / 2
    positions = [[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h]]
    normals = [[0, 1, 0]] * 4
    uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]
    indices = [0, 2, 1, 0, 3, 2]
    return positions, normals, uvs, indices

def make_torus(major_r=0.35, minor_r=0.12, major_seg=32, minor_seg=16):
    """Torus centered at origin lying in the XZ plane."""
    positions = []
    normals = []
    uvs = []
    indices = []

    for i in range(major_seg + 1):
        theta = 2.0 * math.pi * i / major_seg
        ct, st = math.cos(theta), math.sin(theta)
        for j in range(minor_seg + 1):
            phi = 2.0 * math.pi * j / minor_seg
            cp, sp = math.cos(phi), math.sin(phi)
            x = (major_r + minor_r * cp) * ct
            y = minor_r * sp
            z = (major_r + minor_r * cp) * st
            nx = cp * ct
            ny = sp
            nz = cp * st
            positions.append([x, y, z])
            normals.append([nx, ny, nz])
            uvs.append([i / major_seg, j / minor_seg])

    for i in range(major_seg):
        for j in range(minor_seg):
            a = i * (minor_seg + 1) + j
            b = a + minor_seg + 1
            indices.extend([a, b, a + 1])
            indices.extend([a + 1, b, b + 1])

    return positions, normals, uvs, indices

def make_cylinder(radius=0.3, height=1.0, segments=32):
    """Cylinder along Y axis."""
    positions = []
    normals = []
    uvs = []
    indices = []
    half = height / 2

    # Side
    for i in range(segments + 1):
        theta = 2.0 * math.pi * i / segments
        c, s = math.cos(theta), math.sin(theta)
        x, z = radius * c, radius * s
        positions.append([x, -half, z])
        normals.append([c, 0, s])
        uvs.append([i / segments, 0])
        positions.append([x, half, z])
        normals.append([c, 0, s])
        uvs.append([i / segments, 1])

    for i in range(segments):
        a = i * 2
        indices.extend([a, a + 2, a + 1])
        indices.extend([a + 1, a + 2, a + 3])

    # Caps
    base = len(positions)
    # Top cap center
    positions.append([0, half, 0])
    normals.append([0, 1, 0])
    uvs.append([0.5, 0.5])
    center_top = base
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        c, s = math.cos(theta), math.sin(theta)
        positions.append([radius * c, half, radius * s])
        normals.append([0, 1, 0])
        uvs.append([0.5 + 0.5 * c, 0.5 + 0.5 * s])
    for i in range(segments):
        n = i + 1 if i + 1 < segments else 0
        indices.extend([center_top, center_top + 1 + i, center_top + 1 + n])

    base2 = len(positions)
    # Bottom cap center
    positions.append([0, -half, 0])
    normals.append([0, -1, 0])
    uvs.append([0.5, 0.5])
    center_bot = base2
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        c, s = math.cos(theta), math.sin(theta)
        positions.append([radius * c, -half, radius * s])
        normals.append([0, -1, 0])
        uvs.append([0.5 + 0.5 * c, 0.5 + 0.5 * s])
    for i in range(segments):
        n = i + 1 if i + 1 < segments else 0
        indices.extend([center_bot, center_bot + 1 + n, center_bot + 1 + i])

    return positions, normals, uvs, indices

def encode_mesh(positions, normals, uvs, indices):
    """Encode mesh data into binary buffers, return (pos_bytes, norm_bytes, uv_bytes, idx_bytes, bounds)."""
    pos_b = b''.join(pack_vec3(*p) for p in positions)
    norm_b = b''.join(pack_vec3(*n) for n in normals)
    uv_b = b''.join(pack_vec2(*u) for u in uvs)

    use_u32 = max(indices) > 65535
    if use_u32:
        idx_b = b''.join(struct.pack('<I', i) for i in indices)
    else:
        idx_b = b''.join(pack_u16(i) for i in indices)

    mins = [min(p[i] for p in positions) for i in range(3)]
    maxs = [max(p[i] for p in positions) for i in range(3)]

    return pos_b, norm_b, uv_b, idx_b, mins, maxs, use_u32

def pad4(data):
    """Pad bytes to 4-byte alignment."""
    r = len(data) % 4
    return data + b'\x00' * ((4 - r) % 4)

def build_glb():
    # Generate all meshes
    meshes = {
        'Ground': make_plane(8.0),
        'Cube': make_cube(),
        'Sphere': make_uv_sphere(24, 48),
        'Torus': make_torus(),
        'Cylinder': make_cylinder(),
    }

    # Materials: name -> (baseColor RGBA, metallic, roughness)
    materials = {
        'Ground':   ([0.35, 0.35, 0.38, 1.0], 0.0, 0.9),
        'Cube':     ([0.9, 0.15, 0.12, 1.0], 0.0, 0.45),
        'Sphere':   ([0.12, 0.45, 0.9, 1.0], 0.85, 0.15),
        'Torus':    ([0.95, 0.75, 0.1, 1.0], 0.9, 0.1),
        'Cylinder': ([0.15, 0.8, 0.3, 1.0], 0.0, 0.6),
    }

    # Node transforms: name -> (tx, ty, tz)
    transforms = {
        'Ground':   (0.0, -0.5, 0.0),
        'Cube':     (-1.5, 0.0, 0.0),
        'Sphere':   (0.0, 0.0, 1.0),
        'Torus':    (1.5, 0.2, 0.0),
        'Cylinder': (0.0, 0.01, -1.5),
    }

    mesh_names = ['Ground', 'Cube', 'Sphere', 'Torus', 'Cylinder']

    # Encode all meshes and build one big binary buffer
    binary = b''
    buffer_views = []
    accessors = []
    gltf_meshes = []
    gltf_materials = []
    gltf_nodes = []

    for mi, name in enumerate(mesh_names):
        positions, normals, uvs, indices = meshes[name]
        pos_b, norm_b, uv_b, idx_b, mins, maxs, use_u32 = encode_mesh(positions, normals, uvs, indices)

        # Pad each segment to 4-byte alignment
        segments = [pad4(idx_b), pad4(pos_b), pad4(norm_b), pad4(uv_b)]
        bv_start = len(binary)
        offsets = []
        for seg in segments:
            offsets.append(len(binary))
            binary += seg

        idx_count = len(indices)
        vert_count = len(positions)

        # Buffer views: idx, pos, norm, uv
        bv_base = len(buffer_views)
        # Index buffer view
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offsets[0],
            "byteLength": len(idx_b),
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        })
        # Position buffer view
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offsets[1],
            "byteLength": len(pos_b),
            "byteStride": 12,
            "target": 34962  # ARRAY_BUFFER
        })
        # Normal buffer view
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offsets[2],
            "byteLength": len(norm_b),
            "byteStride": 12,
            "target": 34962
        })
        # UV buffer view
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offsets[3],
            "byteLength": len(uv_b),
            "byteStride": 8,
            "target": 34962
        })

        # Accessors: idx, pos, norm, uv
        acc_base = len(accessors)
        accessors.append({
            "bufferView": bv_base,
            "componentType": 5125 if use_u32 else 5123,  # U32 or U16
            "count": idx_count,
            "type": "SCALAR",
            "max": [max(indices)],
            "min": [min(indices)]
        })
        accessors.append({
            "bufferView": bv_base + 1,
            "componentType": 5126,  # FLOAT
            "count": vert_count,
            "type": "VEC3",
            "max": maxs,
            "min": mins
        })
        accessors.append({
            "bufferView": bv_base + 2,
            "componentType": 5126,
            "count": vert_count,
            "type": "VEC3"
        })
        accessors.append({
            "bufferView": bv_base + 3,
            "componentType": 5126,
            "count": vert_count,
            "type": "VEC2"
        })

        # Mesh
        gltf_meshes.append({
            "name": name,
            "primitives": [{
                "attributes": {
                    "POSITION": acc_base + 1,
                    "NORMAL": acc_base + 2,
                    "TEXCOORD_0": acc_base + 3
                },
                "indices": acc_base,
                "material": mi
            }]
        })

        # Material
        base_color, metallic, roughness = materials[name]
        gltf_materials.append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": base_color,
                "metallicFactor": metallic,
                "roughnessFactor": roughness
            },
            "doubleSided": name == 'Ground'
        })

        # Node
        tx, ty, tz = transforms[name]
        node = {"name": name, "mesh": mi}
        if tx != 0 or ty != 0 or tz != 0:
            node["translation"] = [tx, ty, tz]
        gltf_nodes.append(node)

    # Build glTF JSON
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "esox-sample-gen"
        },
        "scene": 0,
        "scenes": [{
            "name": "Sample Scene",
            "nodes": list(range(len(gltf_nodes)))
        }],
        "nodes": gltf_nodes,
        "meshes": gltf_meshes,
        "materials": gltf_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{
            "byteLength": len(binary)
        }]
    }

    json_str = json.dumps(gltf, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    # Pad JSON to 4-byte alignment with spaces
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * json_pad

    # Pad binary to 4-byte alignment
    bin_pad = (4 - len(binary) % 4) % 4
    binary += b'\x00' * bin_pad

    # GLB structure
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary)

    glb = b''
    # Header
    glb += struct.pack('<III', 0x46546C67, 2, total_length)  # magic, version, length
    # JSON chunk
    glb += struct.pack('<II', len(json_bytes), 0x4E4F534A)
    glb += json_bytes
    # BIN chunk
    glb += struct.pack('<II', len(binary), 0x004E4942)
    glb += binary

    return glb

if __name__ == '__main__':
    glb = build_glb()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample.glb')
    with open(out, 'wb') as f:
        f.write(glb)
    print(f"Wrote {len(glb)} bytes to {out}")
