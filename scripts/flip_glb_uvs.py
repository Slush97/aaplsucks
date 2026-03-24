"""Flip UV V-axis in a GLB file without any external dependencies.

Usage:
    python3 flip_glb_uvs.py input.glb output.glb

Reads the GLB binary, finds all TEXCOORD_0 accessors, and flips V (v = 1 - v).
Preserves all textures, animations, and other data byte-for-byte.
"""
import json
import struct
import sys
import copy

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.glb output.glb")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    with open(src, "rb") as f:
        data = bytearray(f.read())

    # ── Parse GLB header ──
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, f"Not a GLB file (magic={magic:#x})"
    assert version == 2, f"Unsupported GLB version {version}"

    # ── Parse chunks ──
    offset = 12
    json_chunk = None
    bin_chunk_offset = None
    bin_chunk_length = None

    while offset < length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        chunk_data_offset = offset + 8
        if chunk_type == 0x4E4F534A:  # JSON
            json_chunk = json.loads(data[chunk_data_offset:chunk_data_offset + chunk_length])
        elif chunk_type == 0x004E4942:  # BIN
            bin_chunk_offset = chunk_data_offset
            bin_chunk_length = chunk_length
        offset = chunk_data_offset + chunk_length

    assert json_chunk is not None, "No JSON chunk found"
    assert bin_chunk_offset is not None, "No BIN chunk found"

    accessors = json_chunk.get("accessors", [])
    buffer_views = json_chunk.get("bufferViews", [])
    meshes = json_chunk.get("meshes", [])

    # ── Find all TEXCOORD_0 accessor indices ──
    texcoord_accessors = set()
    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if "TEXCOORD_0" in attrs:
                texcoord_accessors.add(attrs["TEXCOORD_0"])

    if not texcoord_accessors:
        print("No TEXCOORD_0 attributes found, nothing to flip.")
        with open(dst, "wb") as f:
            f.write(data)
        return

    # ── Component type sizes ──
    COMP_SIZES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    COMP_FORMATS = {5126: "f", 5122: "h", 5123: "H"}  # float, short, ushort

    flipped = 0
    for acc_idx in texcoord_accessors:
        acc = accessors[acc_idx]
        assert acc["type"] == "VEC2", f"TEXCOORD_0 accessor {acc_idx} is {acc['type']}, expected VEC2"

        comp_type = acc["componentType"]
        count = acc["count"]
        bv_idx = acc["bufferView"]
        bv = buffer_views[bv_idx]

        byte_offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        byte_stride = bv.get("byteStride", 0)
        comp_size = COMP_SIZES[comp_type]

        # Default stride for VEC2 is 2 * comp_size.
        if byte_stride == 0:
            byte_stride = 2 * comp_size

        abs_offset = bin_chunk_offset + byte_offset

        if comp_type == 5126:  # FLOAT
            for i in range(count):
                v_offset = abs_offset + i * byte_stride + comp_size  # V is second component
                v_val = struct.unpack_from("<f", data, v_offset)[0]
                struct.pack_into("<f", data, v_offset, 1.0 - v_val)
            flipped += count

        elif comp_type == 5123:  # UNSIGNED_SHORT (normalized)
            for i in range(count):
                v_offset = abs_offset + i * byte_stride + comp_size
                v_val = struct.unpack_from("<H", data, v_offset)[0]
                struct.pack_into("<H", data, v_offset, 65535 - v_val)
            flipped += count

        elif comp_type == 5121:  # UNSIGNED_BYTE (normalized)
            for i in range(count):
                v_offset = abs_offset + i * byte_stride + comp_size
                data[v_offset] = 255 - data[v_offset]
            flipped += count

        else:
            print(f"  warning: unsupported componentType {comp_type} for accessor {acc_idx}, skipping")

        # Update accessor min/max if present.
        if "min" in acc and "max" in acc:
            old_min_v = acc["min"][1]
            old_max_v = acc["max"][1]
            if comp_type == 5126:
                acc["min"][1] = 1.0 - old_max_v
                acc["max"][1] = 1.0 - old_min_v
            elif comp_type == 5123:
                acc["min"][1] = 65535 - old_max_v
                acc["max"][1] = 65535 - old_min_v

    print(f"[flip] flipped V on {len(texcoord_accessors)} accessors, {flipped} vertices total")

    # ── Rewrite JSON chunk with updated min/max ──
    new_json = json.dumps(json_chunk, separators=(",", ":")).encode("utf-8")
    # Pad to 4-byte alignment with spaces.
    while len(new_json) % 4 != 0:
        new_json += b" "

    # Rebuild GLB.
    # Original JSON chunk location.
    orig_json_offset = 12 + 8  # header + first chunk header
    orig_json_length = struct.unpack_from("<I", data, 12)[0]
    orig_json_padded = orig_json_length

    if len(new_json) == orig_json_padded:
        # Same size — just overwrite in place.
        data[orig_json_offset:orig_json_offset + len(new_json)] = new_json
    else:
        # Different size — rebuild the file.
        bin_data = data[bin_chunk_offset:bin_chunk_offset + bin_chunk_length]

        total = 12 + 8 + len(new_json) + 8 + len(bin_data)
        out = bytearray(total)
        struct.pack_into("<III", out, 0, 0x46546C67, 2, total)
        struct.pack_into("<II", out, 12, len(new_json), 0x4E4F534A)
        out[20:20 + len(new_json)] = new_json
        bin_header_off = 20 + len(new_json)
        struct.pack_into("<II", out, bin_header_off, len(bin_data), 0x004E4942)
        out[bin_header_off + 8:bin_header_off + 8 + len(bin_data)] = bin_data
        data = out

    with open(dst, "wb") as f:
        f.write(data)

    orig_size = len(open(src, "rb").read())
    new_size = len(data)
    print(f"[flip] {src} ({orig_size:,} bytes) -> {dst} ({new_size:,} bytes)")


if __name__ == "__main__":
    main()
