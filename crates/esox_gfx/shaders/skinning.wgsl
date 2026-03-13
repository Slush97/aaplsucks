// Vertex3D: position(3f), normal(3f), uv(2f), color(4f), tangent(4f) = 16 floats = 64 bytes
// Stored as array<vec4<f32>, 4> for alignment.

struct SkinVertex {
    joints: vec4<u32>,
    weights: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> source: array<array<vec4<f32>, 4>>;
@group(0) @binding(1) var<storage, read_write> output: array<array<vec4<f32>, 4>>;
@group(0) @binding(2) var<storage, read> skin_data: array<SkinVertex>;
@group(0) @binding(3) var<storage, read> joints: array<mat4x4<f32>>;

@compute @workgroup_size(64)
fn skin_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let vertex_count = arrayLength(&source);
    if idx >= vertex_count {
        return;
    }

    let src = source[idx];
    let skin = skin_data[idx];

    // Build skin matrix from 4 joint influences.
    var skin_mat = joints[skin.joints.x] * skin.weights.x
                 + joints[skin.joints.y] * skin.weights.y
                 + joints[skin.joints.z] * skin.weights.z
                 + joints[skin.joints.w] * skin.weights.w;

    // Extract position (vec4<f32>[0].xyz) and normal (vec4<f32>[0].w, vec4<f32>[1].xy).
    // Layout: [px py pz nx] [ny nz u v] [cr cg cb ca] [tx ty tz tw]
    let position = vec4<f32>(src[0].xyz, 1.0);
    let normal = vec3<f32>(src[0].w, src[1].x, src[1].y);
    let tangent_xyz = src[3].xyz;
    let tangent_w = src[3].w;

    // Transform position.
    let skinned_pos = skin_mat * position;

    // Transform normal using adjugate (cofactor) for correctness with non-uniform scale.
    let m0 = skin_mat[0].xyz;
    let m1 = skin_mat[1].xyz;
    let m2 = skin_mat[2].xyz;
    let adj = mat3x3<f32>(cross(m1, m2), cross(m2, m0), cross(m0, m1));
    let skinned_normal = normalize(normal * adj);

    // Transform tangent direction (tangents use the regular linear transform).
    let linear_mat = mat3x3<f32>(m0, m1, m2);
    let skinned_tangent = normalize(linear_mat * tangent_xyz);

    // Write output -- preserve UV, color, tangent handedness.
    var out: array<vec4<f32>, 4>;
    out[0] = vec4<f32>(skinned_pos.xyz, skinned_normal.x);
    out[1] = vec4<f32>(skinned_normal.y, skinned_normal.z, src[1].z, src[1].w);
    out[2] = src[2]; // color unchanged
    out[3] = vec4<f32>(skinned_tangent, tangent_w);

    output[idx] = out;
}
