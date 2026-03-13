// -- Bindings --

@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var t_noise: texture_2d<f32>;
@group(0) @binding(2) var<uniform> kernel: array<vec4<f32>, 64>;
@group(0) @binding(3) var<uniform> params: SsaoParams;
@group(0) @binding(4) var s_point: sampler;
@group(0) @binding(5) var s_repeat: sampler;

struct SsaoParams {
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    noise_scale: vec2<f32>,
    radius: f32,
    bias: f32,
    intensity: f32,
    kernel_size: f32,
    _pad: vec2<f32>,
}

// -- Vertex shader -- fullscreen triangle --

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOutput {
    // Generate a fullscreen triangle from vertex_index (0, 1, 2).
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// -- Fragment shader --

/// Reconstruct view-space position from depth and UV.
fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // UV to clip space: [0,1] -> [-1,1], flip Y.
    let clip = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let view_h = params.inv_projection * clip;
    return view_h.xyz / view_h.w;
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) f32 {
    let uv = in.uv;
    let tex_size = vec2<f32>(textureDimensions(t_depth));

    // Sample depth at this fragment.
    let depth = textureLoad(t_depth, vec2<i32>(uv * tex_size), 0);

    // Early out for far plane (no geometry).
    if depth >= 1.0 {
        return 1.0;
    }

    // Reconstruct view-space position.
    let view_pos = reconstruct_view_pos(uv, depth);

    // Reconstruct normal from depth via cross(dpdx, dpdy).
    let view_pos_dx = dpdx(view_pos);
    let view_pos_dy = dpdy(view_pos);
    let normal = normalize(cross(view_pos_dy, view_pos_dx));

    // Sample noise for random tangent rotation.
    let noise_uv = uv * params.noise_scale;
    let noise_val = textureSample(t_noise, s_repeat, noise_uv).rg * 2.0 - 1.0;
    let random_vec = vec3<f32>(noise_val, 0.0);

    // Build TBN matrix (Gram-Schmidt).
    let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    // Accumulate occlusion.
    var occlusion = 0.0;
    let sample_count = i32(params.kernel_size);

    for (var i = 0; i < sample_count; i++) {
        // Rotate kernel sample into view space via TBN.
        let sample_dir = tbn * kernel[i].xyz;
        let sample_pos = view_pos + sample_dir * params.radius;

        // Project sample to screen space.
        let proj = params.projection * vec4<f32>(sample_pos, 1.0);
        var sample_uv = proj.xy / proj.w;
        sample_uv = sample_uv * 0.5 + 0.5;
        sample_uv.y = 1.0 - sample_uv.y;

        // Sample depth at projected position.
        let sample_screen = vec2<i32>(sample_uv * tex_size);
        let sample_depth = textureLoad(t_depth, sample_screen, 0);
        let sample_view = reconstruct_view_pos(sample_uv, sample_depth);

        // Range check: avoid occlusion from distant geometry.
        let range_check = smoothstep(0.0, 1.0, params.radius / abs(view_pos.z - sample_view.z));

        // Occlusion test: is the sample behind the surface?
        if sample_view.z >= sample_pos.z + params.bias {
            occlusion += range_check;
        }
    }

    occlusion = occlusion / f32(sample_count);
    let ao = 1.0 - (occlusion * params.intensity);
    return clamp(ao, 0.0, 1.0);
}
