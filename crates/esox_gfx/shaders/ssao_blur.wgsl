// ── Bindings ──

@group(0) @binding(0) var t_occlusion: texture_2d<f32>;
@group(0) @binding(1) var s_point: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;
@group(0) @binding(3) var t_depth: texture_depth_2d;

struct BlurParams {
    // We reuse the SsaoParams layout; noise_scale.xy encodes viewport dimensions.
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    noise_scale: vec2<f32>,
    radius: f32,
    bias: f32,
    intensity: f32,
    kernel_size: f32,
    _pad: vec2<f32>,
}

// ── Vertex shader — fullscreen triangle ──

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOutput {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// ── Fragment shader — bilateral 5x5 blur ──

@fragment
fn fs_main(in: VsOutput) -> @location(0) f32 {
    let tex_size = vec2<f32>(textureDimensions(t_occlusion));
    let texel_size = 1.0 / tex_size;
    let uv = in.uv;
    let pixel = vec2<i32>(in.position.xy);

    let center_depth = textureLoad(t_depth, pixel, 0);

    // Depth threshold: reject samples that cross a depth discontinuity.
    // Use a relative threshold so it works at all distances.
    let depth_threshold = max(center_depth * 0.02, 0.0002);

    var result = 0.0;
    var total_weight = 0.0;

    for (var x = -2; x <= 2; x++) {
        for (var y = -2; y <= 2; y++) {
            let sample_pixel = pixel + vec2(x, y);
            let sample_depth = textureLoad(t_depth, sample_pixel, 0);
            let depth_diff = abs(center_depth - sample_depth);

            // Weight is 1.0 if depth is similar, 0.0 if across a discontinuity.
            let w = select(0.0, 1.0, depth_diff < depth_threshold);

            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let ao = textureSampleLevel(t_occlusion, s_point, uv + offset, 0.0).r;

            result += ao * w;
            total_weight += w;
        }
    }

    return result / max(total_weight, 1.0);
}
