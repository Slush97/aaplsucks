// -- Bindings --

@group(0) @binding(0) var t_occlusion: texture_2d<f32>;
@group(0) @binding(1) var s_point: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;

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

// -- Vertex shader -- fullscreen triangle --

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

// -- Fragment shader -- 4x4 box blur --

@fragment
fn fs_main(in: VsOutput) -> @location(0) f32 {
    let tex_size = vec2<f32>(textureDimensions(t_occlusion));
    let texel_size = 1.0 / tex_size;
    let uv = in.uv;

    var result = 0.0;
    for (var x = -2; x < 2; x++) {
        for (var y = -2; y < 2; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            result += textureSampleLevel(t_occlusion, s_point, uv + offset, 0.0).r;
        }
    }
    result /= 16.0;
    return result;
}
