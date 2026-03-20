struct CompositeParams {
    bloom_intensity: f32,
    tone_map: f32,
    ssao_enabled: f32,
    _pad0: f32,
    fog_color_enabled: vec4<f32>,
    fog_params: vec4<f32>,
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var ssao_tex: texture_2d<f32>;
@group(0) @binding(3) var linear_samp: sampler;
@group(0) @binding(4) var<uniform> params: CompositeParams;
@group(0) @binding(5) var t_depth: texture_depth_2d;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    var out: VsOut;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// ACES filmic tone mapping (simple fit).
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var color = textureSample(scene_tex, linear_samp, in.uv).rgb;

    // Add bloom.
    let bloom = textureSample(bloom_tex, linear_samp, in.uv).rgb;
    color = color + bloom * params.bloom_intensity;

    // Apply SSAO.
    if params.ssao_enabled > 0.5 {
        let ao = max(textureSample(ssao_tex, linear_samp, in.uv).r, 0.3);
        color = color * ao;
    }

    // Tone mapping.
    if params.tone_map > 0.5 {
        color = aces_filmic(color);
    }

    // Distance fog (applied after tone mapping for correct display-space blending).
    if params.fog_color_enabled.w > 0.5 {
        let pixel = vec2<i32>(in.position.xy);
        let ndc_depth = textureLoad(t_depth, pixel, 0);
        let near = params.fog_params.z;
        let far = params.fog_params.w;
        let linear_depth = (near * far) / (far - ndc_depth * (far - near));
        let fog_start = params.fog_params.x;
        let fog_end = params.fog_params.y;
        let fog_factor = clamp((linear_depth - fog_start) / max(fog_end - fog_start, 0.001), 0.0, 1.0);
        let fog = fog_factor * fog_factor;
        color = mix(color, params.fog_color_enabled.xyz, fog);
    }

    return vec4<f32>(color, 1.0);
}
