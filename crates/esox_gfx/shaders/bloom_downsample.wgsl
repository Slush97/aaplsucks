struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

struct BloomParams {
    texel_size: vec2<f32>,
    threshold: f32,
    soft_knee: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BloomParams;

// Soft brightness thresholding: smoothly ramps from 0 at (threshold - knee)
// to full brightness at (threshold + knee). Returns the color contribution.
fn prefilter(color: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let brightness = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Soft knee curve: quadratic ramp in the transition region.
    let soft = brightness - threshold + knee;
    let soft_clamped = clamp(soft, 0.0, 2.0 * knee);
    var contribution = soft_clamped * soft_clamped / (4.0 * knee + 0.00001);
    // Above threshold, use full excess brightness.
    contribution = max(contribution, brightness - threshold);
    contribution = max(contribution, 0.0);
    return color * (contribution / max(brightness, 0.00001));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let ts = params.texel_size;

    var center = textureSample(src_texture, src_sampler, uv);
    var tl = textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x, -ts.y));
    var tr = textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x, -ts.y));
    var bl = textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x,  ts.y));
    var br = textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x,  ts.y));

    // Apply brightness threshold on first downsample only (threshold > 0).
    if params.threshold > 0.0 {
        center = vec4<f32>(prefilter(center.rgb, params.threshold, params.soft_knee), center.a);
        tl = vec4<f32>(prefilter(tl.rgb, params.threshold, params.soft_knee), tl.a);
        tr = vec4<f32>(prefilter(tr.rgb, params.threshold, params.soft_knee), tr.a);
        bl = vec4<f32>(prefilter(bl.rgb, params.threshold, params.soft_knee), bl.a);
        br = vec4<f32>(prefilter(br.rgb, params.threshold, params.soft_knee), br.a);
    }

    return (center * 4.0 + tl + tr + bl + br) / 8.0;
}
