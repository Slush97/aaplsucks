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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let ts = params.texel_size;

    // 8-tap diamond + cross pattern.
    var result = textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x,  0.0))  * 2.0;
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x,  0.0))  * 2.0;
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0, -ts.y))  * 2.0;
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,  ts.y))  * 2.0;
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x, -ts.y));
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x, -ts.y));
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x,  ts.y));
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x,  ts.y));

    return result / 12.0;
}
