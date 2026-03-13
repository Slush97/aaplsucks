// Depth resolve — resolves MSAA depth to a 1x texture by taking min across samples.
//
// Uses a pipeline-overridable constant for sample count. Outputs via
// @builtin(frag_depth) so the resolved texture is a real depth attachment.

@id(0) override SAMPLE_COUNT: i32 = 4;

@group(0) @binding(0) var t_depth_ms: texture_depth_multisampled_2d;

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

// ── Fragment shader — min-sample depth resolve ──

@fragment
fn fs_main(in: VsOutput) -> @builtin(frag_depth) f32 {
    let coord = vec2<i32>(in.position.xy);
    var min_depth = 1.0;
    for (var i = 0; i < SAMPLE_COUNT; i++) {
        let d = textureLoad(t_depth_ms, coord, i);
        min_depth = min(min_depth, d);
    }
    return min_depth;
}
