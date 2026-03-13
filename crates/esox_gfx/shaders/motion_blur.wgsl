// Fullscreen triangle + directional blur along velocity.

struct BlurParams {
    max_velocity: f32,
    intensity: f32,
    sample_count: f32,
    _pad: f32,
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var velocity_tex: texture_2d<f32>;
@group(0) @binding(2) var linear_samp: sampler;
@group(0) @binding(3) var<uniform> params: BlurParams;

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let vel_raw = textureSample(velocity_tex, linear_samp, in.uv).rg;
    let tex_size = vec2<f32>(textureDimensions(scene_tex));

    // Convert UV-space velocity to pixel-space, clamp magnitude.
    var vel_px = vel_raw * tex_size;
    let mag = length(vel_px);
    if mag > params.max_velocity {
        vel_px = vel_px * (params.max_velocity / mag);
    }

    // Scale by intensity, convert back to UV space.
    let vel_uv = (vel_px * params.intensity) / tex_size;

    let n = i32(params.sample_count);
    var color = vec4<f32>(0.0);
    for (var i = 0; i < n; i = i + 1) {
        let t = (f32(i) / f32(n - 1)) - 0.5; // -0.5 .. +0.5
        let sample_uv = in.uv + vel_uv * t;
        color = color + textureSample(scene_tex, linear_samp, sample_uv);
    }
    color = color / f32(n);

    return color;
}
