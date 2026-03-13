// Fullscreen triangle + velocity reconstruction from depth.

struct VelocityParams {
    inv_view_projection: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
    viewport: vec4<f32>,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var point_samp: sampler;
@group(0) @binding(2) var<uniform> params: VelocityParams;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle covering clip space.
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    var out: VsOut;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<i32>(textureDimensions(depth_tex));
    let coord = vec2<i32>(in.uv * vec2<f32>(tex_size));
    let depth = textureLoad(depth_tex, coord, 0);

    // Reconstruct NDC (wgpu: y-down in framebuffer, but UV is already correct).
    let ndc = vec3<f32>(in.uv.x * 2.0 - 1.0, -(in.uv.y * 2.0 - 1.0), depth);

    // Reconstruct world position.
    let world_h = params.inv_view_projection * vec4<f32>(ndc, 1.0);
    let world = world_h.xyz / world_h.w;

    // Reproject into previous frame clip space.
    let prev_clip = params.prev_view_projection * vec4<f32>(world, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;

    // Velocity in UV space (half NDC delta).
    let velocity = (ndc.xy - prev_ndc) * 0.5;

    return vec4<f32>(velocity, 0.0, 1.0);
}
