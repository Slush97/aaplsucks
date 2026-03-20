//! Embedded shader source strings for the 3D renderer.

use std::collections::HashMap;
use super::material::MaterialType;
use super::shader_library::ShaderLibrary;

pub(crate) fn compile_shader_modules(device: &wgpu::Device, library: &ShaderLibrary) -> HashMap<MaterialType, wgpu::ShaderModule> {
    let mut modules = HashMap::new();

    let unlit_src = library.compose_material_shader(MaterialType::Unlit);
    let lit_src = library.compose_material_shader(MaterialType::Lit);
    let pbr_src = library.compose_material_shader(MaterialType::PBR);
    let toon_src = library.compose_material_shader(MaterialType::Toon);

    modules.insert(
        MaterialType::Unlit,
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_shader_unlit"),
            source: wgpu::ShaderSource::Wgsl(unlit_src.into()),
        }),
    );
    modules.insert(
        MaterialType::Lit,
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_shader_lit"),
            source: wgpu::ShaderSource::Wgsl(lit_src.into()),
        }),
    );
    modules.insert(
        MaterialType::PBR,
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_shader_pbr"),
            source: wgpu::ShaderSource::Wgsl(pbr_src.into()),
        }),
    );
    modules.insert(
        MaterialType::Toon,
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_shader_toon"),
            source: wgpu::ShaderSource::Wgsl(toon_src.into()),
        }),
    );

    modules
}

/// Shared shader preamble: struct definitions, bind groups, vertex shader.
pub(crate) const SHADER_PREAMBLE: &str = r"
struct Uniforms {
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,
    time: vec4<f32>,
    camera_forward: vec4<f32>,
}

struct ShadowUniforms {
    light_vp: array<mat4x4<f32>, 4>,
    splits_count: vec4<f32>,
    shadow_config: vec4<f32>,
}

struct PointLightGpu {
    position_range: vec4<f32>,
    color_intensity: vec4<f32>,
}

struct SpotLightGpu {
    position_range: vec4<f32>,
    direction_inner: vec4<f32>,
    color_intensity: vec4<f32>,
    outer_pad: vec4<f32>,
}

struct LightUniforms {
    ambient: vec4<f32>,
    directional_dir_intensity: vec4<f32>,
    directional_color_count: vec4<f32>,
    spot_count_pad: vec4<f32>,
    point_lights: array<PointLightGpu, 8>,
    spot_lights: array<SpotLightGpu, 4>,
}

struct MaterialUniforms {
    albedo: vec4<f32>,
    emissive_metallic: vec4<f32>,
    roughness_opacity_flags: vec4<f32>,
    texture_flags: vec4<f32>,
    extra: vec4<f32>,
}

struct OmniShadowUniforms {
    point_light_vp: array<mat4x4<f32>, 24>,
    spot_light_vp: array<mat4x4<f32>, 4>,
    omni_config: vec4<f32>,
    omni_config2: vec4<f32>,
}

// Group 0: Scene uniforms + shadow.
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(2) var shadow_depth: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;
@group(0) @binding(4) var<uniform> omni_shadow: OmniShadowUniforms;
@group(0) @binding(5) var point_shadow_depth: texture_depth_2d_array;
@group(0) @binding(6) var spot_shadow_depth: texture_depth_2d_array;

// Group 1: Lights + IBL.
@group(1) @binding(0) var<uniform> lights: LightUniforms;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(3) var brdf_lut: texture_2d<f32>;
@group(1) @binding(4) var ibl_sampler: sampler;

// Group 2: Material.
@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var albedo_tex: texture_2d<f32>;
@group(2) @binding(2) var normal_tex: texture_2d<f32>;
@group(2) @binding(3) var mr_tex: texture_2d<f32>;
@group(2) @binding(4) var emissive_tex: texture_2d<f32>;
@group(2) @binding(5) var mat_sampler: sampler;

struct VertexInput {
    // Per-vertex (slot 0)
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) tangent: vec4<f32>,
    // Per-instance (slot 1)
    @location(5) model_0: vec4<f32>,
    @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>,
    @location(8) model_3: vec4<f32>,
    @location(9) inst_color: vec4<f32>,
    @location(10) inst_params: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) world_tangent: vec4<f32>,
    @location(6) view_depth: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let world_pos = model * vec4<f32>(in.position, 1.0);

    // Normal matrix: normalize columns of mat3(model).
    let normal_mat = mat3x3<f32>(
        normalize(model[0].xyz),
        normalize(model[1].xyz),
        normalize(model[2].xyz),
    );

    let clip = uniforms.view_projection * world_pos;

    var out: VertexOutput;
    out.clip_position = clip;
    out.world_position = world_pos.xyz;
    out.world_normal = normalize(normal_mat * in.normal);
    out.uv = in.uv;
    out.color = in.color * in.inst_color;
    out.params = in.inst_params;
    out.view_depth = dot(world_pos.xyz - uniforms.camera_position.xyz, uniforms.camera_forward.xyz);

    // Transform tangent to world space (w = bitangent handedness, preserved).
    // Guard against zero-length tangent (procedural meshes without tangents).
    let raw_wt = normal_mat * in.tangent.xyz;
    let wt = select(vec3<f32>(0.0), normalize(raw_wt), length(raw_wt) > 0.001);
    out.world_tangent = vec4<f32>(wt, in.tangent.w);

    return out;
}

// Sample shadow for a single cascade. Returns 0.0 (shadowed) to 1.0 (lit).
fn shadow_sample_cascade(biased_pos: vec3<f32>, cascade: i32) -> f32 {
    let light_pos = shadow.light_vp[cascade] * vec4<f32>(biased_pos, 1.0);
    var proj = light_pos.xyz / light_pos.w;

    // NDC to UV: [-1,1] -> [0,1], flip Y.
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));

    // Out of shadow map bounds -> fully lit.
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0 {
        return 1.0;
    }

    let depth_bias = shadow.shadow_config.x;
    let compare_depth = proj.z - depth_bias;

    // 3x3 PCF (percentage-closer filtering).
    let tex_size = vec2<f32>(textureDimensions(shadow_depth));
    let texel_size = 1.0 / tex_size;
    var total = 0.0;
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel_size;
            total += textureSampleCompareLevel(
                shadow_depth,
                shadow_sampler,
                uv + offset,
                cascade,
                compare_depth,
            );
        }
    }

    return total / 9.0;
}

// Shadow factor: returns 0.0 (fully shadowed) to 1.0 (fully lit).
// Applies normal bias and blends between adjacent cascades at split boundaries.
fn shadow_factor(world_pos: vec3<f32>, normal: vec3<f32>, view_depth: f32) -> f32 {
    let cascade_count = i32(shadow.shadow_config.w);
    if cascade_count == 0 {
        return 1.0;
    }

    // Apply normal bias: offset along surface normal to reduce acne at
    // grazing angles.
    let normal_bias = shadow.shadow_config.y;
    let biased_pos = world_pos + normal * normal_bias;

    // Select cascade by view depth.
    var cascade = cascade_count - 1;
    for (var i = 0; i < cascade_count; i = i + 1) {
        if view_depth < shadow.splits_count[i] {
            cascade = i;
            break;
        }
    }

    let sf = shadow_sample_cascade(biased_pos, cascade);

    // Blend with next cascade near the split boundary to hide seams.
    let next = cascade + 1;
    if next < cascade_count {
        let split_far = shadow.splits_count[cascade];
        // Blend zone: last 10% of current cascade's range.
        let split_near = select(0.0, shadow.splits_count[cascade - 1], cascade > 0);
        let blend_start = mix(split_near, split_far, 0.75);
        if view_depth > blend_start {
            let t = (view_depth - blend_start) / max(split_far - blend_start, 0.001);
            let sf_next = shadow_sample_cascade(biased_pos, next);
            return mix(sf, sf_next, t);
        }
    }

    return sf;
}

// Spot light attenuation.
fn spot_attenuation(cos_theta: f32, cos_inner: f32, cos_outer: f32) -> f32 {
    return smoothstep(cos_outer, cos_inner, cos_theta);
}

// Determine which cube face a direction vector points at and return (u, v, face_index).
fn cube_face_uv(dir: vec3<f32>) -> vec3<f32> {
    let abs_dir = abs(dir);
    var face: f32;
    var u: f32;
    var v: f32;

    if abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z {
        if dir.x > 0.0 {
            face = 0.0;
            u = -dir.z / abs_dir.x * 0.5 + 0.5;
            v = -dir.y / abs_dir.x * 0.5 + 0.5;
        } else {
            face = 1.0;
            u = dir.z / abs_dir.x * 0.5 + 0.5;
            v = -dir.y / abs_dir.x * 0.5 + 0.5;
        }
    } else if abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z {
        if dir.y > 0.0 {
            face = 2.0;
            u = dir.x / abs_dir.y * 0.5 + 0.5;
            v = dir.z / abs_dir.y * 0.5 + 0.5;
        } else {
            face = 3.0;
            u = dir.x / abs_dir.y * 0.5 + 0.5;
            v = -dir.z / abs_dir.y * 0.5 + 0.5;
        }
    } else {
        if dir.z > 0.0 {
            face = 4.0;
            u = dir.x / abs_dir.z * 0.5 + 0.5;
            v = -dir.y / abs_dir.z * 0.5 + 0.5;
        } else {
            face = 5.0;
            u = -dir.x / abs_dir.z * 0.5 + 0.5;
            v = -dir.y / abs_dir.z * 0.5 + 0.5;
        }
    }

    return vec3<f32>(u, v, face);
}

// Sample point shadow for a single cubemap face. Returns 0.0-1.0.
fn sample_point_shadow_face(light_idx: i32, face: i32, biased_pos: vec3<f32>) -> f32 {
    let layer = light_idx * 6 + face;

    let light_pos_h = omni_shadow.point_light_vp[layer] * vec4<f32>(biased_pos, 1.0);
    var proj = light_pos_h.xyz / light_pos_h.w;

    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));

    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0 {
        return 1.0;
    }

    let depth_bias = omni_shadow.omni_config.z;
    let compare_depth = proj.z - depth_bias;

    let tex_size = vec2<f32>(textureDimensions(point_shadow_depth));
    let texel_size = 1.0 / tex_size;
    var total = 0.0;
    for (var dx = 0; dx <= 1; dx = dx + 1) {
        for (var dy = 0; dy <= 1; dy = dy + 1) {
            let offset = (vec2<f32>(f32(dx), f32(dy)) - 0.5) * texel_size;
            total += textureSampleCompareLevel(
                point_shadow_depth,
                shadow_sampler,
                uv + offset,
                layer,
                compare_depth,
            );
        }
    }

    return total / 4.0;
}

fn cube_face_index(dir: vec3<f32>) -> i32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(1, 0, dir.x > 0.0);
    } else if a.y >= a.x && a.y >= a.z {
        return select(3, 2, dir.y > 0.0);
    } else {
        return select(5, 4, dir.z > 0.0);
    }
}

fn cube_secondary_face(dir: vec3<f32>) -> i32 {
    let a = abs(dir);
    var second_axis: i32;
    if a.x >= a.y && a.x >= a.z {
        second_axis = select(2, 1, a.z > a.y);
    } else if a.y >= a.x && a.y >= a.z {
        second_axis = select(0, 2, a.z > a.x);
    } else {
        second_axis = select(0, 1, a.y > a.x);
    }
    if second_axis == 0 {
        return select(1, 0, dir.x > 0.0);
    } else if second_axis == 1 {
        return select(3, 2, dir.y > 0.0);
    } else {
        return select(5, 4, dir.z > 0.0);
    }
}

fn point_shadow_factor(light_idx: i32, world_pos: vec3<f32>, light_pos: vec3<f32>, light_range: f32, normal: vec3<f32>) -> f32 {
    let point_shadow_count = i32(omni_shadow.omni_config.x);
    if light_idx >= point_shadow_count {
        return 1.0;
    }

    // Normal bias: offset along surface normal to prevent self-shadowing
    // at grazing angles (e.g. ground seen from side cubemap faces).
    let to_light = normalize(light_pos - world_pos);
    let n_dot_l = max(dot(normal, to_light), 0.0);
    let normal_bias = omni_shadow.omni_config.w;
    let bias_scale = max(1.0 - n_dot_l, 0.1);
    let biased_pos = world_pos + normal * normal_bias * bias_scale;

    let to_frag = biased_pos - light_pos;
    let a = abs(to_frag);

    let primary = cube_face_index(to_frag);
    let sf_primary = sample_point_shadow_face(light_idx, primary, biased_pos);

    // Blend between adjacent cubemap faces near boundaries to hide seams.
    let max_comp = max(max(a.x, a.y), a.z);
    let min_of_xy = min(a.x, a.y);
    let max_of_xy = max(a.x, a.y);
    let mid_comp = max(min(max_of_xy, a.z), min_of_xy);
    let edge_ratio = (max_comp - mid_comp) / max(max_comp, 0.001);

    if edge_ratio > 0.15 {
        return sf_primary;
    }

    let secondary = cube_secondary_face(to_frag);
    let sf_secondary = sample_point_shadow_face(light_idx, secondary, biased_pos);

    let blend = smoothstep(0.0, 0.15, edge_ratio);
    return mix(sf_secondary, sf_primary, blend);
}

fn spot_shadow_factor(light_idx: i32, world_pos: vec3<f32>) -> f32 {
    let spot_shadow_count = i32(omni_shadow.omni_config.y);
    if light_idx >= spot_shadow_count {
        return 1.0;
    }

    let light_pos_h = omni_shadow.spot_light_vp[light_idx] * vec4<f32>(world_pos, 1.0);
    var proj = light_pos_h.xyz / light_pos_h.w;

    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));

    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0 {
        return 1.0;
    }

    let depth_bias = omni_shadow.omni_config2.x;
    let compare_depth = proj.z - depth_bias;

    let tex_size = vec2<f32>(textureDimensions(spot_shadow_depth));
    let texel_size = 1.0 / tex_size;
    var total = 0.0;
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel_size;
            total += textureSampleCompareLevel(
                spot_shadow_depth,
                shadow_sampler,
                uv + offset,
                light_idx,
                compare_depth,
            );
        }
    }

    return total / 9.0;
}

// Helper: apply normal map using TBN matrix.
// Returns perturbed world normal. If tangent is zero-length, returns geometric normal.
fn apply_normal_map(geo_normal: vec3<f32>, world_tangent: vec4<f32>, uv: vec2<f32>, normal_scale: f32) -> vec3<f32> {
    let has_normal = material.texture_flags.y;
    let tang_len = length(world_tangent.xyz);
    if has_normal < 0.5 || tang_len < 0.001 {
        return geo_normal;
    }

    let t = normalize(world_tangent.xyz);
    let n = normalize(geo_normal);
    let b = cross(n, t) * world_tangent.w;

    let tbn = mat3x3<f32>(t, b, n);

    let sampled = textureSample(normal_tex, mat_sampler, uv).xyz;
    var ts_normal = sampled * 2.0 - 1.0;
    ts_normal.x = ts_normal.x * normal_scale;
    ts_normal.y = ts_normal.y * normal_scale;

    return normalize(tbn * ts_normal);
}
";

/// Fragment shader: Unlit — flat color + emissive.
pub(crate) const FS_UNLIT: &str = r"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(albedo_tex, mat_sampler, in.uv);
    let has_tex = material.texture_flags.x;
    let tex_color = mix(vec4<f32>(1.0), texel, has_tex);
    let base = in.color * material.albedo * tex_color;

    var emissive = material.emissive_metallic.xyz;
    let has_emissive_tex = material.texture_flags.w;
    if has_emissive_tex > 0.5 {
        let emissive_texel = textureSample(emissive_tex, mat_sampler, in.uv).rgb;
        emissive = emissive * emissive_texel;
    }

    return vec4<f32>(base.rgb + emissive, base.a);
}
";

/// Fragment shader: Lit — Lambertian diffuse + ambient + point lights + spot lights + shadows + normal mapping.
pub(crate) const FS_LIT: &str = r"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(albedo_tex, mat_sampler, in.uv);
    let has_tex = material.texture_flags.x;
    let tex_color = mix(vec4<f32>(1.0), texel, has_tex);
    let base = in.color * material.albedo * tex_color;

    var emissive = material.emissive_metallic.xyz;
    let has_emissive_tex = material.texture_flags.w;
    if has_emissive_tex > 0.5 {
        let emissive_texel = textureSample(emissive_tex, mat_sampler, in.uv).rgb;
        emissive = emissive * emissive_texel;
    }

    let normal_scale = material.extra.x;
    let n = apply_normal_map(in.world_normal, in.world_tangent, in.uv, normal_scale);

    // Ambient.
    let ambient = lights.ambient.rgb * lights.ambient.w;

    // Shadow factor for directional light.
    let sf = shadow_factor(in.world_position, n, in.view_depth);

    // Directional light (with shadows).
    // Negate: stored direction is light travel; shading needs surface-to-light.
    let light_dir = -normalize(lights.directional_dir_intensity.xyz);
    let dir_intensity = lights.directional_dir_intensity.w;
    let dir_color = lights.directional_color_count.xyz;
    let ndotl = max(dot(n, light_dir), 0.0);
    var diffuse = dir_color * dir_intensity * ndotl * sf;

    // Point lights.
    let point_count = i32(lights.directional_color_count.w);
    for (var i = 0; i < point_count; i = i + 1) {
        let pl = lights.point_lights[i];
        let pl_pos = pl.position_range.xyz;
        let pl_range = pl.position_range.w;
        let pl_color = pl.color_intensity.xyz;
        let pl_intensity = pl.color_intensity.w;

        let to_light = pl_pos - in.world_position;
        let dist = length(to_light);
        if dist < pl_range {
            let l = to_light / max(dist, 0.001);
            let dist_ratio = dist / pl_range;
            let range_atten = saturate(1.0 - dist_ratio * dist_ratio);
            let atten = pl_intensity * range_atten * range_atten / max(dist * dist, 0.01);
            let pl_ndotl = max(dot(n, l), 0.0);
            let psf = point_shadow_factor(i, in.world_position, pl_pos, pl_range, n);
            diffuse = diffuse + pl_color * atten * pl_ndotl * psf;
        }
    }

    // Spot lights.
    let spot_count = i32(lights.spot_count_pad.x);
    for (var i = 0; i < spot_count; i = i + 1) {
        let sl = lights.spot_lights[i];
        let sl_pos = sl.position_range.xyz;
        let sl_range = sl.position_range.w;
        let sl_dir = sl.direction_inner.xyz;
        let sl_cos_inner = sl.direction_inner.w;
        let sl_color = sl.color_intensity.xyz;
        let sl_intensity = sl.color_intensity.w;
        let sl_cos_outer = sl.outer_pad.x;

        let to_light = sl_pos - in.world_position;
        let dist = length(to_light);
        if dist < sl_range {
            let l = to_light / max(dist, 0.001);
            let cos_theta = dot(normalize(-sl_dir), l);
            let spot_atten = spot_attenuation(cos_theta, sl_cos_inner, sl_cos_outer);
            let dist_ratio = dist / sl_range;
            let range_atten = saturate(1.0 - dist_ratio * dist_ratio);
            let dist_atten = sl_intensity * range_atten * range_atten / max(dist * dist, 0.01);
            let sl_ndotl = max(dot(n, l), 0.0);
            let ssf = spot_shadow_factor(i, in.world_position);
            diffuse = diffuse + sl_color * dist_atten * spot_atten * sl_ndotl * ssf;
        }
    }

    let lit = base.rgb * (ambient + diffuse) + emissive;
    return vec4<f32>(lit, base.a);
}
";

/// Fragment shader: PBR — Cook-Torrance microfacet BRDF with shadows, spot lights, IBL.
pub(crate) const FS_PBR: &str = r"
const PI: f32 = 3.14159265358979323846;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn cook_torrance_brdf(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_v = max(dot(n, v), 0.001);
    let n_dot_l = max(dot(n, l), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);

    let numerator = d * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let specular = numerator / denominator;

    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

    return (k_d * albedo / PI + specular) * n_dot_l;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(albedo_tex, mat_sampler, in.uv);
    let has_tex = material.texture_flags.x;
    let tex_color = mix(vec4<f32>(1.0), texel, has_tex);
    let base = in.color * material.albedo * tex_color;
    let albedo = base.rgb;

    var emissive = material.emissive_metallic.xyz;
    let has_emissive_tex = material.texture_flags.w;
    if has_emissive_tex > 0.5 {
        let emissive_texel = textureSample(emissive_tex, mat_sampler, in.uv).rgb;
        emissive = emissive * emissive_texel;
    }

    // Read metallic/roughness from uniform or texture.
    var metallic = material.emissive_metallic.w;
    var roughness = max(material.roughness_opacity_flags.x, 0.04);

    let has_mr_tex = material.texture_flags.z;
    if has_mr_tex > 0.5 {
        let mr_sample = textureSample(mr_tex, mat_sampler, in.uv);
        // glTF channel packing: G=roughness, B=metallic
        roughness = max(roughness * mr_sample.g, 0.04);
        metallic = metallic * mr_sample.b;
    }

    // Normal mapping.
    let normal_scale = material.extra.x;
    let n = apply_normal_map(in.world_normal, in.world_tangent, in.uv, normal_scale);
    let v = normalize(uniforms.camera_position.xyz - in.world_position);
    let n_dot_v = max(dot(n, v), 0.001);

    // Shadow factor for directional light.
    let sf = shadow_factor(in.world_position, n, in.view_depth);

    // Directional light (with shadows).
    // Negate: stored direction is light travel; BRDF expects surface-to-light.
    let light_dir = -normalize(lights.directional_dir_intensity.xyz);
    let dir_intensity = lights.directional_dir_intensity.w;
    let dir_color = lights.directional_color_count.xyz;
    var lo = dir_color * dir_intensity * sf * cook_torrance_brdf(n, v, light_dir, albedo, metallic, roughness);

    // Point lights.
    let point_count = i32(lights.directional_color_count.w);
    for (var i = 0; i < point_count; i = i + 1) {
        let pl = lights.point_lights[i];
        let pl_pos = pl.position_range.xyz;
        let pl_range = pl.position_range.w;
        let pl_color = pl.color_intensity.xyz;
        let pl_intensity = pl.color_intensity.w;

        let to_light = pl_pos - in.world_position;
        let dist = length(to_light);
        if dist < pl_range {
            let l = to_light / max(dist, 0.001);
            let dist_ratio = dist / pl_range;
            let range_atten = saturate(1.0 - dist_ratio * dist_ratio);
            let atten = pl_intensity * range_atten * range_atten / max(dist * dist, 0.01);
            let psf = point_shadow_factor(i, in.world_position, pl_pos, pl_range, n);
            lo = lo + pl_color * atten * psf * cook_torrance_brdf(n, v, l, albedo, metallic, roughness);
        }
    }

    // Spot lights.
    let spot_count = i32(lights.spot_count_pad.x);
    for (var i = 0; i < spot_count; i = i + 1) {
        let sl = lights.spot_lights[i];
        let sl_pos = sl.position_range.xyz;
        let sl_range = sl.position_range.w;
        let sl_dir = sl.direction_inner.xyz;
        let sl_cos_inner = sl.direction_inner.w;
        let sl_color = sl.color_intensity.xyz;
        let sl_intensity = sl.color_intensity.w;
        let sl_cos_outer = sl.outer_pad.x;

        let to_light = sl_pos - in.world_position;
        let dist = length(to_light);
        if dist < sl_range {
            let l = to_light / max(dist, 0.001);
            let cos_theta = dot(normalize(-sl_dir), l);
            let s_atten = spot_attenuation(cos_theta, sl_cos_inner, sl_cos_outer);
            let dist_ratio = dist / sl_range;
            let range_atten = saturate(1.0 - dist_ratio * dist_ratio);
            let d_atten = sl_intensity * range_atten * range_atten / max(dist * dist, 0.01);
            let ssf = spot_shadow_factor(i, in.world_position);
            lo = lo + sl_color * d_atten * s_atten * ssf * cook_torrance_brdf(n, v, l, albedo, metallic, roughness);
        }
    }

    // IBL ambient (split-sum approximation).
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let k_d_ibl = (vec3<f32>(1.0) - f_ibl) * (1.0 - metallic);

    let diffuse_ibl = textureSample(irradiance_map, ibl_sampler, n).rgb * albedo;
    let r = reflect(-v, n);
    let prefiltered = textureSampleLevel(prefiltered_map, ibl_sampler, r, roughness * 4.0).rgb;
    let brdf_sample = textureSample(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness)).rg;
    let specular_ibl = prefiltered * (f_ibl * brdf_sample.x + brdf_sample.y);

    let ambient_ibl = (k_d_ibl * diffuse_ibl + specular_ibl) * lights.ambient.w;

    let color = ambient_ibl + lo + emissive;
    return vec4<f32>(color, base.a);
}
";

/// Fragment shader: Toon — Wind Waker-style cel shading with quantized diffuse, rim lighting.
pub(crate) const FS_TOON: &str = r"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // ── Base color ──
    let texel = textureSample(albedo_tex, mat_sampler, in.uv);
    let has_tex = material.texture_flags.x;
    let tex_color = mix(vec4<f32>(1.0), texel, has_tex);
    let base = in.color * material.albedo * tex_color;

    var emissive = material.emissive_metallic.xyz;
    let has_emissive_tex = material.texture_flags.w;
    if has_emissive_tex > 0.5 {
        let emissive_texel = textureSample(emissive_tex, mat_sampler, in.uv).rgb;
        emissive = emissive * emissive_texel;
    }

    // ── Normal mapping ──
    let normal_scale = material.extra.x;
    let n = apply_normal_map(in.world_normal, in.world_tangent, in.uv, normal_scale);
    let v = normalize(uniforms.camera_position.xyz - in.world_position);

    // ── Toon parameters ──
    let bands = max(material.extra.y, 2.0);
    let rim_power = material.extra.z;
    let rim_intensity = material.extra.w;

    // ── Ambient (with generous floor for Wind Waker-style readability) ──
    let raw_ambient = lights.ambient.rgb * lights.ambient.w;
    let ambient = max(raw_ambient, vec3<f32>(0.25));

    // ── Shadow factor (directional light) ──
    // Smooth the raw shadow factor so cascade-boundary noise doesn't
    // get amplified by the hard toon band quantization.
    let sf_raw = shadow_factor(in.world_position, n, in.view_depth);
    let sf = smoothstep(0.0, 0.5, sf_raw);

    // ── Directional light — quantized diffuse ──
    let light_dir = -normalize(lights.directional_dir_intensity.xyz);
    let dir_intensity = lights.directional_dir_intensity.w;
    let dir_color = lights.directional_color_count.xyz;
    let ndotl = max(dot(n, light_dir), 0.0);

    // Quantize NdotL into discrete bands, apply shadow.
    // Add small bias so shadow-side isn't completely dark.
    let quantized = floor(ndotl * bands + 0.5) / bands;
    let toon_diffuse = max(quantized * sf, 0.08);
    var diffuse = dir_color * dir_intensity * toon_diffuse;

    // ── Specular: hard-edged highlight ──
    let h = normalize(v + light_dir);
    let ndoth = max(dot(n, h), 0.0);
    let spec = step(0.92, ndoth) * sf;
    var specular = dir_color * dir_intensity * spec * 0.25;

    // ── Point lights (quantized) ──
    let point_count = i32(lights.directional_color_count.w);
    for (var i = 0; i < point_count; i = i + 1) {
        let pl = lights.point_lights[i];
        let pl_pos = pl.position_range.xyz;
        let pl_range = pl.position_range.w;
        let pl_color = pl.color_intensity.xyz;
        let pl_intensity = pl.color_intensity.w;

        let to_light = pl_pos - in.world_position;
        let dist = length(to_light);
        if dist < pl_range {
            let l = to_light / max(dist, 0.001);
            let dist_ratio = dist / pl_range;
            let range_atten = saturate(1.0 - dist_ratio * dist_ratio);
            let atten = pl_intensity * range_atten * range_atten / max(dist * dist, 0.01);
            let pl_ndotl = max(dot(n, l), 0.0);
            let pl_quant = floor(pl_ndotl * bands + 0.5) / bands;
            let psf = point_shadow_factor(i, in.world_position, pl_pos, pl_range, n);
            diffuse = diffuse + pl_color * atten * pl_quant * psf;
        }
    }

    // ── Spot lights (quantized) ──
    let spot_count = i32(lights.spot_count_pad.x);
    for (var i = 0; i < spot_count; i = i + 1) {
        let sl = lights.spot_lights[i];
        let sl_pos = sl.position_range.xyz;
        let sl_range = sl.position_range.w;
        let sl_dir = sl.direction_inner.xyz;
        let sl_cos_inner = sl.direction_inner.w;
        let sl_color = sl.color_intensity.xyz;
        let sl_intensity = sl.color_intensity.w;
        let sl_cos_outer = sl.outer_pad.x;

        let to_light = sl_pos - in.world_position;
        let dist = length(to_light);
        if dist < sl_range {
            let l = to_light / max(dist, 0.001);
            let cos_theta = dot(normalize(-sl_dir), l);
            let s_atten = spot_attenuation(cos_theta, sl_cos_inner, sl_cos_outer);
            let dist_ratio = dist / sl_range;
            let range_atten = saturate(1.0 - dist_ratio * dist_ratio);
            let dist_atten = sl_intensity * range_atten * range_atten / max(dist * dist, 0.01);
            let sl_ndotl = max(dot(n, l), 0.0);
            let sl_quant = floor(sl_ndotl * bands + 0.5) / bands;
            let ssf = spot_shadow_factor(i, in.world_position);
            diffuse = diffuse + sl_color * dist_atten * s_atten * sl_quant * ssf;
        }
    }

    // ── Rim lighting (Fresnel-based silhouette glow, lit side only) ──
    let ndotv = max(dot(n, v), 0.0);
    let rim = pow(1.0 - ndotv, rim_power) * rim_intensity;
    let rim_color = dir_color * rim * sf;

    let lit = base.rgb * (ambient + diffuse) + specular * base.rgb + rim_color + emissive;
    return vec4<f32>(lit, base.a);
}
";

/// Composite shader: fullscreen triangle blending scene HDR + bloom + SSAO, with ACES tone mapping.
pub(crate) const COMPOSITE_SHADER_3D: &str = r"
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

    // Apply SSAO (clamped to avoid crushing dark areas to pure black —
    // full-scene AO multiplication is an approximation; a min floor keeps
    // ambient-only surfaces visible).
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
        // Linearize depth: standard perspective projection.
        let linear_depth = (near * far) / (far - ndc_depth * (far - near));
        let fog_start = params.fog_params.x;
        let fog_end = params.fog_params.y;
        let fog_factor = clamp((linear_depth - fog_start) / max(fog_end - fog_start, 0.001), 0.0, 1.0);
        // Smooth fog curve.
        let fog = fog_factor * fog_factor;
        color = mix(color, params.fog_color_enabled.xyz, fog);
    }

    return vec4<f32>(color, 1.0);
}
";
