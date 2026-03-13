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

    // Normal matrix: inverse-transpose via cofactor (cross products of columns).
    // Handles non-uniform scaling correctly; determinant cancels after normalize.
    let m0 = model[0].xyz;
    let m1 = model[1].xyz;
    let m2 = model[2].xyz;
    let normal_mat = mat3x3<f32>(
        cross(m1, m2),
        cross(m2, m0),
        cross(m0, m1),
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
    let wt = normalize(normal_mat * in.tangent.xyz);
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

// Determine which cube face a direction vector points at and return (face_index, u, v).
fn cube_face_uv(dir: vec3<f32>) -> vec3<f32> {
    let abs_dir = abs(dir);
    var face: f32;
    var u: f32;
    var v: f32;

    if abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z {
        // +X (face 0) or -X (face 1)
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
        // +Y (face 2) or -Y (face 3)
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
        // +Z (face 4) or -Z (face 5)
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

// Sample point shadow for a single cubemap face. Returns 0.0–1.0.
// `biased_pos` should already have normal bias applied.
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

    // 2x2 PCF (4 taps — cheaper for 512px maps).
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

// Determine the cubemap face index for a direction vector.
// Face layout: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
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

// Determine the secondary (adjacent) cubemap face for cross-face blending.
// Returns the face that would be selected if the second-largest axis were dominant.
fn cube_secondary_face(dir: vec3<f32>) -> i32 {
    let a = abs(dir);
    // Find the second-largest component and return its face.
    var second_axis: i32;
    if a.x >= a.y && a.x >= a.z {
        // Primary is X; secondary is max(y, z).
        second_axis = select(2, 1, a.z > a.y); // 1=Y-axis, 2=Z-axis
    } else if a.y >= a.x && a.y >= a.z {
        // Primary is Y; secondary is max(x, z).
        second_axis = select(0, 2, a.z > a.x); // 0=X-axis, 2=Z-axis
    } else {
        // Primary is Z; secondary is max(x, y).
        second_axis = select(0, 1, a.y > a.x); // 0=X-axis, 1=Y-axis
    }
    // Map axis + sign to face index.
    if second_axis == 0 {
        return select(1, 0, dir.x > 0.0);
    } else if second_axis == 1 {
        return select(3, 2, dir.y > 0.0);
    } else {
        return select(5, 4, dir.z > 0.0);
    }
}

// Point light shadow factor. Returns 1.0 (fully lit) to 0.0 (shadowed).
// Applies normal bias to prevent acne at grazing angles, and blends between
// adjacent cubemap faces near face boundaries to eliminate seams.
fn point_shadow_factor(light_idx: i32, world_pos: vec3<f32>, light_pos: vec3<f32>, light_range: f32, normal: vec3<f32>) -> f32 {
    let point_shadow_count = i32(omni_shadow.omni_config.x);
    if light_idx >= point_shadow_count {
        return 1.0;
    }

    // Normal bias: offset along surface normal to prevent self-shadowing.
    // Scale by the point normal bias and by how much the surface faces away
    // from the light (grazing angles need more bias).
    let to_light = normalize(light_pos - world_pos);
    let n_dot_l = max(dot(normal, to_light), 0.0);
    let normal_bias = omni_shadow.omni_config.w;
    let bias_scale = max(1.0 - n_dot_l, 0.1);
    let biased_pos = world_pos + normal * normal_bias * bias_scale;

    let to_frag = biased_pos - light_pos;
    let a = abs(to_frag);

    let primary = cube_face_index(to_frag);
    let sf_primary = sample_point_shadow_face(light_idx, primary, biased_pos);

    // Compute how close we are to a cubemap face boundary.
    // max_comp is the dominant axis; mid_comp is the second-largest.
    let max_comp = max(max(a.x, a.y), a.z);
    let min_of_xy = min(a.x, a.y);
    let max_of_xy = max(a.x, a.y);
    let mid_comp = max(min(max_of_xy, a.z), min_of_xy);
    let edge_ratio = (max_comp - mid_comp) / max(max_comp, 0.001);

    // If well within one face (edge_ratio > 15%), skip the second sample.
    if edge_ratio > 0.15 {
        return sf_primary;
    }

    // Near a face boundary — sample the adjacent face and blend.
    let secondary = cube_secondary_face(to_frag);
    let sf_secondary = sample_point_shadow_face(light_idx, secondary, biased_pos);

    // Smooth blend: at edge_ratio=0 (exact boundary) → 50/50 mix;
    // at edge_ratio=0.15 → fully primary.
    let blend = smoothstep(0.0, 0.15, edge_ratio);
    return mix(sf_secondary, sf_primary, blend);
}

// Spot light shadow factor. Returns 1.0 (fully lit) to 0.0 (shadowed).
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

    // 3x3 PCF (9 taps).
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
