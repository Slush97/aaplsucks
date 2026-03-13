struct Uniforms {
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,
    time: vec4<f32>,
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

// Group 0: Scene uniforms + shadow.
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(2) var shadow_depth: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;

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
    out.view_depth = length(world_pos.xyz - uniforms.camera_position.xyz);

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
