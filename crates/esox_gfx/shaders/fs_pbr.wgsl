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
            let atten = pl_intensity / max(dist * dist, 0.01);
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
            let d_atten = sl_intensity / max(dist * dist, 0.01);
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
