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
            let atten = pl_intensity / max(dist * dist, 0.01);
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
            let dist_atten = sl_intensity / max(dist * dist, 0.01);
            let sl_ndotl = max(dot(n, l), 0.0);
            let ssf = spot_shadow_factor(i, in.world_position);
            diffuse = diffuse + sl_color * dist_atten * spot_atten * sl_ndotl * ssf;
        }
    }

    let lit = base.rgb * (ambient + diffuse) + emissive;
    return vec4<f32>(lit, base.a);
}
