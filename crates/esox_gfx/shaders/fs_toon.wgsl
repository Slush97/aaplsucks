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

    // ── Ambient ──
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
