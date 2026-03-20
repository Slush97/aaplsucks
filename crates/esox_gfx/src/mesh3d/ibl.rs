//! Image-Based Lighting — environment mapping for PBR materials.
//!
//! Replaces flat ambient with diffuse irradiance + specular pre-filtered
//! reflections + BRDF LUT integration (split-sum approximation).
//!
//! # Overview
//!
//! [`IblState`] owns three GPU textures:
//! - **Irradiance cubemap** (32x32 per face, `Rgba16Float`) — diffuse hemisphere
//!   convolution of the environment map.
//! - **Prefiltered environment map** (128x128 per face, 5 mip levels, `Rgba16Float`) —
//!   specular GGX convolution at increasing roughness.
//! - **BRDF integration LUT** (256x256, `Rg16Float`) — (NdotV, roughness) to
//!   (scale, bias) for the Fresnel-Schlick split-sum approximation.
//!
//! All precomputation is CPU-side to avoid compute shader complexity.

use std::f32::consts::PI;

use glam::Vec3;

// ── Constants ──

/// Irradiance cubemap face resolution.
const IRRADIANCE_SIZE: u32 = 32;

/// Prefiltered environment map base face resolution.
const PREFILTERED_SIZE: u32 = 128;

/// Number of mip levels for the prefiltered environment map (roughness 0..1).
const PREFILTERED_MIP_LEVELS: u32 = 5;

/// BRDF LUT resolution (square).
const BRDF_LUT_SIZE: u32 = 256;

/// Number of samples for irradiance hemisphere convolution.
const IRRADIANCE_SAMPLES: u32 = 1024;

/// Number of importance samples for specular prefiltering per texel per mip.
const SPECULAR_SAMPLES: u32 = 256;

/// Number of importance samples for BRDF LUT integration.
const BRDF_SAMPLES: u32 = 1024;

// ── f32 -> f16 conversion ──

/// Convert an f32 to IEEE 754 half-precision (binary16) and return its two LE bytes.
///
/// Handles normals, denormals, infinities, and NaN. Rounds to nearest even.
fn f32_to_f16_bytes(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    let h = if exponent == 255 {
        // Inf or NaN
        if mantissa == 0 {
            sign | 0x7C00 // Inf
        } else {
            sign | 0x7C00 | (mantissa >> 13).max(1) // NaN (preserve some payload)
        }
    } else if exponent > 142 {
        // Overflow -> Inf
        sign | 0x7C00
    } else if exponent > 112 {
        // Normal range for f16
        let exp16 = ((exponent - 112) as u32) << 10;
        let man16 = mantissa >> 13;
        // Round to nearest even
        let round_bit = (mantissa >> 12) & 1;
        let sticky = mantissa & 0xFFF;
        let rounded = if round_bit == 1 && (sticky != 0 || (man16 & 1) == 1) {
            1u32
        } else {
            0
        };
        let result = sign | exp16 | man16 + rounded;
        // Handle carry into exponent
        result.min(sign | 0x7BFF) // Cap at max normal (not inf)
    } else if exponent > 101 {
        // Denormal range for f16
        let shift = (125 - exponent) + 13; // Total right-shift for mantissa
        if shift >= 32 {
            sign // Too small — flushes to signed zero
        } else {
            let man_full = mantissa | 0x0080_0000; // Restore implicit leading 1
            let man16 = man_full >> shift;
            sign | man16
        }
    } else {
        // Too small -> zero
        sign
    };

    (h as u16).to_le_bytes()
}

// ── IblState ──

/// Owns irradiance cubemap, prefiltered environment map, and BRDF integration LUT.
pub struct IblState {
    /// Diffuse irradiance cubemap (32x32 per face).
    /// Kept alive so the GPU texture isn't deallocated while the view is in use.
    #[allow(dead_code)]
    pub(crate) irradiance_texture: wgpu::Texture,
    pub(crate) irradiance_view: wgpu::TextureView,
    /// Specular prefiltered environment map (128x128 per face, 5 mip levels).
    #[allow(dead_code)]
    pub(crate) prefiltered_texture: wgpu::Texture,
    pub(crate) prefiltered_view: wgpu::TextureView,
    /// BRDF integration LUT (256x256, Rg16Float).
    #[allow(dead_code)]
    pub(crate) brdf_lut_texture: wgpu::Texture,
    pub(crate) brdf_lut_view: wgpu::TextureView,
}

impl IblState {
    /// Create fallback 1x1 cubemaps that produce the same result as flat white ambient.
    ///
    /// Use this when no environment map is loaded — the renderer sees valid IBL textures
    /// that behave identically to the old constant ambient path.
    pub fn fallback(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // 1x1 white cubemap for irradiance (Rgba16Float: 8 bytes per texel, 6 faces)
        let white_f16 = f32_to_f16_bytes(1.0);
        let mut white_faces = Vec::with_capacity(6 * 8);
        for _ in 0..6 {
            // R, G, B, A — each as f16
            for _ in 0..4 {
                white_faces.extend_from_slice(&white_f16);
            }
        }

        let (irradiance_texture, irradiance_view) = create_cubemap_with_data(
            device,
            queue,
            1,
            1,
            wgpu::TextureFormat::Rgba16Float,
            "esox_ibl_irradiance_fallback",
            &white_faces,
        );

        // 1x1 white cubemap for prefiltered (single mip)
        let (prefiltered_texture, prefiltered_view) = create_cubemap_with_data(
            device,
            queue,
            1,
            1,
            wgpu::TextureFormat::Rgba16Float,
            "esox_ibl_prefiltered_fallback",
            &white_faces,
        );

        // Full-size BRDF LUT (still needed for correct split-sum math)
        let (brdf_lut_texture, brdf_lut_view) = generate_brdf_lut(device, queue, BRDF_LUT_SIZE);

        Self {
            irradiance_texture,
            irradiance_view,
            prefiltered_texture,
            prefiltered_view,
            brdf_lut_texture,
            brdf_lut_view,
        }
    }

    /// Generate IBL textures from a procedural sky environment.
    ///
    /// Creates a sky gradient with a sun disc and glow, then runs the full IBL
    /// pipeline (equirect → cubemap → irradiance + specular prefiltering + BRDF LUT).
    pub fn from_procedural_sky(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sun_dir: Vec3,
        sun_color: Vec3,
        sun_intensity: f32,
        sky_color: Vec3,
        ground_color: Vec3,
    ) -> Self {
        let (data, w, h) = generate_procedural_sky_equirect(
            sun_dir, sun_color, sun_intensity, sky_color, ground_color,
        );
        Self::from_equirect(device, queue, &data, w, h).unwrap()
    }

    /// Generate IBL textures from equirectangular HDR image data.
    ///
    /// `hdr_data` is a flat array of f32 RGB pixels, row-major, `width * height * 3` elements.
    /// Returns `Err` if the data length doesn't match `width * height * 3`.
    pub fn from_equirect(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hdr_data: &[f32],
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize) * 3;
        if hdr_data.len() != expected {
            return Err(format!(
                "HDR data length mismatch: expected {expected} floats ({}x{}x3), got {}",
                width,
                height,
                hdr_data.len(),
            ));
        }

        // Step 1: equirect -> source cubemap faces (128x128 each, f32 RGBA)
        let source_size = PREFILTERED_SIZE;
        let source_faces = equirect_to_cubemap_faces(hdr_data, width, height, source_size);

        // Step 2: diffuse irradiance convolution (32x32)
        let irradiance_faces =
            convolve_irradiance(&source_faces, source_size, IRRADIANCE_SIZE);
        let irradiance_bytes = faces_f32_to_f16_bytes(&irradiance_faces, IRRADIANCE_SIZE);
        let (irradiance_texture, irradiance_view) = create_cubemap_with_data(
            device,
            queue,
            IRRADIANCE_SIZE,
            1,
            wgpu::TextureFormat::Rgba16Float,
            "esox_ibl_irradiance",
            &irradiance_bytes,
        );

        // Step 3: specular prefiltering (128x128, 5 mip levels)
        let (prefiltered_texture, prefiltered_view) = create_prefiltered_env_map(
            device,
            queue,
            &source_faces,
            source_size,
        );

        // Step 4: BRDF LUT
        let (brdf_lut_texture, brdf_lut_view) = generate_brdf_lut(device, queue, BRDF_LUT_SIZE);

        Ok(Self {
            irradiance_texture,
            irradiance_view,
            prefiltered_texture,
            prefiltered_view,
            brdf_lut_texture,
            brdf_lut_view,
        })
    }
}

// ── Procedural sky ──

/// Generate a procedural sky environment as equirectangular f32 RGB data.
///
/// Returns `(data, width, height)` where `data` is `width * height * 3` floats.
/// The sky is a simple gradient with a sun disc and glow, suitable for IBL.
pub fn generate_procedural_sky_equirect(
    sun_dir: Vec3,
    sun_color: Vec3,
    sun_intensity: f32,
    sky_color: Vec3,
    ground_color: Vec3,
) -> (Vec<f32>, u32, u32) {
    let width = 256u32;
    let height = 128u32;
    let sun_dir = sun_dir.normalize();
    let horizon_color = sky_color * 0.5 + sun_color * 0.2;

    let mut data = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        let phi = (v - 0.5) * PI;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let theta = (u - 0.5) * 2.0 * PI;

            let dir = Vec3::new(
                cos_phi * theta.cos(),
                sin_phi,
                cos_phi * theta.sin(),
            );

            // Sky gradient
            let up_factor = dir.y;
            let base = if up_factor >= 0.0 {
                // Upper hemisphere: horizon → sky
                lerp_vec3(horizon_color, sky_color, up_factor)
            } else {
                // Lower hemisphere: horizon → ground
                lerp_vec3(horizon_color, ground_color, up_factor.abs())
            };

            // Sun disc and glow
            let cos_angle = dir.dot(sun_dir).max(0.0);
            let mut pixel = base;

            // Sun glow: wider warm halo
            pixel += sun_color * 0.5 * cos_angle.powf(8.0);

            // Sun disc: tight bright spot
            if cos_angle > 0.995 {
                let t = smoothstep(0.995, 1.0, cos_angle);
                pixel += sun_color * sun_intensity * t;
            }

            data.push(pixel.x.max(0.0));
            data.push(pixel.y.max(0.0));
            data.push(pixel.z.max(0.0));
        }
    }

    (data, width, height)
}

fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Cubemap helpers ──

/// Direction vectors for the six cubemap faces.
///
/// Each face is defined by (right, up, forward) basis where forward points into the face.
/// Layout: +X, -X, +Y, -Y, +Z, -Z.
fn cubemap_face_bases() -> [(Vec3, Vec3, Vec3); 6] {
    [
        // +X
        (Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        // -X
        (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
        // +Y
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0)),
        // -Y
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, -1.0, 0.0)),
        // +Z
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        // -Z
        (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, -1.0)),
    ]
}

/// Compute the 3D direction for a texel on a cubemap face.
///
/// `(u, v)` in [0, face_size), `face` in [0, 6).
fn cubemap_texel_direction(face: usize, u: u32, v: u32, face_size: u32) -> Vec3 {
    let bases = cubemap_face_bases();
    let (right, up, forward) = bases[face];
    // Map texel to [-1, 1] with half-texel offset for center sampling.
    let uf = (2.0 * (u as f32 + 0.5) / face_size as f32) - 1.0;
    let vf = (2.0 * (v as f32 + 0.5) / face_size as f32) - 1.0;
    (forward + right * uf + up * vf).normalize()
}

/// Convert a 3D direction to equirectangular UV coordinates.
///
/// Returns `(u, v)` in [0, 1].
pub(crate) fn direction_to_equirect_uv(dir: Vec3) -> (f32, f32) {
    let u = dir.z.atan2(dir.x) / (2.0 * PI) + 0.5;
    let v = dir.y.asin() / PI + 0.5;
    (u, v)
}

/// Sample the equirectangular HDR image at a direction. Returns RGB.
fn sample_equirect(hdr_data: &[f32], width: u32, height: u32, dir: Vec3) -> [f32; 3] {
    let (u, v) = direction_to_equirect_uv(dir);
    // Bilinear coordinates
    let fx = u * width as f32 - 0.5;
    let fy = v * height as f32 - 0.5;
    let x0 = (fx.floor() as i32).rem_euclid(width as i32) as u32;
    let y0 = fy.floor().clamp(0.0, (height - 1) as f32) as u32;
    let x1 = (x0 + 1) % width;
    let y1 = (y0 + 1).min(height - 1);
    let sx = fx.fract().max(0.0);
    let sy = fy.fract().max(0.0);

    let sample = |x: u32, y: u32| -> [f32; 3] {
        let idx = ((y * width + x) * 3) as usize;
        [hdr_data[idx], hdr_data[idx + 1], hdr_data[idx + 2]]
    };

    let s00 = sample(x0, y0);
    let s10 = sample(x1, y0);
    let s01 = sample(x0, y1);
    let s11 = sample(x1, y1);

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    [
        lerp(lerp(s00[0], s10[0], sx), lerp(s01[0], s11[0], sx), sy),
        lerp(lerp(s00[1], s10[1], sx), lerp(s01[1], s11[1], sx), sy),
        lerp(lerp(s00[2], s10[2], sx), lerp(s01[2], s11[2], sx), sy),
    ]
}

/// Sample a cubemap (6 faces, RGBA f32) at a direction. Returns RGB.
fn sample_cubemap(faces: &[[f32; 4]], face_size: u32, dir: Vec3) -> [f32; 3] {
    let abs = dir.abs();
    let (face, uc, vc) = if abs.x >= abs.y && abs.x >= abs.z {
        if dir.x > 0.0 {
            (0, -dir.z / abs.x, -dir.y / abs.x) // +X
        } else {
            (1, dir.z / abs.x, -dir.y / abs.x) // -X
        }
    } else if abs.y >= abs.x && abs.y >= abs.z {
        if dir.y > 0.0 {
            (2, dir.x / abs.y, dir.z / abs.y) // +Y
        } else {
            (3, dir.x / abs.y, -dir.z / abs.y) // -Y
        }
    } else if dir.z > 0.0 {
        (4, dir.x / abs.z, -dir.y / abs.z) // +Z
    } else {
        (5, -dir.x / abs.z, -dir.y / abs.z) // -Z
    };

    let u = ((uc * 0.5 + 0.5) * face_size as f32).clamp(0.0, (face_size - 1) as f32) as u32;
    let v = ((vc * 0.5 + 0.5) * face_size as f32).clamp(0.0, (face_size - 1) as f32) as u32;
    let texels_per_face = (face_size * face_size) as usize;
    let idx = face * texels_per_face + (v * face_size + u) as usize;
    let px = faces[idx];
    [px[0], px[1], px[2]]
}

/// Convert equirectangular HDR data to 6 cubemap faces (RGBA f32).
///
/// Returns a flat Vec of `[f32; 4]` with `6 * face_size * face_size` entries.
fn equirect_to_cubemap_faces(
    hdr_data: &[f32],
    width: u32,
    height: u32,
    face_size: u32,
) -> Vec<[f32; 4]> {
    let texels_per_face = (face_size * face_size) as usize;
    let mut faces = vec![[0.0f32; 4]; 6 * texels_per_face];

    for face in 0..6 {
        for v in 0..face_size {
            for u in 0..face_size {
                let dir = cubemap_texel_direction(face, u, v, face_size);
                let rgb = sample_equirect(hdr_data, width, height, dir);
                let idx = face * texels_per_face + (v * face_size + u) as usize;
                faces[idx] = [rgb[0], rgb[1], rgb[2], 1.0];
            }
        }
    }

    faces
}

// ── Irradiance convolution ──

/// Convolve source cubemap into a diffuse irradiance cubemap.
///
/// For each texel of the output, integrates over the hemisphere weighted by
/// `max(dot(n, l), 0)` using uniform sampling with a Hammersley sequence.
fn convolve_irradiance(
    source_faces: &[[f32; 4]],
    source_size: u32,
    output_size: u32,
) -> Vec<[f32; 4]> {
    let texels_per_face = (output_size * output_size) as usize;
    let mut result = vec![[0.0f32; 4]; 6 * texels_per_face];

    for face in 0..6 {
        for v in 0..output_size {
            for u in 0..output_size {
                let normal = cubemap_texel_direction(face, u, v, output_size);
                let (tangent, bitangent) = build_tangent_frame(normal);

                let mut accum = [0.0f32; 3];
                let mut weight = 0.0f32;

                for i in 0..IRRADIANCE_SAMPLES {
                    let xi = hammersley(i, IRRADIANCE_SAMPLES);
                    // Cosine-weighted hemisphere sampling
                    let phi = 2.0 * PI * xi.0;
                    let cos_theta = xi.1;
                    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

                    // Tangent-space direction
                    let ts = Vec3::new(
                        sin_theta * phi.cos(),
                        sin_theta * phi.sin(),
                        cos_theta,
                    );
                    // World-space direction
                    let sample_dir = tangent * ts.x + bitangent * ts.y + normal * ts.z;

                    let n_dot_l = normal.dot(sample_dir).max(0.0);
                    if n_dot_l > 0.0 {
                        let rgb = sample_cubemap(source_faces, source_size, sample_dir);
                        accum[0] += rgb[0] * n_dot_l;
                        accum[1] += rgb[1] * n_dot_l;
                        accum[2] += rgb[2] * n_dot_l;
                        weight += n_dot_l;
                    }
                }

                if weight > 0.0 {
                    accum[0] /= weight;
                    accum[1] /= weight;
                    accum[2] /= weight;
                }

                let idx = face * texels_per_face + (v * output_size + u) as usize;
                result[idx] = [accum[0], accum[1], accum[2], 1.0];
            }
        }
    }

    result
}

// ── Specular prefiltering ──

/// Create the prefiltered environment map with 5 mip levels of increasing roughness.
fn create_prefiltered_env_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source_faces: &[[f32; 4]],
    source_size: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let format = wgpu::TextureFormat::Rgba16Float;
    let bytes_per_texel = 8u32; // Rgba16Float = 4 channels * 2 bytes

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("esox_ibl_prefiltered"),
        size: wgpu::Extent3d {
            width: PREFILTERED_SIZE,
            height: PREFILTERED_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: PREFILTERED_MIP_LEVELS,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for mip in 0..PREFILTERED_MIP_LEVELS {
        let mip_size = PREFILTERED_SIZE >> mip;
        let roughness = mip as f32 / (PREFILTERED_MIP_LEVELS - 1) as f32;
        let face_texels = (mip_size * mip_size) as usize;

        let faces_f32 =
            prefilter_cubemap_mip(source_faces, source_size, mip_size, roughness);

        // Convert f32 RGBA -> f16 bytes
        let mut data = Vec::with_capacity(6 * face_texels * bytes_per_texel as usize);
        for px in &faces_f32 {
            for &c in px {
                data.extend_from_slice(&f32_to_f16_bytes(c));
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_texel * mip_size),
                rows_per_image: Some(mip_size),
            },
            wgpu::Extent3d {
                width: mip_size,
                height: mip_size,
                depth_or_array_layers: 6,
            },
        );
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    (texture, view)
}

/// Prefilter the source cubemap at a single roughness level.
///
/// Uses GGX importance sampling to average the environment in the specular lobe.
fn prefilter_cubemap_mip(
    source_faces: &[[f32; 4]],
    source_size: u32,
    output_size: u32,
    roughness: f32,
) -> Vec<[f32; 4]> {
    let texels_per_face = (output_size * output_size) as usize;
    let mut result = vec![[0.0f32; 4]; 6 * texels_per_face];

    // For roughness ~0, just copy the source (mirror reflection).
    let sample_count = if roughness < 0.001 {
        1u32
    } else {
        SPECULAR_SAMPLES
    };

    for face in 0..6 {
        for v in 0..output_size {
            for u in 0..output_size {
                let normal = cubemap_texel_direction(face, u, v, output_size);
                let view = normal; // Assume V = N (common approximation for prefiltering)
                let (tangent, bitangent) = build_tangent_frame(normal);

                let mut accum = [0.0f32; 3];
                let mut weight = 0.0f32;

                for i in 0..sample_count {
                    let xi = hammersley(i, sample_count);
                    let half_vec =
                        importance_sample_ggx(xi, roughness, &tangent, &bitangent, &normal);
                    let light = (2.0 * view.dot(half_vec) * half_vec - view).normalize();
                    let n_dot_l = normal.dot(light).max(0.0);

                    if n_dot_l > 0.0 {
                        let rgb = sample_cubemap(source_faces, source_size, light);
                        accum[0] += rgb[0] * n_dot_l;
                        accum[1] += rgb[1] * n_dot_l;
                        accum[2] += rgb[2] * n_dot_l;
                        weight += n_dot_l;
                    }
                }

                if weight > 0.0 {
                    accum[0] /= weight;
                    accum[1] /= weight;
                    accum[2] /= weight;
                }

                let idx = face * texels_per_face + (v * output_size + u) as usize;
                result[idx] = [accum[0], accum[1], accum[2], 1.0];
            }
        }
    }

    result
}

// ── BRDF LUT generation ──

/// Generate the BRDF integration LUT.
///
/// Each texel `(x, y)` maps to `(NdotV, roughness)` and stores `(scale, bias)`
/// for the split-sum Fresnel-Schlick approximation as `Rg16Float`.
fn generate_brdf_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texel_count = (size * size) as usize;
    let mut data = Vec::with_capacity(texel_count * 4); // 4 bytes per texel (2x f16)

    for y in 0..size {
        let roughness = (y as f32 + 0.5) / size as f32;
        for x in 0..size {
            let n_dot_v = (x as f32 + 0.5) / size as f32;
            let n_dot_v = n_dot_v.max(0.001); // Avoid degenerate case at 0

            let (scale, bias) = integrate_brdf(n_dot_v, roughness);

            data.extend_from_slice(&f32_to_f16_bytes(scale));
            data.extend_from_slice(&f32_to_f16_bytes(bias));
        }
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("esox_ibl_brdf_lut"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * size), // 4 bytes per texel (Rg16Float)
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    (texture, view)
}

/// Integrate the BRDF for a single (NdotV, roughness) pair.
///
/// Returns (scale, bias) where `F = F0 * scale + bias`.
fn integrate_brdf(n_dot_v: f32, roughness: f32) -> (f32, f32) {
    let v = Vec3::new((1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), 0.0, n_dot_v);
    let n = Vec3::Z;
    let _ = n; // Normal is implicitly Z in tangent space.

    let mut scale = 0.0f32;
    let mut bias = 0.0f32;

    for i in 0..BRDF_SAMPLES {
        let xi = hammersley(i, BRDF_SAMPLES);
        let h = importance_sample_ggx_tangent(xi, roughness);
        let l = (2.0 * v.dot(h) * h - v).normalize();

        let n_dot_l = l.z.max(0.0);
        let n_dot_h = h.z.max(0.0);
        let v_dot_h = v.dot(h).max(0.0);

        if n_dot_l > 0.0 {
            let g = geometry_smith(n_dot_v, n_dot_l, roughness);
            let g_vis = (g * v_dot_h) / (n_dot_h * n_dot_v).max(0.001);
            let fc = (1.0 - v_dot_h).powi(5);

            scale += g_vis * (1.0 - fc);
            bias += g_vis * fc;
        }
    }

    (
        (scale / BRDF_SAMPLES as f32).clamp(0.0, 1.0),
        (bias / BRDF_SAMPLES as f32).clamp(0.0, 1.0),
    )
}

// ── Sampling utilities ──

/// Hammersley low-discrepancy sequence point.
///
/// Returns `(xi1, xi2)` in [0, 1)^2 for sample `i` out of `n`.
fn hammersley(i: u32, n: u32) -> (f32, f32) {
    (i as f32 / n as f32, radical_inverse_vdc(i))
}

/// Van der Corput radical inverse (base 2) via bit manipulation.
fn radical_inverse_vdc(mut bits: u32) -> f32 {
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
    bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
    bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
    bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
    bits as f32 * 2.328_306_4e-10 // 1.0 / 0x100000000
}

/// Importance sample the GGX distribution in tangent space.
///
/// Returns a half-vector `H` in tangent space aligned with Z-up normal.
fn importance_sample_ggx_tangent(xi: (f32, f32), roughness: f32) -> Vec3 {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.0;
    let cos_theta = ((1.0 - xi.1) / (1.0 + (a * a - 1.0) * xi.1)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

    Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

/// Importance sample the GGX distribution, producing a world-space half-vector.
fn importance_sample_ggx(
    xi: (f32, f32),
    roughness: f32,
    tangent: &Vec3,
    bitangent: &Vec3,
    normal: &Vec3,
) -> Vec3 {
    let ts = importance_sample_ggx_tangent(xi, roughness);
    (*tangent * ts.x + *bitangent * ts.y + *normal * ts.z).normalize()
}

/// Smith's geometry function for GGX (Schlick-GGX formulation, IBL variant).
///
/// Uses `k = roughness^2 / 2` (IBL-specific remapping).
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness * roughness) / 2.0;
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    g1_v * g1_l
}

/// Build an orthonormal tangent frame from a normal vector.
///
/// Returns `(tangent, bitangent)` such that `(tangent, bitangent, normal)` form
/// a right-handed basis.
fn build_tangent_frame(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.y.abs() < 0.999 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = up.cross(normal).normalize();
    let bitangent = normal.cross(tangent);
    (tangent, bitangent)
}

// ── Texture creation helpers ──

/// Create a cubemap texture and upload data for all 6 faces at mip 0.
fn create_cubemap_with_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    face_size: u32,
    mip_levels: u32,
    format: wgpu::TextureFormat,
    label: &str,
    data: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let bytes_per_texel = match format {
        wgpu::TextureFormat::Rgba16Float => 8u32,
        wgpu::TextureFormat::Rg16Float => 4u32,
        _ => 4u32,
    };

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_texel * face_size),
            rows_per_image: Some(face_size),
        },
        wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    (texture, view)
}

/// Convert `[f32; 4]` face data to `Rgba16Float` byte representation.
fn faces_f32_to_f16_bytes(faces: &[[f32; 4]], face_size: u32) -> Vec<u8> {
    let texels_per_face = (face_size * face_size) as usize;
    let total_texels = 6 * texels_per_face;
    let mut bytes = Vec::with_capacity(total_texels * 8); // 8 bytes per Rgba16Float texel
    for px in faces.iter().take(total_texels) {
        for &c in px {
            bytes.extend_from_slice(&f32_to_f16_bytes(c));
        }
    }
    bytes
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brdf_lut_dimensions() {
        // The constant that governs the LUT size.
        assert_eq!(BRDF_LUT_SIZE, 256);
    }

    #[test]
    fn equirect_to_dir_roundtrip() {
        // Test several directions survive the direction -> UV -> direction round-trip.
        let test_dirs = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(-0.5, 0.3, 0.8).normalize(),
        ];

        for original in &test_dirs {
            let (u, v) = direction_to_equirect_uv(*original);

            // UV should be in [0, 1]
            assert!(
                (0.0..=1.0).contains(&u),
                "u={u} out of range for dir={original}"
            );
            assert!(
                (0.0..=1.0).contains(&v),
                "v={v} out of range for dir={original}"
            );

            // Reconstruct direction from UV (inverse mapping)
            let phi = (u - 0.5) * 2.0 * PI;
            let theta = (v - 0.5) * PI;
            let reconstructed = Vec3::new(
                theta.cos() * phi.cos(),
                theta.sin(),
                theta.cos() * phi.sin(),
            )
            .normalize();

            let dot = original.dot(reconstructed);
            assert!(
                dot > 0.99,
                "round-trip failed: original={original}, reconstructed={reconstructed}, dot={dot}"
            );
        }
    }

    #[test]
    fn hammersley_range() {
        for n in [16, 64, 256, 1024] {
            for i in 0..n {
                let (xi1, xi2) = hammersley(i, n);
                assert!((0.0..1.0).contains(&xi1), "xi1={xi1} out of range");
                assert!((0.0..1.0).contains(&xi2), "xi2={xi2} out of range");
            }
        }
    }

    #[test]
    fn radical_inverse_vdc_values() {
        assert!((radical_inverse_vdc(0) - 0.0).abs() < 1e-6);
        assert!((radical_inverse_vdc(1) - 0.5).abs() < 1e-6);
        assert!((radical_inverse_vdc(2) - 0.25).abs() < 1e-6);
        assert!((radical_inverse_vdc(3) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn importance_sample_ggx_tangent_unit_length() {
        for i in 0..64 {
            let xi = hammersley(i, 64);
            let h = importance_sample_ggx_tangent(xi, 0.5);
            let len = h.length();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "half-vector not unit: len={len}"
            );
        }
    }

    #[test]
    fn importance_sample_ggx_tangent_hemisphere() {
        // All samples should have positive z (upper hemisphere)
        for i in 0..256 {
            let xi = hammersley(i, 256);
            let h = importance_sample_ggx_tangent(xi, 0.3);
            assert!(h.z >= 0.0, "sample below hemisphere: z={}", h.z);
        }
    }

    #[test]
    fn geometry_smith_boundary() {
        // At roughness = 0, geometry should be ~1 (no shadowing)
        let g = geometry_smith(1.0, 1.0, 0.0);
        assert!((g - 1.0).abs() < 1e-6, "G(1,1,0)={g} expected ~1");

        // At roughness = 1, should still be positive
        let g = geometry_smith(0.5, 0.5, 1.0);
        assert!(g > 0.0, "G should be positive at roughness=1");
        assert!(g < 1.0, "G should be <1 at roughness=1");
    }

    #[test]
    fn integrate_brdf_values() {
        // At NdotV=1, roughness~0 (perfect mirror, head-on), scale should be ~1, bias ~0
        let (scale, bias) = integrate_brdf(1.0, 0.01);
        assert!(
            scale > 0.9,
            "scale={scale} expected >0.9 at NdotV=1, roughness~0"
        );
        assert!(
            bias < 0.1,
            "bias={bias} expected <0.1 at NdotV=1, roughness~0"
        );

        // scale + bias should be <= 1 (energy conservation)
        for &ndv in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            for &r in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                let (s, b) = integrate_brdf(ndv, r);
                assert!(
                    s + b <= 1.05,
                    "energy violation: scale={s} + bias={b} > 1 at NdotV={ndv}, roughness={r}"
                );
            }
        }
    }

    #[test]
    fn build_tangent_frame_orthogonal() {
        let normals = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec3::new(-0.3, 0.7, 0.5).normalize(),
        ];
        for n in &normals {
            let (t, b) = build_tangent_frame(*n);
            // Orthogonality
            assert!(
                n.dot(t).abs() < 1e-5,
                "T not orthogonal to N: dot={}, N={n}, T={t}",
                n.dot(t)
            );
            assert!(
                n.dot(b).abs() < 1e-5,
                "B not orthogonal to N: dot={}, N={n}, B={b}",
                n.dot(b)
            );
            assert!(
                t.dot(b).abs() < 1e-5,
                "T not orthogonal to B: dot={}, T={t}, B={b}",
                t.dot(b)
            );
            // Unit length
            assert!((t.length() - 1.0).abs() < 1e-5, "T not unit: len={}", t.length());
            assert!((b.length() - 1.0).abs() < 1e-5, "B not unit: len={}", b.length());
        }
    }

    #[test]
    fn cubemap_texel_direction_unit_length() {
        for face in 0..6 {
            for v in 0..4 {
                for u in 0..4 {
                    let dir = cubemap_texel_direction(face, u, v, 4);
                    let len = dir.length();
                    assert!(
                        (len - 1.0).abs() < 1e-5,
                        "direction not unit: face={face}, u={u}, v={v}, len={len}"
                    );
                }
            }
        }
    }

    #[test]
    fn faces_f32_to_f16_bytes_length() {
        let face_size = 4u32;
        let texels = 6 * (face_size * face_size) as usize;
        let faces: Vec<[f32; 4]> = vec![[1.0, 0.5, 0.25, 1.0]; texels];
        let bytes = faces_f32_to_f16_bytes(&faces, face_size);
        assert_eq!(bytes.len(), texels * 8); // 8 bytes per Rgba16Float
    }

    #[test]
    fn equirect_to_cubemap_face_count() {
        // 2x1 equirect (minimal)
        let hdr = vec![1.0f32; 2 * 1 * 3];
        let face_size = 2u32;
        let faces = equirect_to_cubemap_faces(&hdr, 2, 1, face_size);
        assert_eq!(faces.len(), 6 * (face_size * face_size) as usize);
    }

    #[test]
    fn f32_to_f16_known_values() {
        // Zero
        assert_eq!(f32_to_f16_bytes(0.0), [0x00, 0x00]);
        // One: sign=0, exp=15(0b01111=0xF), mantissa=0 -> 0x3C00
        assert_eq!(f32_to_f16_bytes(1.0), [0x00, 0x3C]);
        // Negative zero
        assert_eq!(f32_to_f16_bytes(-0.0), [0x00, 0x80]);
        // Infinity
        assert_eq!(f32_to_f16_bytes(f32::INFINITY), [0x00, 0x7C]);
        // Negative infinity
        assert_eq!(f32_to_f16_bytes(f32::NEG_INFINITY), [0x00, 0xFC]);
        // 0.5: sign=0, exp=14(0b01110), mantissa=0 -> 0x3800
        assert_eq!(f32_to_f16_bytes(0.5), [0x00, 0x38]);
        // Tiny denormal values must not panic (shift overflow regression).
        let _ = f32_to_f16_bytes(1.0e-20);
        let _ = f32_to_f16_bytes(5.0e-42);
    }

    #[test]
    fn procedural_sky_dimensions() {
        let (data, w, h) = generate_procedural_sky_equirect(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.95, 0.85),
            2.0,
            Vec3::new(0.4, 0.6, 1.0),
            Vec3::new(0.15, 0.12, 0.1),
        );
        assert_eq!(w, 256);
        assert_eq!(h, 128);
        assert_eq!(data.len(), (256 * 128 * 3) as usize);
    }

    #[test]
    fn procedural_sky_sun_brighter_than_sky() {
        // Sun pointing straight up — zenith pixel should be brighter than horizon.
        let (data, w, h) = generate_procedural_sky_equirect(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.95, 0.85),
            5.0,
            Vec3::new(0.4, 0.6, 1.0),
            Vec3::new(0.15, 0.12, 0.1),
        );
        // Zenith is at v=1.0 → y = h-1, any x.
        let zenith_idx = ((h - 1) * w * 3) as usize;
        let zenith_lum = data[zenith_idx] + data[zenith_idx + 1] + data[zenith_idx + 2];
        // Horizon is at v=0.5 → y = h/2.
        let horizon_idx = ((h / 2) * w * 3) as usize;
        let horizon_lum = data[horizon_idx] + data[horizon_idx + 1] + data[horizon_idx + 2];
        assert!(
            zenith_lum > horizon_lum,
            "zenith ({zenith_lum}) should be brighter than horizon ({horizon_lum})"
        );
    }

    #[test]
    fn procedural_sky_all_positive() {
        let (data, _, _) = generate_procedural_sky_equirect(
            Vec3::new(-0.5, -1.0, -0.3).normalize(),
            Vec3::new(1.0, 0.95, 0.85),
            2.0,
            Vec3::new(0.4, 0.6, 1.0),
            Vec3::new(0.15, 0.12, 0.1),
        );
        for (i, &v) in data.iter().enumerate() {
            assert!(v >= 0.0, "negative value at index {i}: {v}");
        }
    }

    #[test]
    fn f32_to_f16_roundtrip_accuracy() {
        // Values representable in f16 should survive the round-trip with minimal error.
        let test_values = [0.0, 0.25, 0.5, 1.0, 2.0, 0.001, 100.0, -1.0, -0.5];
        for &val in &test_values {
            let bytes = f32_to_f16_bytes(val);
            let h = u16::from_le_bytes(bytes);
            // Decode f16 back to f32 for comparison
            let sign = ((h >> 15) & 1) as u32;
            let exp = ((h >> 10) & 0x1F) as i32;
            let man = (h & 0x3FF) as u32;
            let decoded = if exp == 0 {
                if man == 0 {
                    if sign == 1 { -0.0 } else { 0.0 }
                } else {
                    let s = if sign == 1 { -1.0 } else { 1.0 };
                    s * (man as f32 / 1024.0) * 2.0f32.powi(-14)
                }
            } else if exp == 31 {
                if man == 0 {
                    if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
                } else {
                    f32::NAN
                }
            } else {
                let s = if sign == 1 { -1.0 } else { 1.0 };
                s * (1.0 + man as f32 / 1024.0) * 2.0f32.powi(exp - 15)
            };

            if val == 0.0 {
                assert!(decoded == 0.0, "0.0 roundtrip failed: got {decoded}");
            } else {
                let rel_err = ((decoded - val) / val).abs();
                assert!(
                    rel_err < 0.002,
                    "f16 roundtrip for {val}: decoded={decoded}, rel_err={rel_err}"
                );
            }
        }
    }
}
