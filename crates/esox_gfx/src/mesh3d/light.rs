//! Lighting — ambient, directional, point, and spot lights for 3D rendering.

/// Maximum number of point lights supported in a single frame.
pub const MAX_POINT_LIGHTS: usize = 8;

/// Maximum number of spot lights supported in a single frame.
pub const MAX_SPOT_LIGHTS: usize = 4;

/// A directional light (infinite distance, like the sun).
#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    /// Light direction (will be normalized in the shader).
    pub direction: [f32; 3],
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
}

/// A point light with position, color, intensity, and range.
#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    /// World-space position.
    pub position: [f32; 3],
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
    /// Maximum range (attenuation cutoff distance).
    pub range: f32,
    /// Whether this light casts shadows.
    pub cast_shadows: bool,
}

/// A spot light with position, direction, cone angles, color, intensity, and range.
#[derive(Debug, Clone, Copy)]
pub struct SpotLight {
    /// World-space position.
    pub position: [f32; 3],
    /// Light direction (will be normalized).
    pub direction: [f32; 3],
    /// Light color (linear RGB).
    pub color: [f32; 3],
    /// Intensity multiplier.
    pub intensity: f32,
    /// Maximum range (attenuation cutoff distance).
    pub range: f32,
    /// Inner cone angle in radians (full intensity within this cone).
    pub inner_cone_angle: f32,
    /// Outer cone angle in radians (falloff from inner to outer).
    pub outer_cone_angle: f32,
    /// Whether this light casts shadows.
    pub cast_shadows: bool,
}

/// GPU-packed point light data (32 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PointLightGpu {
    /// xyz = position, w = range.
    pub position_range: [f32; 4],
    /// xyz = color, w = intensity.
    pub color_intensity: [f32; 4],
}

/// GPU-packed spot light data (64 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SpotLightGpu {
    /// xyz = position, w = range.
    pub position_range: [f32; 4],
    /// xyz = direction (normalized), w = cos(inner_cone_angle).
    pub direction_inner: [f32; 4],
    /// xyz = color, w = intensity.
    pub color_intensity: [f32; 4],
    /// x = cos(outer_cone_angle), yzw = padding.
    pub outer_pad: [f32; 4],
}

/// GPU uniform block for all lights (576 bytes, 16-byte aligned).
///
/// Layout: ambient(16) + dir_dir_intensity(16) + dir_color_count(16) +
/// spot_count_pad(16) + point_lights(256) + spot_lights(256) = 576
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct LightUniforms {
    /// Ambient light: rgb + intensity.
    pub ambient: [f32; 4],
    /// Directional light direction (xyz) + intensity (w).
    pub directional_dir_intensity: [f32; 4],
    /// Directional light color (rgb) + point light count (w, as f32).
    pub directional_color_count: [f32; 4],
    /// x = spot light count (as f32), yzw = padding.
    pub spot_count_pad: [f32; 4],
    /// Point lights array.
    pub point_lights: [PointLightGpu; MAX_POINT_LIGHTS],
    /// Spot lights array.
    pub spot_lights: [SpotLightGpu; MAX_SPOT_LIGHTS],
}

/// High-level light configuration for a frame.
#[derive(Debug, Clone)]
pub struct LightEnvironment {
    /// Ambient light color (linear RGB).
    pub ambient_color: [f32; 3],
    /// Ambient light intensity.
    pub ambient_intensity: f32,
    /// Primary directional light.
    pub directional: DirectionalLight,
    /// Point lights (capped at [`MAX_POINT_LIGHTS`]).
    pub point_lights: Vec<PointLight>,
    /// Spot lights (capped at [`MAX_SPOT_LIGHTS`]).
    pub spot_lights: Vec<SpotLight>,
}

impl Default for LightEnvironment {
    fn default() -> Self {
        Self {
            ambient_color: [1.0, 1.0, 1.0],
            ambient_intensity: 0.15,
            directional: DirectionalLight {
                direction: [0.3, 1.0, 0.5],
                color: [1.0, 1.0, 1.0],
                intensity: 0.85,
            },
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
        }
    }
}

impl LightEnvironment {
    /// Create a default light environment (ambient + directional, no point/spot lights).
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns (point_shadow_count, spot_shadow_count) — number of shadow-casting
    /// lights of each type, capped at 4 each.
    pub fn shadow_casting_counts(&self) -> (usize, usize) {
        let point = self
            .point_lights
            .iter()
            .filter(|pl| pl.cast_shadows)
            .count()
            .min(4);
        let spot = self
            .spot_lights
            .iter()
            .filter(|sl| sl.cast_shadows)
            .count()
            .min(4);
        (point, spot)
    }

    /// Pack into GPU-ready [`LightUniforms`].
    ///
    /// Shadow-casting lights are sorted to the front so shader indices 0..N
    /// correspond to the shadow map array layers.
    pub fn to_uniforms(&self) -> LightUniforms {
        // Sort shadow-casters first.
        let mut sorted_points: Vec<&PointLight> = self.point_lights.iter().collect();
        sorted_points.sort_by_key(|pl| !pl.cast_shadows);

        let point_count = sorted_points.len().min(MAX_POINT_LIGHTS);
        let mut point_lights = [PointLightGpu {
            position_range: [0.0; 4],
            color_intensity: [0.0; 4],
        }; MAX_POINT_LIGHTS];

        for (i, pl) in sorted_points.iter().take(point_count).enumerate() {
            point_lights[i] = PointLightGpu {
                position_range: [pl.position[0], pl.position[1], pl.position[2], pl.range],
                color_intensity: [pl.color[0], pl.color[1], pl.color[2], pl.intensity],
            };
        }

        let mut sorted_spots: Vec<&SpotLight> = self.spot_lights.iter().collect();
        sorted_spots.sort_by_key(|sl| !sl.cast_shadows);

        let spot_count = sorted_spots.len().min(MAX_SPOT_LIGHTS);
        let mut spot_lights = [SpotLightGpu {
            position_range: [0.0; 4],
            direction_inner: [0.0; 4],
            color_intensity: [0.0; 4],
            outer_pad: [0.0; 4],
        }; MAX_SPOT_LIGHTS];

        for (i, sl) in sorted_spots.iter().take(spot_count).enumerate() {
            spot_lights[i] = SpotLightGpu {
                position_range: [sl.position[0], sl.position[1], sl.position[2], sl.range],
                direction_inner: [
                    sl.direction[0],
                    sl.direction[1],
                    sl.direction[2],
                    sl.inner_cone_angle.cos(),
                ],
                color_intensity: [sl.color[0], sl.color[1], sl.color[2], sl.intensity],
                outer_pad: [sl.outer_cone_angle.cos(), 0.0, 0.0, 0.0],
            };
        }

        let d = &self.directional;
        LightUniforms {
            ambient: [
                self.ambient_color[0],
                self.ambient_color[1],
                self.ambient_color[2],
                self.ambient_intensity,
            ],
            directional_dir_intensity: [
                d.direction[0],
                d.direction[1],
                d.direction[2],
                d.intensity,
            ],
            directional_color_count: [d.color[0], d.color[1], d.color[2], point_count as f32],
            spot_count_pad: [spot_count as f32, 0.0, 0.0, 0.0],
            point_lights,
            spot_lights,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_uniforms_size_and_alignment() {
        assert_eq!(size_of::<LightUniforms>(), 576);
        assert_eq!(align_of::<LightUniforms>(), 4);
    }

    #[test]
    fn point_light_gpu_is_32_bytes() {
        assert_eq!(size_of::<PointLightGpu>(), 32);
    }

    #[test]
    fn spot_light_gpu_is_64_bytes() {
        assert_eq!(size_of::<SpotLightGpu>(), 64);
    }

    #[test]
    fn light_uniforms_is_pod() {
        let u = LightUniforms {
            ambient: [0.0; 4],
            directional_dir_intensity: [0.0; 4],
            directional_color_count: [0.0; 4],
            spot_count_pad: [0.0; 4],
            point_lights: [PointLightGpu {
                position_range: [0.0; 4],
                color_intensity: [0.0; 4],
            }; MAX_POINT_LIGHTS],
            spot_lights: [SpotLightGpu {
                position_range: [0.0; 4],
                direction_inner: [0.0; 4],
                color_intensity: [0.0; 4],
                outer_pad: [0.0; 4],
            }; MAX_SPOT_LIGHTS],
        };
        let _bytes: &[u8] = bytemuck::bytes_of(&u);
    }

    #[test]
    fn default_light_environment() {
        let env = LightEnvironment::new();
        assert_eq!(env.ambient_intensity, 0.15);
        assert!(env.point_lights.is_empty());
        assert!(env.spot_lights.is_empty());
        assert_eq!(env.directional.color, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn to_uniforms_packs_correctly() {
        let env = LightEnvironment {
            ambient_color: [0.1, 0.2, 0.3],
            ambient_intensity: 0.5,
            directional: DirectionalLight {
                direction: [0.0, 1.0, 0.0],
                color: [1.0, 0.9, 0.8],
                intensity: 0.7,
            },
            point_lights: vec![PointLight {
                position: [1.0, 2.0, 3.0],
                color: [1.0, 0.0, 0.0],
                intensity: 5.0,
                range: 10.0,
                cast_shadows: false,
            }],
            spot_lights: Vec::new(),
        };
        let u = env.to_uniforms();
        assert_eq!(u.ambient, [0.1, 0.2, 0.3, 0.5]);
        assert_eq!(u.directional_dir_intensity, [0.0, 1.0, 0.0, 0.7]);
        assert_eq!(u.directional_color_count[3], 1.0);
        assert_eq!(u.point_lights[0].position_range, [1.0, 2.0, 3.0, 10.0]);
        assert_eq!(u.point_lights[0].color_intensity, [1.0, 0.0, 0.0, 5.0]);
        assert_eq!(u.point_lights[1].position_range, [0.0; 4]);
    }

    #[test]
    fn point_lights_capped_at_max() {
        let env = LightEnvironment {
            point_lights: (0..20)
                .map(|i| PointLight {
                    position: [i as f32, 0.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    intensity: 1.0,
                    range: 5.0,
                    cast_shadows: false,
                })
                .collect(),
            ..Default::default()
        };
        let u = env.to_uniforms();
        assert_eq!(u.directional_color_count[3], MAX_POINT_LIGHTS as f32);
        assert_eq!(u.point_lights[7].position_range[0], 7.0);
    }

    #[test]
    fn spot_lights_pack_correctly() {
        let env = LightEnvironment {
            spot_lights: vec![SpotLight {
                position: [1.0, 2.0, 3.0],
                direction: [0.0, -1.0, 0.0],
                color: [1.0, 1.0, 0.0],
                intensity: 10.0,
                range: 20.0,
                inner_cone_angle: 0.3,
                outer_cone_angle: 0.5,
                cast_shadows: false,
            }],
            ..Default::default()
        };
        let u = env.to_uniforms();
        assert_eq!(u.spot_count_pad[0], 1.0);
        assert_eq!(u.spot_lights[0].position_range, [1.0, 2.0, 3.0, 20.0]);
        assert_eq!(u.spot_lights[0].direction_inner[0], 0.0);
        assert_eq!(u.spot_lights[0].direction_inner[1], -1.0);
        let cos_inner = 0.3_f32.cos();
        assert!((u.spot_lights[0].direction_inner[3] - cos_inner).abs() < 1e-6);
        assert_eq!(u.spot_lights[0].color_intensity, [1.0, 1.0, 0.0, 10.0]);
        let cos_outer = 0.5_f32.cos();
        assert!((u.spot_lights[0].outer_pad[0] - cos_outer).abs() < 1e-6);
    }

    #[test]
    fn spot_lights_capped_at_max() {
        let env = LightEnvironment {
            spot_lights: (0..10)
                .map(|i| SpotLight {
                    position: [i as f32, 0.0, 0.0],
                    direction: [0.0, -1.0, 0.0],
                    color: [1.0, 1.0, 1.0],
                    intensity: 1.0,
                    range: 5.0,
                    inner_cone_angle: 0.3,
                    outer_cone_angle: 0.5,
                    cast_shadows: false,
                })
                .collect(),
            ..Default::default()
        };
        let u = env.to_uniforms();
        assert_eq!(u.spot_count_pad[0], MAX_SPOT_LIGHTS as f32);
    }
}
