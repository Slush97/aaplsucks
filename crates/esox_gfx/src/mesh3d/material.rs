//! Material system — surface appearance for 3D meshes.

use std::collections::HashMap;

use super::instance::instance_buffer_layout;
use super::texture::TextureHandle;
use super::vertex::vertex_buffer_layout;

/// Material type controlling which fragment shader is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MaterialType {
    /// No lighting — flat color + emissive only.
    Unlit,
    /// Lambertian diffuse with ambient, directional, and point lights.
    Lit,
    /// Cook-Torrance PBR with metallic-roughness workflow.
    PBR,
    /// Wind Waker-style cel shading with quantized diffuse bands.
    Toon,
}

/// Blend mode for 3D materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BlendMode3D {
    /// Fully opaque — no blending, depth write enabled.
    Opaque,
    /// Standard alpha blending.
    AlphaBlend,
    /// Additive blending (e.g. for particles, glows).
    Additive,
}

/// Face culling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CullMode3D {
    Back,
    #[allow(dead_code)]
    Front,
    None,
}

/// Pipeline cache key — materials with the same key share a render pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub material_type: MaterialType,
    pub blend_mode: BlendMode3D,
    pub cull_mode: CullMode3D,
    pub depth_write: bool,
}

/// Description for creating a material.
#[derive(Debug, Clone)]
pub struct MaterialDescriptor {
    /// Which shading model to use.
    pub material_type: MaterialType,
    /// Base color (linear RGBA, premultiplied alpha).
    pub albedo: [f32; 4],
    /// Emissive color (linear RGB, added after lighting).
    pub emissive: [f32; 3],
    /// Metallic factor (0.0 = dielectric, 1.0 = metal). PBR only.
    pub metallic: f32,
    /// Roughness factor (0.0 = mirror, 1.0 = fully rough). PBR only.
    pub roughness: f32,
    /// Blend mode.
    pub blend_mode: BlendMode3D,
    /// If true, both faces are rendered (cull mode = None).
    pub double_sided: bool,
    /// Whether to write to the depth buffer.
    pub depth_write: bool,
    /// Optional albedo texture (sampled and multiplied with albedo color).
    pub texture: Option<TextureHandle>,
    /// Optional normal map texture (linear, tangent-space).
    pub normal_texture: Option<TextureHandle>,
    /// Optional metallic-roughness texture (linear, G=roughness, B=metallic).
    pub metallic_roughness_texture: Option<TextureHandle>,
    /// Optional emissive texture (sRGB, multiplied with emissive color).
    pub emissive_texture: Option<TextureHandle>,
    /// Normal map scale factor (default 1.0).
    pub normal_scale: f32,
    /// Number of toon diffuse bands (2 or 3). Toon only.
    pub toon_bands: f32,
    /// Rim lighting Fresnel exponent. Toon only.
    pub rim_power: f32,
    /// Rim lighting intensity. Toon only.
    pub rim_intensity: f32,
}

impl Default for MaterialDescriptor {
    fn default() -> Self {
        Self {
            material_type: MaterialType::Lit,
            albedo: [1.0, 1.0, 1.0, 1.0],
            emissive: [0.0, 0.0, 0.0],
            metallic: 0.0,
            roughness: 0.5,
            blend_mode: BlendMode3D::Opaque,
            double_sided: false,
            depth_write: true,
            texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
            normal_scale: 1.0,
            toon_bands: 3.0,
            rim_power: 3.0,
            rim_intensity: 0.4,
        }
    }
}

impl MaterialDescriptor {
    pub(crate) fn pipeline_key(&self) -> PipelineKey {
        PipelineKey {
            material_type: self.material_type,
            blend_mode: self.blend_mode,
            cull_mode: if self.double_sided {
                CullMode3D::None
            } else {
                CullMode3D::Back
            },
            depth_write: self.depth_write,
        }
    }
}

/// GPU uniform block for material parameters (80 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MaterialUniforms {
    /// Base color (linear RGBA).
    pub albedo: [f32; 4],
    /// Emissive (rgb) + metallic (w).
    pub emissive_metallic: [f32; 4],
    /// x = roughness, y = opacity, z = material_type as f32, w = pad.
    pub roughness_opacity_flags: [f32; 4],
    /// Texture presence flags: (has_albedo, has_normal, has_mr, has_emissive).
    pub texture_flags: [f32; 4],
    /// Extra parameters: (normal_scale, toon_bands, rim_power, rim_intensity).
    pub extra: [f32; 4],
}

impl MaterialDescriptor {
    /// Pack into GPU-ready uniform data.
    pub fn to_uniforms(&self) -> MaterialUniforms {
        let type_flag = match self.material_type {
            MaterialType::Unlit => 0.0,
            MaterialType::Lit => 1.0,
            MaterialType::PBR => 2.0,
            MaterialType::Toon => 3.0,
        };
        MaterialUniforms {
            albedo: self.albedo,
            emissive_metallic: [
                self.emissive[0],
                self.emissive[1],
                self.emissive[2],
                self.metallic,
            ],
            roughness_opacity_flags: [self.roughness, self.albedo[3], type_flag, 0.0],
            texture_flags: [
                if self.texture.is_some() { 1.0 } else { 0.0 },
                if self.normal_texture.is_some() {
                    1.0
                } else {
                    0.0
                },
                if self.metallic_roughness_texture.is_some() {
                    1.0
                } else {
                    0.0
                },
                if self.emissive_texture.is_some() {
                    1.0
                } else {
                    0.0
                },
            ],
            extra: [self.normal_scale, self.toon_bands, self.rim_power, self.rim_intensity],
        }
    }
}

/// Handle to a material stored in the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialHandle(pub u32);

/// Internal material — pipeline key + GPU resources.
pub(crate) struct Material {
    pub pipeline_key: PipelineKey,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub texture: Option<TextureHandle>,
    pub normal_texture: Option<TextureHandle>,
    pub metallic_roughness_texture: Option<TextureHandle>,
    pub emissive_texture: Option<TextureHandle>,
    pub descriptor: MaterialDescriptor,
}

/// Convert [`BlendMode3D`] to wgpu blend state.
pub(crate) fn blend_state(mode: BlendMode3D) -> Option<wgpu::BlendState> {
    match mode {
        BlendMode3D::Opaque => None,
        BlendMode3D::AlphaBlend => Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
        BlendMode3D::Additive => Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        }),
    }
}

/// Convert [`CullMode3D`] to wgpu cull mode.
pub(crate) fn cull_face(mode: CullMode3D) -> Option<wgpu::Face> {
    match mode {
        CullMode3D::Back => Some(wgpu::Face::Back),
        CullMode3D::Front => Some(wgpu::Face::Front),
        CullMode3D::None => None,
    }
}

/// Create a render pipeline for the given pipeline key.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    shader_modules: &HashMap<MaterialType, wgpu::ShaderModule>,
    key: &PipelineKey,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = &shader_modules[&key.material_type];
    create_pipeline_with_shader(device, format, layout, shader, key, sample_count)
}

/// Create a render pipeline using an explicit shader module (for custom materials).
pub(crate) fn create_pipeline_with_shader(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    key: &PipelineKey,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let depth_format = wgpu::TextureFormat::Depth32Float;

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("esox_3d_material_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: blend_state(key.blend_mode),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: cull_face(key.cull_mode),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: key.depth_write,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_uniforms_is_80_bytes() {
        assert_eq!(size_of::<MaterialUniforms>(), 80);
    }

    #[test]
    fn material_uniforms_is_pod() {
        let u = MaterialUniforms {
            albedo: [1.0, 1.0, 1.0, 1.0],
            emissive_metallic: [0.0; 4],
            roughness_opacity_flags: [0.5, 1.0, 1.0, 0.0],
            texture_flags: [0.0; 4],
            extra: [1.0, 0.0, 0.0, 0.0],
        };
        let _bytes: &[u8] = bytemuck::bytes_of(&u);
    }

    #[test]
    fn default_descriptor_is_opaque_lit() {
        let d = MaterialDescriptor::default();
        assert_eq!(d.material_type, MaterialType::Lit);
        assert_eq!(d.blend_mode, BlendMode3D::Opaque);
        assert!(d.depth_write);
        assert!(!d.double_sided);
    }

    #[test]
    fn pipeline_key_dedup() {
        let a = MaterialDescriptor::default();
        let b = MaterialDescriptor {
            albedo: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        };
        assert_eq!(a.pipeline_key(), b.pipeline_key());
    }

    #[test]
    fn pipeline_key_differs_on_blend() {
        let a = MaterialDescriptor::default();
        let b = MaterialDescriptor {
            blend_mode: BlendMode3D::Additive,
            ..Default::default()
        };
        assert_ne!(a.pipeline_key(), b.pipeline_key());
    }

    #[test]
    fn pipeline_key_differs_on_cull() {
        let a = MaterialDescriptor::default();
        let b = MaterialDescriptor {
            double_sided: true,
            ..Default::default()
        };
        assert_ne!(a.pipeline_key(), b.pipeline_key());
    }

    #[test]
    fn material_descriptor_default_no_texture() {
        let d = MaterialDescriptor::default();
        assert!(d.texture.is_none());
        assert!(d.normal_texture.is_none());
        assert!(d.metallic_roughness_texture.is_none());
        assert!(d.emissive_texture.is_none());
    }

    #[test]
    fn material_uniforms_texture_flags_off() {
        let d = MaterialDescriptor::default();
        let u = d.to_uniforms();
        assert_eq!(u.texture_flags, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn material_uniforms_texture_flags_on() {
        let d = MaterialDescriptor {
            texture: Some(TextureHandle(0)),
            normal_texture: Some(TextureHandle(1)),
            metallic_roughness_texture: Some(TextureHandle(2)),
            emissive_texture: Some(TextureHandle(3)),
            ..Default::default()
        };
        let u = d.to_uniforms();
        assert_eq!(u.texture_flags, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn normal_scale_default() {
        let d = MaterialDescriptor::default();
        let u = d.to_uniforms();
        assert_eq!(u.extra[0], 1.0);
    }

    #[test]
    fn blend_mode_to_wgpu() {
        assert!(blend_state(BlendMode3D::Opaque).is_none());
        assert!(blend_state(BlendMode3D::AlphaBlend).is_some());
        assert!(blend_state(BlendMode3D::Additive).is_some());
    }
}
