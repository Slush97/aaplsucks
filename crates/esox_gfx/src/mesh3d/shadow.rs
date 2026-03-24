//! Cascaded shadow maps — depth-only rendering from the light's perspective.
//!
//! Provides up to [`MAX_SHADOW_CASCADES`] cascaded shadow map layers, each
//! rendered into a `Depth32Float` texture array slice at 2048x2048. The cascade
//! splits use a practical split scheme (lambda-blended logarithmic + linear)
//! and tight orthographic projections computed from the camera frustum corners.

use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use crate::pipeline::GpuContext;
use super::camera::Camera;
use super::instance::instance_buffer_layout;
use super::vertex::vertex_buffer_layout;

// ── Constants ──

/// Maximum number of shadow cascades (array layers in the depth texture).
pub const MAX_SHADOW_CASCADES: usize = 4;

/// Shadow map resolution per cascade layer (width and height).
const SHADOW_MAP_SIZE: u32 = 2048;

/// Maximum point lights that can cast shadows simultaneously.
pub const MAX_SHADOW_POINT_LIGHTS: usize = 4;

/// Maximum spot lights that can cast shadows simultaneously.
pub const MAX_SHADOW_SPOT_LIGHTS: usize = 4;

/// Shadow map resolution for point light cube faces.
pub const POINT_SHADOW_MAP_SIZE: u32 = 512;

/// Shadow map resolution for spot light shadow maps.
pub const SPOT_SHADOW_MAP_SIZE: u32 = 1024;

// ── Shadow vertex shader ──

pub(crate) const SHADOW_VERTEX_SHADER: &str = r#"
struct ShadowUniforms {
    light_vp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> shadow_uniforms: ShadowUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) tangent: vec4<f32>,
    @location(5) model_0: vec4<f32>,
    @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>,
    @location(8) model_3: vec4<f32>,
    @location(9) inst_color: vec4<f32>,
    @location(10) inst_params: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let world_pos = model * vec4<f32>(in.position, 1.0);
    return shadow_uniforms.light_vp * world_pos;
}
"#;

// ── GPU uniform struct ──

/// GPU-side shadow uniform data — uploaded to the scene shader so fragments can
/// sample the shadow map and compare depths.
///
/// 288 bytes = 4 × mat4x4 (256) + splits_count vec4 (16) + shadow_config vec4 (16).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ShadowUniforms {
    /// Light-space view-projection matrices, one per cascade.
    pub light_vp: [[[f32; 4]; 4]; MAX_SHADOW_CASCADES],
    /// Cascade far-plane splits packed into a vec4 (up to 4 values).
    /// `splits_count.w` is unused padding.
    pub splits_count: [f32; 4],
    /// Shadow configuration:
    /// `[depth_bias, normal_bias, shadow_distance, active_cascade_count]`.
    pub shadow_config: [f32; 4],
}

// ── Configuration ──

/// High-level shadow configuration.
#[derive(Debug, Clone, Copy)]
pub struct ShadowConfig {
    /// Whether shadow mapping is enabled.
    pub enabled: bool,
    /// Number of active cascades (clamped to 2..=MAX_SHADOW_CASCADES).
    pub cascade_count: usize,
    /// Maximum shadow rendering distance from the camera.
    pub shadow_distance: f32,
    /// Constant depth bias to reduce shadow acne.
    pub depth_bias: f32,
    /// Normal-direction bias to reduce peter-panning.
    pub normal_bias: f32,
    /// Distance to pull the light back from the frustum center.
    /// Higher values accommodate taller geometry. Default: 50.0.
    pub light_distance: f32,
    /// Depth bias for point light shadows.
    pub point_depth_bias: f32,
    /// Normal bias for point light shadows.
    pub point_normal_bias: f32,
    /// Depth bias for spot light shadows.
    pub spot_depth_bias: f32,
    /// Normal bias for spot light shadows.
    pub spot_normal_bias: f32,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cascade_count: 3,
            shadow_distance: 100.0,
            depth_bias: 0.002,
            normal_bias: 0.02,
            light_distance: 50.0,
            point_depth_bias: 0.005,
            point_normal_bias: 0.08,
            spot_depth_bias: 0.003,
            spot_normal_bias: 0.02,
        }
    }
}

// ── Omni shadow GPU uniform struct ──

/// GPU-side uniform data for point and spot light shadow maps.
///
/// 1824 bytes = 24 mat4x4 (1536) + 4 mat4x4 (256) + 2 vec4 (32).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct OmniShadowUniforms {
    /// View-projection matrices for point light cube faces (4 lights × 6 faces = 24).
    pub point_light_vp: [[[f32; 4]; 4]; 24],
    /// View-projection matrices for spot lights (up to 4).
    pub spot_light_vp: [[[f32; 4]; 4]; MAX_SHADOW_SPOT_LIGHTS],
    /// x=point_shadow_count, y=spot_shadow_count, z=point_depth_bias, w=point_normal_bias.
    pub omni_config: [f32; 4],
    /// x=spot_depth_bias, y=spot_normal_bias, zw=pad.
    pub omni_config2: [f32; 4],
}

// ── Per-cascade uniform (just a single light-VP matrix) ──

/// Per-cascade uniform uploaded before each shadow render pass.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CascadeUniforms {
    light_vp: [[f32; 4]; 4],
}

// ── Cascade split computation ──

/// Compute cascade split distances using a practical split scheme.
///
/// Blends logarithmic and linear splits controlled by `lambda` (0 = fully linear,
/// 1 = fully logarithmic). `lambda = 0.5` is a good practical default.
///
/// Returns `count + 1` split distances: `[near, split_1, ..., split_count]`.
pub fn compute_cascade_splits(near: f32, far: f32, count: usize, lambda: f32) -> Vec<f32> {
    let mut splits = Vec::with_capacity(count + 1);
    splits.push(near);

    for i in 1..=count {
        let p = i as f32 / count as f32;
        let lin_split = near + (far - near) * p;
        let split = if near > 0.0 && lambda > 0.0 {
            let log_split = near * (far / near).powf(p);
            lambda * log_split + (1.0 - lambda) * lin_split
        } else {
            lin_split
        };
        splits.push(split);
    }

    splits
}

// ── Cascade matrix computation ──

/// Compute a tight orthographic light-space view-projection matrix for a single
/// cascade, given the camera frustum corners for that cascade's depth range.
fn compute_cascade_matrix(
    camera: &super::camera::Camera,
    aspect: f32,
    near_split: f32,
    far_split: f32,
    light_dir: Vec3,
    light_distance: f32,
) -> Mat4 {
    // Build a projection for this sub-frustum (handles both perspective and ortho).
    let proj = camera.sub_projection(aspect, near_split, far_split);
    let view = camera.view_matrix();
    let inv_vp = (proj * view).inverse();

    // NDC corners of the unit cube in clip space.
    let ndc_corners: [[f32; 3]; 8] = [
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [-1.0,  1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0, -1.0, 1.0],
        [ 1.0, -1.0, 1.0],
        [-1.0,  1.0, 1.0],
        [ 1.0,  1.0, 1.0],
    ];

    // Unproject to world space.
    let mut world_corners = [Vec3::ZERO; 8];
    for (i, ndc) in ndc_corners.iter().enumerate() {
        let clip = Vec4::new(ndc[0], ndc[1], ndc[2], 1.0);
        let world = inv_vp * clip;
        world_corners[i] = world.truncate() / world.w;
    }

    // Frustum center.
    let center = world_corners.iter().copied().sum::<Vec3>() / 8.0;

    // Light view matrix: look at the center from the light direction.
    let light_dir_n = light_dir.normalize();
    let light_pos = center - light_dir_n * light_distance;
    let light_view = Mat4::look_at_rh(light_pos, center, Vec3::Y);

    // Project corners into light view space and find tight AABB.
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for corner in &world_corners {
        let lv = light_view * Vec4::new(corner.x, corner.y, corner.z, 1.0);
        min_x = min_x.min(lv.x);
        max_x = max_x.max(lv.x);
        min_y = min_y.min(lv.y);
        max_y = max_y.max(lv.y);
        min_z = min_z.min(lv.z);
        max_z = max_z.max(lv.z);
    }

    // Extend the near plane to catch shadow casters behind the frustum.
    let z_margin = (max_z - min_z) * 0.5;
    min_z -= z_margin;

    // Texel snapping: round ortho bounds to the shadow map texel grid so
    // shadow edges don't shimmer when the camera moves.
    let texels_per_unit_x = SHADOW_MAP_SIZE as f32 / (max_x - min_x);
    let texels_per_unit_y = SHADOW_MAP_SIZE as f32 / (max_y - min_y);
    min_x = (min_x * texels_per_unit_x).floor() / texels_per_unit_x;
    max_x = (max_x * texels_per_unit_x).ceil() / texels_per_unit_x;
    min_y = (min_y * texels_per_unit_y).floor() / texels_per_unit_y;
    max_y = (max_y * texels_per_unit_y).ceil() / texels_per_unit_y;

    let light_proj = Mat4::orthographic_rh(min_x, max_x, min_y, max_y, min_z, max_z);

    light_proj * light_view
}

/// Compute light-space view-projection matrices for all active cascades.
pub fn compute_cascade_matrices(
    camera: &super::camera::Camera,
    aspect: f32,
    light_dir: Vec3,
    splits: &[f32],
    light_distance: f32,
) -> Vec<Mat4> {
    let cascade_count = splits.len().saturating_sub(1);
    let mut matrices = Vec::with_capacity(cascade_count);

    for i in 0..cascade_count {
        let near_split = splits[i];
        let far_split = splits[i + 1];
        matrices.push(compute_cascade_matrix(camera, aspect, near_split, far_split, light_dir, light_distance));
    }

    matrices
}

// ── ShadowState ──

/// Bundled shadow map state for the 3D renderer.
pub(super) struct ShadowState {
    pub(super) shadow_pass: Option<ShadowPass>,
    pub(super) shadow_uniform_buffer: wgpu::Buffer,
    pub(super) comparison_sampler: wgpu::Sampler,
    pub(super) fallback_shadow_depth_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(super) fallback_shadow_depth_texture: wgpu::Texture,
    pub(super) point_shadow_pass: Option<PointShadowPass>,
    pub(super) spot_shadow_pass: Option<SpotShadowPass>,
    pub(super) omni_shadow_uniform_buffer: wgpu::Buffer,
    pub(super) fallback_point_shadow_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(super) fallback_point_shadow_texture: wgpu::Texture,
    pub(super) fallback_spot_shadow_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub(super) fallback_spot_shadow_texture: wgpu::Texture,
}

impl ShadowState {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let shadow_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_shadow_uniforms"),
            size: size_of::<ShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Comparison sampler for shadow mapping.
        let comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("esox_3d_comparison_sampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Fallback 1x1 depth texture array (4 layers) for when shadows are disabled.
        let fallback_shadow_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("esox_3d_fallback_shadow_depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: MAX_SHADOW_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fallback_shadow_depth_view =
            fallback_shadow_depth_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("esox_3d_fallback_shadow_depth_view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        // Fallback 1x1 depth texture arrays for point/spot shadows when disabled.
        let fallback_point_shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("esox_3d_fallback_point_shadow_depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: (MAX_SHADOW_POINT_LIGHTS * 6) as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fallback_point_shadow_view =
            fallback_point_shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("esox_3d_fallback_point_shadow_depth_view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        let fallback_spot_shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("esox_3d_fallback_spot_shadow_depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: MAX_SHADOW_SPOT_LIGHTS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fallback_spot_shadow_view =
            fallback_spot_shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("esox_3d_fallback_spot_shadow_depth_view"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        // Clear all fallback shadow depth textures to 1.0 (fully lit) so that
        // sampling them before real shadow passes run never produces garbage.
        {
            let clear_textures: &[(&wgpu::Texture, u32)] = &[
                (&fallback_shadow_depth_texture, MAX_SHADOW_CASCADES as u32),
                (&fallback_point_shadow_texture, (MAX_SHADOW_POINT_LIGHTS * 6) as u32),
                (&fallback_spot_shadow_texture, MAX_SHADOW_SPOT_LIGHTS as u32),
            ];
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("esox_3d_fallback_shadow_clear"),
            });
            for &(tex, layers) in clear_textures {
                for layer in 0..layers {
                    let view = tex.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    });
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("esox_3d_fallback_shadow_clear_pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                }
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        // Omni shadow uniform buffer.
        let omni_shadow_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_omni_shadow_uniforms"),
            size: size_of::<OmniShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            shadow_pass: None,
            shadow_uniform_buffer,
            comparison_sampler,
            fallback_shadow_depth_view,
            fallback_shadow_depth_texture,
            point_shadow_pass: None,
            spot_shadow_pass: None,
            omni_shadow_uniform_buffer,
            fallback_point_shadow_view,
            fallback_point_shadow_texture,
            fallback_spot_shadow_view,
            fallback_spot_shadow_texture,
        }
    }
}

// ── ShadowPass ──

/// Owns the shadow depth texture array, shadow pipeline, and per-cascade uniform
/// buffers/bind groups. Call [`ShadowPass::update_cascades`] each frame to compute
/// cascade splits and light-space matrices, then [`ShadowPass::begin_cascade_pass`]
/// for each cascade to render depth.
pub struct ShadowPass {
    /// 2D array depth texture (`Depth32Float`, `MAX_SHADOW_CASCADES` layers).
    /// Kept alive so the GPU texture isn't deallocated while the view is in use.
    #[allow(dead_code)]
    pub(crate) depth_texture: wgpu::Texture,
    /// View of the entire depth texture array (for binding as a sampled texture).
    pub(crate) depth_view: wgpu::TextureView,
    /// Per-cascade views for render-pass depth attachments.
    pub(crate) cascade_views: Vec<wgpu::TextureView>,
    /// Depth-only render pipeline (vertex shader only).
    pub(crate) pipeline: wgpu::RenderPipeline,
    /// Bind group layout for per-cascade uniforms.
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    /// Per-cascade uniform buffers (one light-VP mat4x4 each).
    cascade_buffers: Vec<wgpu::Buffer>,
    /// Per-cascade bind groups referencing the uniform buffers.
    cascade_bind_groups: Vec<wgpu::BindGroup>,
    /// Shadow configuration.
    pub config: ShadowConfig,
}

impl ShadowPass {
    /// Create the shadow pass pipeline, textures, and per-cascade resources.
    pub fn new(device: &wgpu::Device) -> Self {
        let config = ShadowConfig::default();

        // ── Depth texture array ──
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow_depth_texture"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: MAX_SHADOW_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Full array view (for sampling in the scene shader).
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_depth_view_array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Per-cascade views (each targets a single array layer).
        let mut cascade_views = Vec::with_capacity(MAX_SHADOW_CASCADES);
        for i in 0..MAX_SHADOW_CASCADES {
            cascade_views.push(depth_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("shadow_cascade_{i}_view")),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            }));
        }

        // ── Bind group layout (one uniform buffer per cascade) ──
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        size_of::<CascadeUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });

        // Per-cascade uniform buffers and bind groups.
        let mut cascade_buffers = Vec::with_capacity(MAX_SHADOW_CASCADES);
        let mut cascade_bind_groups = Vec::with_capacity(MAX_SHADOW_CASCADES);

        for i in 0..MAX_SHADOW_CASCADES {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("shadow_cascade_{i}_uniform")),
                contents: bytemuck::bytes_of(&CascadeUniforms {
                    light_vp: Mat4::IDENTITY.to_cols_array_2d(),
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shadow_cascade_{i}_bind_group")),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

            cascade_buffers.push(buffer);
            cascade_bind_groups.push(bind_group);
        }

        // ── Pipeline layout ──
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        // ── Shader module ──
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_vertex_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_VERTEX_SHADER.into()),
        });

        // ── Depth-only render pipeline (no fragment shader) ──
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Cull front faces: depth values come from back faces, which
                // prevents front-face self-shadowing (shadow acne).
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            depth_texture,
            depth_view,
            cascade_views,
            pipeline,
            bind_group_layout,
            cascade_buffers,
            cascade_bind_groups,
            config,
        }
    }

    /// Rebuild the shadow pipeline with new shader source.
    #[cfg(feature = "hot-reload")]
    pub fn rebuild_pipeline(&mut self, device: &wgpu::Device, src: &str) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_vertex_shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pipeline_layout"),
            bind_group_layouts: &[&self.bind_group_layout],
            immediate_size: 0,
        });

        self.pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[vertex_buffer_layout(), instance_buffer_layout()],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 1.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
    }

    /// Compute cascade splits and light-space matrices for this frame, upload
    /// per-cascade uniforms, and return a [`ShadowUniforms`] for the scene shader.
    pub fn update_cascades(
        &self,
        queue: &wgpu::Queue,
        camera: &super::camera::Camera,
        light_dir: [f32; 3],
        aspect: f32,
    ) -> ShadowUniforms {
        let count = self
            .config
            .cascade_count
            .clamp(2, MAX_SHADOW_CASCADES);

        let shadow_far = self.config.shadow_distance.min(camera.far);
        let splits = compute_cascade_splits(camera.near, shadow_far, count, 0.5);
        let light = Vec3::from(light_dir);
        let matrices = compute_cascade_matrices(camera, aspect, light, &splits, self.config.light_distance);

        // Upload per-cascade light-VP matrices.
        let mut light_vp = [[[0.0f32; 4]; 4]; MAX_SHADOW_CASCADES];
        for (i, mat) in matrices.iter().enumerate() {
            light_vp[i] = mat.to_cols_array_2d();
            queue.write_buffer(
                &self.cascade_buffers[i],
                0,
                bytemuck::bytes_of(&CascadeUniforms {
                    light_vp: light_vp[i],
                }),
            );
        }

        // Pack split far-planes into splits_count (skip the near plane at index 0).
        let mut splits_count = [0.0f32; 4];
        for i in 0..count.min(MAX_SHADOW_CASCADES) {
            splits_count[i] = splits[i + 1];
        }

        ShadowUniforms {
            light_vp,
            splits_count,
            shadow_config: [
                self.config.depth_bias,
                self.config.normal_bias,
                self.config.shadow_distance,
                count as f32,
            ],
        }
    }

    /// Begin a depth-only render pass for the given cascade layer.
    ///
    /// The returned [`wgpu::RenderPass`] already has the shadow pipeline and the
    /// cascade's bind group set. The caller should issue draw commands (set vertex/
    /// index/instance buffers and call `draw_indexed`).
    pub fn begin_cascade_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        cascade: usize,
    ) -> wgpu::RenderPass<'a> {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!("shadow_cascade_{cascade}_pass")),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.cascade_views[cascade],
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.cascade_bind_groups[cascade], &[]);

        pass
    }
}

// ── Point light face matrices ──

/// Compute 6 view-projection matrices for a point light's cube shadow map.
///
/// Each face covers a 90° FOV, aspect 1:1, rendering into one layer of the
/// depth texture array.
pub fn point_light_face_matrices(position: Vec3, range: f32) -> [Mat4; 6] {
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, range);

    // +X, -X, +Y, -Y, +Z, -Z
    let targets_and_ups: [(Vec3, Vec3); 6] = [
        (Vec3::X, -Vec3::Y),
        (-Vec3::X, -Vec3::Y),
        (Vec3::Y, Vec3::Z),
        (-Vec3::Y, -Vec3::Z),
        (Vec3::Z, -Vec3::Y),
        (-Vec3::Z, -Vec3::Y),
    ];

    let mut matrices = [Mat4::IDENTITY; 6];
    for (i, (dir, up)) in targets_and_ups.iter().enumerate() {
        let view = Mat4::look_at_rh(position, position + *dir, *up);
        matrices[i] = proj * view;
    }
    matrices
}

/// Compute a view-projection matrix for a spot light shadow map.
pub fn spot_light_vp(position: Vec3, direction: Vec3, outer_angle: f32, range: f32) -> Mat4 {
    let fov = (outer_angle * 2.0).min(std::f32::consts::PI - 0.01);
    let proj = Mat4::perspective_rh(fov, 1.0, 0.1, range);

    let dir = direction.normalize();
    // Choose an up vector that isn't parallel to direction.
    let up = if dir.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(position, position + dir, up);
    proj * view
}

// ── PointShadowPass ──

/// Depth-only render pass for point light cube shadow maps.
///
/// Uses a `texture_depth_2d_array` with 24 layers (4 lights × 6 faces).
/// Reuses the same shadow vertex shader and pipeline as the CSM pass.
pub struct PointShadowPass {
    /// Depth texture array (512×512, 24 layers).
    #[allow(dead_code)]
    pub(crate) depth_texture: wgpu::Texture,
    /// Full array view for sampling in the fragment shader.
    pub(crate) depth_view: wgpu::TextureView,
    /// Per-face views (24 total) for render-pass depth attachments.
    pub(crate) face_views: Vec<wgpu::TextureView>,
    /// Per-face uniform buffers (one light-VP mat4x4 each).
    face_buffers: Vec<wgpu::Buffer>,
    /// Per-face bind groups.
    face_bind_groups: Vec<wgpu::BindGroup>,
}

impl PointShadowPass {
    pub fn new(device: &wgpu::Device, shadow_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let total_layers = (MAX_SHADOW_POINT_LIGHTS * 6) as u32; // 24

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("point_shadow_depth_texture"),
            size: wgpu::Extent3d {
                width: POINT_SHADOW_MAP_SIZE,
                height: POINT_SHADOW_MAP_SIZE,
                depth_or_array_layers: total_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("point_shadow_depth_view_array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let mut face_views = Vec::with_capacity(total_layers as usize);
        let mut face_buffers = Vec::with_capacity(total_layers as usize);
        let mut face_bind_groups = Vec::with_capacity(total_layers as usize);

        for i in 0..total_layers {
            face_views.push(depth_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("point_shadow_face_{i}_view")),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i,
                array_layer_count: Some(1),
                ..Default::default()
            }));

            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("point_shadow_face_{i}_uniform")),
                contents: bytemuck::bytes_of(&CascadeUniforms {
                    light_vp: Mat4::IDENTITY.to_cols_array_2d(),
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("point_shadow_face_{i}_bg")),
                layout: shadow_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

            face_buffers.push(buffer);
            face_bind_groups.push(bind_group);
        }

        Self {
            depth_texture,
            depth_view,
            face_views,
            face_buffers,
            face_bind_groups,
        }
    }

    /// Begin a depth-only render pass for the given face (0..24).
    pub fn begin_face_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        face_idx: usize,
        pipeline: &'a wgpu::RenderPipeline,
    ) -> wgpu::RenderPass<'a> {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!("point_shadow_face_{face_idx}_pass")),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.face_views[face_idx],
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.face_bind_groups[face_idx], &[]);
        pass
    }

    /// Upload the VP matrix for a given face.
    pub fn update_face(&self, queue: &wgpu::Queue, face_idx: usize, vp: &Mat4) {
        queue.write_buffer(
            &self.face_buffers[face_idx],
            0,
            bytemuck::bytes_of(&CascadeUniforms {
                light_vp: vp.to_cols_array_2d(),
            }),
        );
    }
}

// ── SpotShadowPass ──

/// Depth-only render pass for spot light shadow maps.
///
/// Uses a `texture_depth_2d_array` with 4 layers (one per spot light).
pub struct SpotShadowPass {
    /// Depth texture array (1024×1024, 4 layers).
    #[allow(dead_code)]
    pub(crate) depth_texture: wgpu::Texture,
    /// Full array view for sampling in the fragment shader.
    pub(crate) depth_view: wgpu::TextureView,
    /// Per-light views for render-pass depth attachments.
    pub(crate) layer_views: Vec<wgpu::TextureView>,
    /// Per-light uniform buffers.
    layer_buffers: Vec<wgpu::Buffer>,
    /// Per-light bind groups.
    layer_bind_groups: Vec<wgpu::BindGroup>,
}

impl SpotShadowPass {
    pub fn new(device: &wgpu::Device, shadow_bind_group_layout: &wgpu::BindGroupLayout) -> Self {
        let total_layers = MAX_SHADOW_SPOT_LIGHTS as u32; // 4

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("spot_shadow_depth_texture"),
            size: wgpu::Extent3d {
                width: SPOT_SHADOW_MAP_SIZE,
                height: SPOT_SHADOW_MAP_SIZE,
                depth_or_array_layers: total_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("spot_shadow_depth_view_array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let mut layer_views = Vec::with_capacity(total_layers as usize);
        let mut layer_buffers = Vec::with_capacity(total_layers as usize);
        let mut layer_bind_groups = Vec::with_capacity(total_layers as usize);

        for i in 0..total_layers {
            layer_views.push(depth_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("spot_shadow_layer_{i}_view")),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i,
                array_layer_count: Some(1),
                ..Default::default()
            }));

            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("spot_shadow_layer_{i}_uniform")),
                contents: bytemuck::bytes_of(&CascadeUniforms {
                    light_vp: Mat4::IDENTITY.to_cols_array_2d(),
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("spot_shadow_layer_{i}_bg")),
                layout: shadow_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });

            layer_buffers.push(buffer);
            layer_bind_groups.push(bind_group);
        }

        Self {
            depth_texture,
            depth_view,
            layer_views,
            layer_buffers,
            layer_bind_groups,
        }
    }

    /// Begin a depth-only render pass for the given spot light layer.
    pub fn begin_layer_pass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
        layer: usize,
        pipeline: &'a wgpu::RenderPipeline,
    ) -> wgpu::RenderPass<'a> {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&format!("spot_shadow_layer_{layer}_pass")),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.layer_views[layer],
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.layer_bind_groups[layer], &[]);
        pass
    }

    /// Upload the VP matrix for a given spot light layer.
    pub fn update_layer(&self, queue: &wgpu::Queue, layer: usize, vp: &Mat4) {
        queue.write_buffer(
            &self.layer_buffers[layer],
            0,
            bytemuck::bytes_of(&CascadeUniforms {
                light_vp: vp.to_cols_array_2d(),
            }),
        );
    }
}

// ── Shadow pass encoding (Renderer3D integration) ──

impl super::renderer::Renderer3D {
    /// Encode CSM, point-light, and spot-light shadow passes.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_shadow_passes(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        viewport_width: u32,
        viewport_height: u32,
        shadow_opaque_cmds: &[(u32, u32, u32)],
    ) {
        // ── Shadow pass (before scene) ──
        // Skip shadow rendering when directional light intensity is negligible —
        // CSM shadows only apply to the directional light, so there's nothing to shadow.
        let dir_intensity = self.light_env.directional.intensity;
        if let Some(shadow_pass) = &self.shadow_state.shadow_pass {
            if shadow_pass.config.enabled && dir_intensity > 0.001 {
                let aspect = viewport_width as f32 / viewport_height.max(1) as f32;
                let shadow_uniforms = shadow_pass.update_cascades(
                    &gpu.queue,
                    camera,
                    self.light_env.directional.direction,
                    aspect,
                );
                gpu.queue.write_buffer(
                    &self.shadow_state.shadow_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&shadow_uniforms),
                );

                // Render depth from light's perspective for each cascade.
                // Use pre-cull draw data so shadow casters outside the camera
                // frustum still cast visible shadows into the view.
                let cascade_count = shadow_pass.config.cascade_count.clamp(2, MAX_SHADOW_CASCADES);
                for cascade in 0..cascade_count {
                    let mut pass = shadow_pass.begin_cascade_pass(encoder, cascade);

                    // Bind mega-buffer and instance buffer, draw all opaque meshes.
                    pass.set_vertex_buffer(0, self.mega_buffer.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        self.mega_buffer.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

                    for &(mesh_raw, inst_offset, inst_count) in shadow_opaque_cmds {
                        let mesh_idx = mesh_raw as usize;
                        if mesh_idx >= self.mesh_regions.len() {
                            continue;
                        }
                        let r = &self.mesh_regions[mesh_idx];
                        pass.draw_indexed(
                            r.index_offset..r.index_offset + r.index_count,
                            r.vertex_offset as i32,
                            inst_offset..inst_offset + inst_count,
                        );
                    }
                }
            } else {
                // Write zeroed shadow uniforms (shadow_config.w = 0 -> shadow_factor returns 1).
                let zeroed = ShadowUniforms {
                    light_vp: [[[0.0; 4]; 4]; MAX_SHADOW_CASCADES],
                    splits_count: [0.0; 4],
                    shadow_config: [0.0, 0.0, 0.0, 0.0],
                };
                gpu.queue.write_buffer(
                    &self.shadow_state.shadow_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&zeroed),
                );
            }
        } else {
            // No shadow pass: write zeroed shadow uniforms.
            let zeroed = ShadowUniforms {
                light_vp: [[[0.0; 4]; 4]; MAX_SHADOW_CASCADES],
                splits_count: [0.0; 4],
                shadow_config: [0.0, 0.0, 0.0, 0.0],
            };
            gpu.queue.write_buffer(
                &self.shadow_state.shadow_uniform_buffer,
                0,
                bytemuck::bytes_of(&zeroed),
            );
        }

        // ── Point / Spot light shadow passes ──
        {
            let (point_shadow_count, spot_shadow_count) = self.light_env.shadow_casting_counts();
            let mut omni = OmniShadowUniforms {
                point_light_vp: [[[0.0; 4]; 4]; 24],
                spot_light_vp: [[[0.0; 4]; 4]; MAX_SHADOW_SPOT_LIGHTS],
                omni_config: [0.0; 4],
                omni_config2: [0.0; 4],
            };

            let shadow_cfg = self
                .shadow_state.shadow_pass
                .as_ref()
                .map(|sp| sp.config)
                .unwrap_or_default();

            // Point light shadows.
            if let Some(point_pass) = &self.shadow_state.point_shadow_pass {
                // Sort shadow-casters first (to_uniforms already did this).
                let sorted_points: Vec<_> = self.light_env.point_lights.iter()
                    .filter(|pl| pl.cast_shadows)
                    .take(MAX_SHADOW_POINT_LIGHTS)
                    .collect();

                for (li, pl) in sorted_points.iter().enumerate() {
                    let pos = glam::Vec3::from(pl.position);
                    let face_mats = point_light_face_matrices(pos, pl.range);

                    for (fi, mat) in face_mats.iter().enumerate() {
                        let idx = li * 6 + fi;
                        omni.point_light_vp[idx] = mat.to_cols_array_2d();
                        point_pass.update_face(&gpu.queue, idx, mat);
                    }

                    // Render 6 depth-only passes for this point light.
                    let pipeline = &self.shadow_state.shadow_pass.as_ref().unwrap().pipeline;
                    for fi in 0..6 {
                        let face_idx = li * 6 + fi;
                        let mut pass = point_pass.begin_face_pass(encoder, face_idx, pipeline);

                        pass.set_vertex_buffer(0, self.mega_buffer.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            self.mega_buffer.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

                        for &(mesh_raw, inst_offset, inst_count) in shadow_opaque_cmds {
                            let mesh_idx = mesh_raw as usize;
                            let r = &self.mesh_regions[mesh_idx];
                            pass.draw_indexed(
                                r.index_offset..r.index_offset + r.index_count,
                                r.vertex_offset as i32,
                                inst_offset..inst_offset + inst_count,
                            );
                        }
                    }
                }
            }

            // Spot light shadows.
            if let Some(spot_pass) = &self.shadow_state.spot_shadow_pass {
                let sorted_spots: Vec<_> = self.light_env.spot_lights.iter()
                    .filter(|sl| sl.cast_shadows)
                    .take(MAX_SHADOW_SPOT_LIGHTS)
                    .collect();

                for (li, sl) in sorted_spots.iter().enumerate() {
                    let pos = glam::Vec3::from(sl.position);
                    let dir = glam::Vec3::from(sl.direction);
                    let vp = spot_light_vp(pos, dir, sl.outer_cone_angle, sl.range);
                    omni.spot_light_vp[li] = vp.to_cols_array_2d();
                    spot_pass.update_layer(&gpu.queue, li, &vp);

                    let pipeline = &self.shadow_state.shadow_pass.as_ref().unwrap().pipeline;
                    let mut pass = spot_pass.begin_layer_pass(encoder, li, pipeline);

                    pass.set_vertex_buffer(0, self.mega_buffer.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        self.mega_buffer.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

                    for &(mesh_raw, inst_offset, inst_count) in shadow_opaque_cmds {
                        let mesh_idx = mesh_raw as usize;
                        let r = &self.mesh_regions[mesh_idx];
                        pass.draw_indexed(
                            r.index_offset..r.index_offset + r.index_count,
                            r.vertex_offset as i32,
                            inst_offset..inst_offset + inst_count,
                        );
                    }
                }
            }

            omni.omni_config = [
                point_shadow_count as f32,
                spot_shadow_count as f32,
                shadow_cfg.point_depth_bias,
                shadow_cfg.point_normal_bias,
            ];
            omni.omni_config2 = [
                shadow_cfg.spot_depth_bias,
                shadow_cfg.spot_normal_bias,
                0.0,
                0.0,
            ];

            gpu.queue.write_buffer(
                &self.shadow_state.omni_shadow_uniform_buffer,
                0,
                bytemuck::bytes_of(&omni),
            );
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_uniforms_size() {
        assert_eq!(size_of::<ShadowUniforms>(), 288);
    }

    #[test]
    fn compute_cascade_splits_basic() {
        let splits = compute_cascade_splits(0.1, 100.0, 3, 0.5);
        assert_eq!(splits.len(), 4);
        // Monotonically increasing.
        for w in splits.windows(2) {
            assert!(
                w[1] > w[0],
                "splits must be monotonically increasing: {} <= {}",
                w[1],
                w[0],
            );
        }
        // First and last must match near/far.
        assert!((splits[0] - 0.1).abs() < 1e-6);
        assert!((splits[3] - 100.0).abs() < 1e-3);
    }

    #[test]
    fn shadow_config_default() {
        let cfg = ShadowConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.cascade_count, 3);
        assert_eq!(cfg.shadow_distance, 100.0);
        assert!((cfg.depth_bias - 0.002).abs() < 1e-9);
        assert!((cfg.normal_bias - 0.02).abs() < 1e-9);
    }

    #[test]
    fn compute_cascade_splits_two_cascades() {
        let splits = compute_cascade_splits(1.0, 50.0, 2, 0.5);
        assert_eq!(splits.len(), 3);
        assert!((splits[0] - 1.0).abs() < 1e-6);
        assert!(splits[1] > 1.0 && splits[1] < 50.0);
        assert!((splits[2] - 50.0).abs() < 1e-3);
    }

    #[test]
    fn compute_cascade_splits_lambda_zero_is_linear() {
        let splits = compute_cascade_splits(0.0, 100.0, 4, 0.0);
        assert_eq!(splits.len(), 5);
        for (i, &s) in splits.iter().enumerate() {
            let expected = 25.0 * i as f32;
            assert!(
                (s - expected).abs() < 1e-3,
                "linear split {i}: expected {expected}, got {s}",
            );
        }
    }

    #[test]
    fn cascade_uniforms_size() {
        assert_eq!(size_of::<CascadeUniforms>(), 64);
    }

    #[test]
    fn omni_shadow_uniforms_size() {
        assert_eq!(size_of::<OmniShadowUniforms>(), 1824);
    }

    #[test]
    fn point_light_face_matrices_valid() {
        let matrices = point_light_face_matrices(Vec3::ZERO, 10.0);
        for (i, m) in matrices.iter().enumerate() {
            // Each matrix should be invertible (non-degenerate).
            let det = m.determinant();
            assert!(
                det.abs() > 1e-6,
                "face {i} matrix has near-zero determinant: {det}",
            );
        }
    }

    #[test]
    fn spot_light_vp_valid() {
        let vp = spot_light_vp(Vec3::ZERO, Vec3::NEG_Z, 0.5, 10.0);
        assert!(vp.determinant().abs() > 1e-6);
    }
}
