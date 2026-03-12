//! Cascaded shadow maps — depth-only rendering from the light's perspective.
//!
//! Provides up to [`MAX_SHADOW_CASCADES`] cascaded shadow map layers, each
//! rendered into a `Depth32Float` texture array slice at 2048x2048. The cascade
//! splits use a practical split scheme (lambda-blended logarithmic + linear)
//! and tight orthographic projections computed from the camera frustum corners.

use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

use super::instance::instance_buffer_layout;
use super::vertex::vertex_buffer_layout;

// ── Constants ──

/// Maximum number of shadow cascades (array layers in the depth texture).
pub const MAX_SHADOW_CASCADES: usize = 4;

/// Shadow map resolution per cascade layer (width and height).
const SHADOW_MAP_SIZE: u32 = 2048;

// ── Shadow vertex shader ──

const SHADOW_VERTEX_SHADER: &str = r#"
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
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cascade_count: 3,
            shadow_distance: 100.0,
            depth_bias: 0.005,
            normal_bias: 0.02,
        }
    }
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
) -> Mat4 {
    // Build a projection for this sub-frustum.
    let proj = Mat4::perspective_rh(camera.fov_y, aspect, near_split, far_split);
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
    let light_pos = center - light_dir_n * 50.0;
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

    let light_proj = Mat4::orthographic_rh(min_x, max_x, min_y, max_y, min_z, max_z);

    light_proj * light_view
}

/// Compute light-space view-projection matrices for all active cascades.
pub fn compute_cascade_matrices(
    camera: &super::camera::Camera,
    aspect: f32,
    light_dir: Vec3,
    splits: &[f32],
) -> Vec<Mat4> {
    let cascade_count = splits.len().saturating_sub(1);
    let mut matrices = Vec::with_capacity(cascade_count);

    for i in 0..cascade_count {
        let near_split = splits[i];
        let far_split = splits[i + 1];
        matrices.push(compute_cascade_matrix(camera, aspect, near_split, far_split, light_dir));
    }

    matrices
}

// ── ShadowPass ──

/// Owns the shadow depth texture array, shadow pipeline, and per-cascade uniform
/// buffers/bind groups. Call [`ShadowPass::update_cascades`] each frame to compute
/// cascade splits and light-space matrices, then [`ShadowPass::begin_cascade_pass`]
/// for each cascade to render depth.
pub struct ShadowPass {
    /// 2D array depth texture (`Depth32Float`, `MAX_SHADOW_CASCADES` layers).
    pub(crate) depth_texture: wgpu::Texture,
    /// View of the entire depth texture array (for binding as a sampled texture).
    pub(crate) depth_view: wgpu::TextureView,
    /// Per-cascade views for render-pass depth attachments.
    pub(crate) cascade_views: Vec<wgpu::TextureView>,
    /// Depth-only render pipeline (vertex shader only).
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout for per-cascade uniforms.
    bind_group_layout: wgpu::BindGroupLayout,
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
                // Use back-face culling to reduce shadow acne on front faces.
                cull_mode: Some(wgpu::Face::Back),
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
                    slope_scale: 2.0,
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
        let matrices = compute_cascade_matrices(camera, aspect, light, &splits);

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
        assert!((cfg.depth_bias - 0.005).abs() < 1e-9);
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
}
