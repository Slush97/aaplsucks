//! Camera motion blur post-process — fullscreen velocity reconstruction + blur.
//!
//! Objects streak when the camera moves. No per-object velocity buffer is
//! required; instead the velocity is reconstructed from the depth buffer and
//! the current/previous view-projection matrices.
//!
//! Two fullscreen passes:
//! 1. **Velocity pass** — reconstructs world position from depth + inverse VP,
//!    reprojects with previous VP, outputs an `RG16Float` velocity texture.
//! 2. **Blur pass** — samples the scene along the velocity vector (N samples),
//!    averages. Reads from the scene HDR texture, writes to a blur output
//!    texture (`Rgba16Float`).

use glam::Mat4;
use wgpu::util::DeviceExt;

// ── Configuration ──

/// High-level motion blur configuration.
#[derive(Debug, Clone, Copy)]
pub struct MotionBlurConfig {
    /// Number of directional samples along the velocity vector.
    pub samples: u32,
    /// Multiplier applied to the computed velocity (artistic control).
    pub intensity: f32,
    /// Maximum velocity magnitude in pixels before clamping.
    pub max_velocity: f32,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            samples: 8,
            intensity: 1.0,
            max_velocity: 40.0,
        }
    }
}

// ── GPU uniform structs ──

/// Velocity-pass uniforms: inverse VP, previous VP, and viewport dimensions.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct VelocityParams {
    pub inv_view_projection: [[f32; 4]; 4],
    pub prev_view_projection: [[f32; 4]; 4],
    pub viewport: [f32; 4], // width, height, 1/w, 1/h
}

/// Blur-pass uniforms: tuning knobs.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlurParams {
    pub max_velocity: f32,
    pub intensity: f32,
    pub sample_count: f32,
    pub _pad: f32,
}

// ── Shaders ──

pub(crate) const VELOCITY_SHADER: &str = r#"
// Fullscreen triangle + velocity reconstruction from depth.

struct VelocityParams {
    inv_view_projection: mat4x4<f32>,
    prev_view_projection: mat4x4<f32>,
    viewport: vec4<f32>,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var point_samp: sampler;
@group(0) @binding(2) var<uniform> params: VelocityParams;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle covering clip space.
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    var out: VsOut;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let tex_size = vec2<i32>(textureDimensions(depth_tex));
    let coord = vec2<i32>(in.uv * vec2<f32>(tex_size));
    let depth = textureLoad(depth_tex, coord, 0);

    // Reconstruct NDC (wgpu: y-down in framebuffer, but UV is already correct).
    let ndc = vec3<f32>(in.uv.x * 2.0 - 1.0, -(in.uv.y * 2.0 - 1.0), depth);

    // Reconstruct world position.
    let world_h = params.inv_view_projection * vec4<f32>(ndc, 1.0);
    let world = world_h.xyz / world_h.w;

    // Reproject into previous frame clip space.
    let prev_clip = params.prev_view_projection * vec4<f32>(world, 1.0);
    let prev_ndc = prev_clip.xy / prev_clip.w;

    // Velocity in UV space (half NDC delta).
    let velocity = (ndc.xy - prev_ndc) * 0.5;

    return vec4<f32>(velocity, 0.0, 1.0);
}
"#;

pub(crate) const MOTION_BLUR_SHADER: &str = r#"
// Fullscreen triangle + directional blur along velocity.

struct BlurParams {
    max_velocity: f32,
    intensity: f32,
    sample_count: f32,
    _pad: f32,
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var velocity_tex: texture_2d<f32>;
@group(0) @binding(2) var linear_samp: sampler;
@group(0) @binding(3) var<uniform> params: BlurParams;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    var out: VsOut;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let vel_raw = textureSample(velocity_tex, linear_samp, in.uv).rg;
    let tex_size = vec2<f32>(textureDimensions(scene_tex));

    // Convert UV-space velocity to pixel-space, clamp magnitude.
    var vel_px = vel_raw * tex_size;
    let mag = length(vel_px);
    if mag > params.max_velocity {
        vel_px = vel_px * (params.max_velocity / mag);
    }

    // Scale by intensity, convert back to UV space.
    let vel_uv = (vel_px * params.intensity) / tex_size;

    let n = i32(params.sample_count);
    var color = vec4<f32>(0.0);
    for (var i = 0; i < n; i = i + 1) {
        let t = (f32(i) / f32(n - 1)) - 0.5; // -0.5 .. +0.5
        let sample_uv = in.uv + vel_uv * t;
        color = color + textureSample(scene_tex, linear_samp, sample_uv);
    }
    color = color / f32(n);

    return color;
}
"#;

// ── MotionBlurPass ──

/// Camera motion blur post-process.
///
/// Owns velocity + blur textures, pipelines, and bind groups. Call
/// [`MotionBlurPass::encode`] each frame to run the two fullscreen passes.
pub struct MotionBlurPass {
    pub config: MotionBlurConfig,

    // Velocity texture (RG16Float)
    velocity_texture: wgpu::Texture,
    velocity_view: wgpu::TextureView,
    velocity_sample_view: wgpu::TextureView,

    // Blur output texture (Rgba16Float, same as scene HDR)
    blur_texture: wgpu::Texture,
    pub(crate) result_view: wgpu::TextureView,
    blur_sample_view: wgpu::TextureView,

    // Pipelines
    velocity_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,

    // Bind group layouts
    velocity_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group_layout: wgpu::BindGroupLayout,

    // Bind groups (rebuilt on resize or texture change)
    velocity_bind_group: wgpu::BindGroup,
    blur_bind_group: wgpu::BindGroup,

    // Params
    velocity_params_buffer: wgpu::Buffer,
    blur_params_buffer: wgpu::Buffer,

    // Samplers
    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,

    // Dimensions
    width: u32,
    height: u32,
}

impl MotionBlurPass {
    /// Create the motion blur post-process (pipelines, textures, bind groups).
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let config = MotionBlurConfig::default();

        // ── Samplers ──

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("motion_blur_linear_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("motion_blur_point_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ── Textures ──

        let (velocity_texture, velocity_view, velocity_sample_view) =
            create_velocity_texture(device, width, height);

        let (blur_texture, result_view, blur_sample_view) =
            create_blur_texture(device, width, height);

        // ── Uniform buffers ──

        let velocity_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("motion_blur_velocity_params"),
                contents: bytemuck::bytes_of(&VelocityParams {
                    inv_view_projection: Mat4::IDENTITY.to_cols_array_2d(),
                    prev_view_projection: Mat4::IDENTITY.to_cols_array_2d(),
                    viewport: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let blur_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("motion_blur_blur_params"),
                contents: bytemuck::bytes_of(&BlurParams {
                    max_velocity: config.max_velocity,
                    intensity: config.intensity,
                    sample_count: config.samples as f32,
                    _pad: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ── Bind group layouts ──

        let velocity_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motion_blur_velocity_bgl"),
                entries: &[
                    // binding 0: depth texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: point sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // binding 2: velocity params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<VelocityParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("motion_blur_blur_bgl"),
                entries: &[
                    // binding 0: scene HDR texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: velocity texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2: linear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // binding 3: blur params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<BlurParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        // ── Pipelines ──

        let velocity_pipeline = create_fullscreen_pipeline(
            device,
            &velocity_bind_group_layout,
            VELOCITY_SHADER,
            "motion_blur_velocity",
            wgpu::TextureFormat::Rg16Float,
        );

        let blur_pipeline = create_fullscreen_pipeline(
            device,
            &blur_bind_group_layout,
            MOTION_BLUR_SHADER,
            "motion_blur_blur",
            wgpu::TextureFormat::Rgba16Float,
        );

        // ── Initial bind groups (placeholder — will be rebuilt before first use) ──

        let placeholder_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("motion_blur_placeholder_depth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let placeholder_depth_view = placeholder_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let velocity_bind_group = create_velocity_bind_group(
            device,
            &velocity_bind_group_layout,
            &placeholder_depth_view,
            &point_sampler,
            &velocity_params_buffer,
        );

        let blur_bind_group = create_blur_bind_group(
            device,
            &blur_bind_group_layout,
            &blur_sample_view, // scene placeholder
            &velocity_sample_view,
            &linear_sampler,
            &blur_params_buffer,
        );

        Self {
            config,
            velocity_texture,
            velocity_view,
            velocity_sample_view,
            blur_texture,
            result_view,
            blur_sample_view,
            velocity_pipeline,
            blur_pipeline,
            velocity_bind_group_layout,
            blur_bind_group_layout,
            velocity_bind_group,
            blur_bind_group,
            velocity_params_buffer,
            blur_params_buffer,
            linear_sampler,
            point_sampler,
            width,
            height,
        }
    }

    /// Rebuild velocity and blur pipelines with new shader sources.
    #[cfg(feature = "hot-reload")]
    pub fn rebuild_pipelines(&mut self, device: &wgpu::Device, velocity_src: &str, blur_src: &str) {
        self.velocity_pipeline = create_fullscreen_pipeline(
            device,
            &self.velocity_bind_group_layout,
            velocity_src,
            "motion_blur_velocity",
            wgpu::TextureFormat::Rg16Float,
        );
        self.blur_pipeline = create_fullscreen_pipeline(
            device,
            &self.blur_bind_group_layout,
            blur_src,
            "motion_blur_blur",
            wgpu::TextureFormat::Rgba16Float,
        );
    }

    /// Recreate velocity and blur textures after a window resize.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        let (vt, vv, vsv) = create_velocity_texture(device, width, height);
        self.velocity_texture = vt;
        self.velocity_view = vv;
        self.velocity_sample_view = vsv;

        let (bt, rv, bsv) = create_blur_texture(device, width, height);
        self.blur_texture = bt;
        self.result_view = rv;
        self.blur_sample_view = bsv;
    }

    /// Rebuild bind groups when external textures (depth, scene) change.
    ///
    /// Must be called at least once before the first [`encode`](Self::encode)
    /// and after every [`resize`](Self::resize).
    pub fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        scene_view: &wgpu::TextureView,
    ) {
        self.velocity_bind_group = create_velocity_bind_group(
            device,
            &self.velocity_bind_group_layout,
            depth_view,
            &self.point_sampler,
            &self.velocity_params_buffer,
        );

        self.blur_bind_group = create_blur_bind_group(
            device,
            &self.blur_bind_group_layout,
            scene_view,
            &self.velocity_sample_view,
            &self.linear_sampler,
            &self.blur_params_buffer,
        );
    }

    /// Encode both fullscreen passes (velocity reconstruction + blur).
    ///
    /// Writes the blurred result into an internal `Rgba16Float` texture
    /// accessible via [`result_view`](Self::result_view).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        _depth_view: &wgpu::TextureView,
        _scene_view: &wgpu::TextureView,
        inv_vp: Mat4,
        prev_vp: Mat4,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        // ── Upload velocity params ──
        let w = viewport_width as f32;
        let h = viewport_height as f32;
        queue.write_buffer(
            &self.velocity_params_buffer,
            0,
            bytemuck::bytes_of(&VelocityParams {
                inv_view_projection: inv_vp.to_cols_array_2d(),
                prev_view_projection: prev_vp.to_cols_array_2d(),
                viewport: [w, h, 1.0 / w, 1.0 / h],
            }),
        );

        // ── Upload blur params ──
        queue.write_buffer(
            &self.blur_params_buffer,
            0,
            bytemuck::bytes_of(&BlurParams {
                max_velocity: self.config.max_velocity,
                intensity: self.config.intensity,
                sample_count: self.config.samples as f32,
                _pad: 0.0,
            }),
        );

        // ── Pass 1: Velocity ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motion_blur_velocity_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.velocity_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.velocity_pipeline);
            pass.set_bind_group(0, &self.velocity_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Pass 2: Blur ──
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motion_blur_blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.result_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// The blurred output texture view (`Rgba16Float`).
    pub fn result_view(&self) -> &wgpu::TextureView {
        &self.result_view
    }
}

// ── Helpers ──

/// Create the RG16Float velocity texture and its views.
fn create_velocity_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("motion_blur_velocity_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let render_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("motion_blur_velocity_render_view"),
        ..Default::default()
    });

    let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("motion_blur_velocity_sample_view"),
        ..Default::default()
    });

    (texture, render_view, sample_view)
}

/// Create the Rgba16Float blur output texture and its views.
fn create_blur_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("motion_blur_output_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let render_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("motion_blur_result_view"),
        ..Default::default()
    });

    let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("motion_blur_output_sample_view"),
        ..Default::default()
    });

    (texture, render_view, sample_view)
}

/// Build a fullscreen render pipeline (vertex + fragment, no depth, single
/// color target).
fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    shader_source: &str,
    label: &str,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_pipeline_layout")),
        bind_group_layouts: &[bind_group_layout],
        immediate_size: 0,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label}_shader")),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label}_pipeline")),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Build the velocity-pass bind group.
fn create_velocity_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
    point_sampler: &wgpu::Sampler,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("motion_blur_velocity_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(point_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

/// Build the blur-pass bind group.
fn create_blur_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene_view: &wgpu::TextureView,
    velocity_view: &wgpu::TextureView,
    linear_sampler: &wgpu::Sampler,
    params_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("motion_blur_blur_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(velocity_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(linear_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    })
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velocity_params_size() {
        // 2 × mat4x4 (128 bytes each) + 1 × vec4 (16 bytes) = 144 bytes.
        assert_eq!(size_of::<VelocityParams>(), 144);
    }

    #[test]
    fn blur_params_size() {
        // 4 × f32 = 16 bytes.
        assert_eq!(size_of::<BlurParams>(), 16);
    }

    #[test]
    fn motion_blur_config_default() {
        let cfg = MotionBlurConfig::default();
        assert_eq!(cfg.samples, 8);
        assert!((cfg.intensity - 1.0).abs() < f32::EPSILON);
        assert!((cfg.max_velocity - 40.0).abs() < f32::EPSILON);
    }
}
