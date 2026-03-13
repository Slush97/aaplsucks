//! Screen-Space Ambient Occlusion — subtle contact darkening in concavities.
//!
//! Two fullscreen post-process passes reading the depth buffer:
//!
//! 1. **SSAO pass** — hemisphere kernel sampling with depth-reconstructed normals
//!    (`cross(dpdx, dpdy)`), outputs an R8Unorm occlusion texture.
//! 2. **Bilateral blur** — 4x4 box blur to smooth the raw occlusion result.
//!
//! No geometry normal prepass is required — normals are reconstructed from depth.

use glam::Mat4;

/// Maximum kernel size (number of hemisphere samples).
const MAX_KERNEL_SIZE: usize = 64;

/// Noise texture dimension (4x4 random rotation vectors).
const NOISE_DIM: u32 = 4;

// ── Config ──

/// Public configuration for SSAO quality and appearance.
#[derive(Debug, Clone, Copy)]
pub struct SsaoConfig {
    /// Hemisphere sample radius in view-space units.
    pub radius: f32,
    /// Depth bias to prevent self-occlusion on flat surfaces.
    pub bias: f32,
    /// Multiplier for the final occlusion strength.
    pub intensity: f32,
    /// Number of hemisphere kernel samples (clamped to 1..=64).
    pub kernel_size: u32,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            radius: 0.5,
            bias: 0.025,
            intensity: 1.0,
            kernel_size: 32,
        }
    }
}

// ── GPU params ──

/// GPU uniform block for the SSAO pass.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SsaoParams {
    /// Camera projection matrix (for projecting view-space samples to UV).
    pub projection: [[f32; 4]; 4],
    /// Inverse projection matrix (for reconstructing view-space position from depth).
    pub inv_projection: [[f32; 4]; 4],
    /// `viewport_size / noise_texture_size` — for tiling the noise texture.
    pub noise_scale: [f32; 2],
    /// Hemisphere sample radius.
    pub radius: f32,
    /// Depth bias.
    pub bias: f32,
    /// Occlusion intensity multiplier.
    pub intensity: f32,
    /// Number of kernel samples (as f32 for the shader).
    pub kernel_size: f32,
    /// Padding to 16-byte alignment.
    pub _pad: [f32; 2],
}

// ── SSAO pass ──

/// Screen-Space Ambient Occlusion post-process pass.
///
/// Owns the SSAO render pipeline, blur pipeline, occlusion textures, noise
/// texture, and kernel buffer. Created once and resized when the viewport changes.
pub struct SsaoPass {
    /// Current SSAO configuration.
    pub config: SsaoConfig,

    // SSAO raw output.
    occlusion_texture: wgpu::Texture,
    pub(crate) occlusion_view: wgpu::TextureView,
    occlusion_sample_view: wgpu::TextureView,

    // Blurred output (final result composited into lighting).
    blur_texture: wgpu::Texture,
    pub(crate) result_view: wgpu::TextureView,
    blur_sample_view: wgpu::TextureView,

    // Noise texture (4x4 random rotation vectors, Rgba8Unorm).
    #[allow(dead_code)]
    noise_texture: wgpu::Texture,
    noise_view: wgpu::TextureView,

    // Kernel (array of vec4 hemisphere samples in a uniform buffer).
    kernel_buffer: wgpu::Buffer,

    // Params buffer (SsaoParams uniform).
    params_buffer: wgpu::Buffer,

    // Pipelines.
    ssao_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,

    // Bind group layouts.
    ssao_bind_group_layout: wgpu::BindGroupLayout,
    ssao_bind_group: wgpu::BindGroup,
    blur_bind_group_layout: wgpu::BindGroupLayout,
    blur_bind_group: wgpu::BindGroup,

    // Samplers.
    point_sampler: wgpu::Sampler,
    repeat_sampler: wgpu::Sampler,

    // Whether noise texture has been uploaded.
    noise_uploaded: std::cell::Cell<bool>,

    // Dummy 1x1 depth texture for initial/resize blur bind groups.
    #[allow(dead_code)]
    dummy_depth_texture: wgpu::Texture,
    dummy_depth_view: wgpu::TextureView,

    // Current dimensions.
    width: u32,
    height: u32,
}

impl SsaoPass {
    /// Create a new SSAO pass for the given viewport dimensions.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let config = SsaoConfig::default();
        let w = width.max(1);
        let h = height.max(1);

        // ── Kernel samples ──
        let kernel = generate_kernel(config.kernel_size as usize);
        let kernel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_ssao_kernel"),
            size: (MAX_KERNEL_SIZE * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut mapping = kernel_buffer.slice(..).get_mapped_range_mut();
            let dst = bytemuck::cast_slice_mut::<u8, [f32; 4]>(&mut mapping);
            for (i, sample) in kernel.iter().enumerate() {
                dst[i] = *sample;
            }
        }
        kernel_buffer.unmap();

        // ── Noise texture ──
        let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("esox_ssao_noise"),
            size: wgpu::Extent3d {
                width: NOISE_DIM,
                height: NOISE_DIM,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let noise_view = noise_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Params buffer ──
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_ssao_params"),
            size: size_of::<SsaoParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Output textures ──
        let (occlusion_texture, occlusion_view, occlusion_sample_view) =
            create_r8_texture(device, w, h, "esox_ssao_occlusion");
        let (blur_texture, result_view, blur_sample_view) =
            create_r8_texture(device, w, h, "esox_ssao_blur");

        // ── Samplers ──
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("esox_ssao_point_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let repeat_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("esox_ssao_repeat_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ── Bind group layouts ──
        let ssao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esox_ssao_bg_layout"),
                entries: &[
                    // binding 0: depth texture (texture_depth_2d)
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
                    // binding 1: noise texture (texture_2d<f32>)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2: kernel buffer (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3: params buffer (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 4: point sampler (non-filtering)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // binding 5: repeat sampler (for noise tiling)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let blur_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esox_ssao_blur_bg_layout"),
                entries: &[
                    // binding 0: occlusion texture (raw SSAO output)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // binding 2: params buffer (texel_size lives in params)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3: depth texture (for bilateral blur — depth-aware edge preservation)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // ── Pipelines ──
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_SHADER.into()),
        });

        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_ssao_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(SSAO_BLUR_SHADER.into()),
        });

        let ssao_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("esox_ssao_pipeline_layout"),
                bind_group_layouts: &[&ssao_bind_group_layout],
                immediate_size: 0,
            });

        let blur_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("esox_ssao_blur_pipeline_layout"),
                bind_group_layouts: &[&blur_bind_group_layout],
                immediate_size: 0,
            });

        let ssao_pipeline = create_fullscreen_pipeline(
            device,
            &ssao_pipeline_layout,
            &ssao_shader,
            wgpu::TextureFormat::R8Unorm,
            "esox_ssao_pipeline",
        );

        let blur_pipeline = create_fullscreen_pipeline(
            device,
            &blur_pipeline_layout,
            &blur_shader,
            wgpu::TextureFormat::R8Unorm,
            "esox_ssao_blur_pipeline",
        );

        // ── Initial bind groups (with dummy depth — rebuilt in encode()) ──
        let dummy_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("esox_ssao_dummy_depth"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_depth_view = dummy_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let ssao_bind_group = create_ssao_bind_group(
            device,
            &ssao_bind_group_layout,
            &dummy_depth_view,
            &noise_view,
            &kernel_buffer,
            &params_buffer,
            &point_sampler,
            &repeat_sampler,
        );

        let blur_bind_group = create_blur_bind_group(
            device,
            &blur_bind_group_layout,
            &occlusion_sample_view,
            &point_sampler,
            &params_buffer,
            &dummy_depth_view,
        );

        Self {
            config,
            occlusion_texture,
            occlusion_view,
            occlusion_sample_view,
            blur_texture,
            result_view,
            blur_sample_view,
            noise_texture,
            noise_view,
            kernel_buffer,
            params_buffer,
            ssao_pipeline,
            blur_pipeline,
            ssao_bind_group_layout,
            ssao_bind_group,
            blur_bind_group_layout,
            blur_bind_group,
            point_sampler,
            repeat_sampler,
            noise_uploaded: std::cell::Cell::new(false),
            dummy_depth_texture: dummy_depth,
            dummy_depth_view,
            width: w,
            height: h,
        }
    }

    /// Rebuild SSAO and blur pipelines with new shader sources.
    #[cfg(feature = "hot-reload")]
    pub fn rebuild_pipelines(&mut self, device: &wgpu::Device, ssao_src: &str, blur_src: &str) {
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(ssao_src.into()),
        });
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_ssao_blur_shader"),
            source: wgpu::ShaderSource::Wgsl(blur_src.into()),
        });

        let ssao_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("esox_ssao_pipeline_layout"),
            bind_group_layouts: &[&self.ssao_bind_group_layout],
            immediate_size: 0,
        });
        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("esox_ssao_blur_pipeline_layout"),
            bind_group_layouts: &[&self.blur_bind_group_layout],
            immediate_size: 0,
        });

        self.ssao_pipeline = create_fullscreen_pipeline(
            device,
            &ssao_pipeline_layout,
            &ssao_shader,
            wgpu::TextureFormat::R8Unorm,
            "esox_ssao_pipeline",
        );
        self.blur_pipeline = create_fullscreen_pipeline(
            device,
            &blur_pipeline_layout,
            &blur_shader,
            wgpu::TextureFormat::R8Unorm,
            "esox_ssao_blur_pipeline",
        );
    }

    /// Recreate textures and bind groups for a new viewport size.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if w == self.width && h == self.height {
            return;
        }
        self.width = w;
        self.height = h;

        let (occ_tex, occ_view, occ_sample) =
            create_r8_texture(device, w, h, "esox_ssao_occlusion");
        self.occlusion_texture = occ_tex;
        self.occlusion_view = occ_view;
        self.occlusion_sample_view = occ_sample;

        let (blur_tex, res_view, blur_sample) =
            create_r8_texture(device, w, h, "esox_ssao_blur");
        self.blur_texture = blur_tex;
        self.result_view = res_view;
        self.blur_sample_view = blur_sample;

        // Rebuild blur bind group (references the new occlusion texture).
        self.blur_bind_group = create_blur_bind_group(
            device,
            &self.blur_bind_group_layout,
            &self.occlusion_sample_view,
            &self.point_sampler,
            &self.params_buffer,
            &self.dummy_depth_view,
        );
    }

    /// Update params, run the SSAO pass, then run the blur pass.
    ///
    /// `depth_view` must reference a `Depth32Float` texture with `TEXTURE_BINDING` usage.
    /// `projection` is the camera's perspective projection matrix for this frame.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        depth_view: &wgpu::TextureView,
        projection: Mat4,
    ) {
        // Upload noise texture once.
        if !self.noise_uploaded.get() {
            let noise_data = generate_noise();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.noise_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &noise_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(NOISE_DIM * 4),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: NOISE_DIM,
                    height: NOISE_DIM,
                    depth_or_array_layers: 1,
                },
            );
            self.noise_uploaded.set(true);
        }

        // ── Update params ──
        let inv_projection = projection.inverse();
        let params = SsaoParams {
            projection: projection.to_cols_array_2d(),
            inv_projection: inv_projection.to_cols_array_2d(),
            noise_scale: [
                self.width as f32 / NOISE_DIM as f32,
                self.height as f32 / NOISE_DIM as f32,
            ],
            radius: self.config.radius,
            bias: self.config.bias,
            intensity: self.config.intensity,
            kernel_size: self.config.kernel_size.min(MAX_KERNEL_SIZE as u32) as f32,
            _pad: [0.0; 2],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // ── Rebuild SSAO bind group with this frame's depth view ──
        self.ssao_bind_group = create_ssao_bind_group(
            device,
            &self.ssao_bind_group_layout,
            depth_view,
            &self.noise_view,
            &self.kernel_buffer,
            &self.params_buffer,
            &self.point_sampler,
            &self.repeat_sampler,
        );

        // ── Pass 1: SSAO ──
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("esox_ssao_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.occlusion_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rpass.set_pipeline(&self.ssao_pipeline);
            rpass.set_bind_group(0, &self.ssao_bind_group, &[]);
            rpass.draw(0..3, 0..1); // fullscreen triangle
        }

        // ── Rebuild blur bind group (occlusion texture may have been recreated) ──
        self.blur_bind_group = create_blur_bind_group(
            device,
            &self.blur_bind_group_layout,
            &self.occlusion_sample_view,
            &self.point_sampler,
            &self.params_buffer,
            depth_view,
        );

        // ── Pass 2: Blur ──
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("esox_ssao_blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.result_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            rpass.set_pipeline(&self.blur_pipeline);
            rpass.set_bind_group(0, &self.blur_bind_group, &[]);
            rpass.draw(0..3, 0..1); // fullscreen triangle
        }
    }

    /// Returns the blurred occlusion texture view (the final SSAO result).
    ///
    /// Sample this in the lighting pass and multiply ambient/indirect terms by
    /// the R channel value (0 = fully occluded, 1 = fully visible).
    pub fn result_view(&self) -> &wgpu::TextureView {
        &self.result_view
    }
}

// ── Helpers ──

/// Create an R8Unorm texture with both RENDER_ATTACHMENT and TEXTURE_BINDING usage.
fn create_r8_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    // Render target view.
    let render_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{label}_render_view")),
        ..Default::default()
    });
    // Sampling view (same format, but semantically separate for clarity).
    let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(&format!("{label}_sample_view")),
        ..Default::default()
    });
    (texture, render_view, sample_view)
}

/// Create the SSAO bind group.
fn create_ssao_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
    noise_view: &wgpu::TextureView,
    kernel_buffer: &wgpu::Buffer,
    params_buffer: &wgpu::Buffer,
    point_sampler: &wgpu::Sampler,
    repeat_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("esox_ssao_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(noise_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: kernel_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(point_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(repeat_sampler),
            },
        ],
    })
}

/// Create the blur bind group.
fn create_blur_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    occlusion_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    params_buffer: &wgpu::Buffer,
    depth_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("esox_ssao_blur_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(occlusion_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
        ],
    })
}

/// Create a fullscreen-triangle render pipeline (no vertex buffers, no depth).
fn create_fullscreen_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}

// ── Kernel generation ──

/// Simple deterministic hash for pseudo-random sample generation.
/// Based on a variant of the PCG hash (no external RNG dependency).
fn hash_u32(mut state: u32) -> u32 {
    state = state.wrapping_mul(747796405).wrapping_add(2891336453);
    state = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    (state >> 22) ^ state
}

/// Convert a u32 hash to a float in [0, 1).
fn hash_to_f32(h: u32) -> f32 {
    (h & 0x00FF_FFFF) as f32 / 16_777_216.0
}

/// Generate cosine-weighted hemisphere kernel samples.
///
/// Returns up to `MAX_KERNEL_SIZE` samples as `[f32; 4]` where xyz is the
/// hemisphere direction (positive Z) and w is unused (0.0).
///
/// Samples are biased toward the surface: scale = lerp(0.1, 1.0, (i/n)^2).
pub(crate) fn generate_kernel(count: usize) -> Vec<[f32; 4]> {
    let n = count.clamp(1, MAX_KERNEL_SIZE);
    let mut kernel = Vec::with_capacity(n);

    for i in 0..n {
        // Two hash calls per sample for x, y; one more for z.
        let h0 = hash_u32((i as u32).wrapping_mul(3).wrapping_add(0));
        let h1 = hash_u32((i as u32).wrapping_mul(3).wrapping_add(1));
        let h2 = hash_u32((i as u32).wrapping_mul(3).wrapping_add(2));

        // Map to [-1, 1] for x/y, [0, 1] for z (hemisphere).
        let x = hash_to_f32(h0) * 2.0 - 1.0;
        let y = hash_to_f32(h1) * 2.0 - 1.0;
        let z = hash_to_f32(h2); // positive hemisphere

        // Normalize.
        let len = (x * x + y * y + z * z).sqrt();
        if len < 1e-6 {
            kernel.push([0.0, 0.0, 1.0, 0.0]);
            continue;
        }
        let (nx, ny, nz) = (x / len, y / len, z / len);

        // Ensure positive z (hemisphere).
        let nz = nz.abs().max(0.01);
        let len2 = (nx * nx + ny * ny + nz * nz).sqrt();
        let (nx, ny, nz) = (nx / len2, ny / len2, nz / len2);

        // Scale: bias samples closer to the origin (surface).
        let t = i as f32 / n as f32;
        let scale = lerp(0.1, 1.0, t * t);

        kernel.push([nx * scale, ny * scale, nz * scale, 0.0]);
    }

    kernel
}

/// Linear interpolation.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Generate 4x4 noise texture data (Rgba8Unorm, 64 bytes).
///
/// Each texel contains a random tangent-space rotation vector in xy,
/// with z=0 and a=255 (opaque). The rotation is encoded as a unit vector
/// in the tangent plane, packed into [0, 255].
fn generate_noise() -> Vec<u8> {
    let count = (NOISE_DIM * NOISE_DIM) as usize;
    let mut data = Vec::with_capacity(count * 4);

    for i in 0..count {
        let h0 = hash_u32((i as u32).wrapping_add(1000));
        let h1 = hash_u32((i as u32).wrapping_add(2000));

        // Random rotation in tangent plane: unit vector (x, y, 0).
        let angle = hash_to_f32(h0) * std::f32::consts::TAU;
        let (sin_a, cos_a) = angle.sin_cos();

        // Pack [-1, 1] -> [0, 255].
        let r = ((cos_a * 0.5 + 0.5) * 255.0) as u8;
        let g = ((sin_a * 0.5 + 0.5) * 255.0) as u8;
        let b = 0u8; // z = 0 (tangent plane rotation)
        let a = 255u8;

        // Use second hash for slight variation (unused but fills the channel).
        let _ = h1;

        data.push(r);
        data.push(g);
        data.push(b);
        data.push(a);
    }

    data
}

// ── Shaders ──

/// SSAO fragment shader (WGSL).
///
/// Fullscreen triangle vertex shader + hemisphere kernel sampling with
/// depth-reconstructed normals.
pub(crate) const SSAO_SHADER: &str = r#"
// ── Bindings ──

@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var t_noise: texture_2d<f32>;
@group(0) @binding(2) var<uniform> kernel: array<vec4<f32>, 64>;
@group(0) @binding(3) var<uniform> params: SsaoParams;
@group(0) @binding(4) var s_point: sampler;
@group(0) @binding(5) var s_repeat: sampler;

struct SsaoParams {
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    noise_scale: vec2<f32>,
    radius: f32,
    bias: f32,
    intensity: f32,
    kernel_size: f32,
    _pad: vec2<f32>,
}

// ── Vertex shader — fullscreen triangle ──

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOutput {
    // Generate a fullscreen triangle from vertex_index (0, 1, 2).
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// ── Fragment shader ──

/// Reconstruct view-space position from depth and UV.
fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // UV to clip space: [0,1] -> [-1,1], flip Y (UV y=0 is top, clip y=+1 is top).
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let view_h = params.inv_projection * clip;
    return view_h.xyz / view_h.w;
}

/// Reconstruct view-space position from an integer pixel coordinate.
fn view_pos_at(pixel: vec2<i32>, tex_size: vec2<f32>) -> vec3<f32> {
    let uv = (vec2<f32>(pixel) + 0.5) / tex_size;
    let depth = textureLoad(t_depth, pixel, 0);
    return reconstruct_view_pos(uv, depth);
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) f32 {
    let uv = in.uv;
    let tex_size = vec2<f32>(textureDimensions(t_depth));
    let pixel = vec2<i32>(in.position.xy);

    // Sample depth at this fragment.
    let depth = textureLoad(t_depth, pixel, 0);

    // Early out for far plane (no geometry).
    if depth >= 1.0 {
        return 1.0;
    }

    // Reconstruct view-space position.
    let view_pos = reconstruct_view_pos(uv, depth);

    // Reconstruct normal from depth — pick the shorter differential per axis
    // to avoid crossing depth discontinuities at geometry edges.
    let left   = view_pos_at(pixel + vec2(-1, 0), tex_size);
    let right  = view_pos_at(pixel + vec2( 1, 0), tex_size);
    let top    = view_pos_at(pixel + vec2( 0,-1), tex_size);
    let bottom = view_pos_at(pixel + vec2( 0, 1), tex_size);

    let dl = view_pos - left;
    let dr = right - view_pos;
    let dt = view_pos - top;
    let db = bottom - view_pos;

    // Skip SSAO at large depth discontinuities (geometry silhouette edges)
    // where the reconstructed normal is unreliable.
    let min_dz = min(min(abs(dl.z), abs(dr.z)), min(abs(dt.z), abs(db.z)));
    let max_dz = max(max(abs(dl.z), abs(dr.z)), max(abs(dt.z), abs(db.z)));
    if max_dz > abs(view_pos.z) * 0.1 && max_dz > min_dz * 10.0 {
        return 1.0;
    }

    let ddx = select(dr, dl, abs(dl.z) < abs(dr.z));
    let ddy = select(db, dt, abs(dt.z) < abs(db.z));

    let normal = normalize(cross(ddy, ddx));

    // Sample noise for random tangent rotation.
    let noise_uv = uv * params.noise_scale;
    let noise_val = textureSample(t_noise, s_repeat, noise_uv).rg * 2.0 - 1.0;
    let random_vec = vec3<f32>(noise_val, 0.0);

    // Build TBN matrix (Gram-Schmidt).
    let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    // Accumulate occlusion.
    var occlusion = 0.0;
    let sample_count = i32(params.kernel_size);

    for (var i = 0; i < sample_count; i++) {
        // Rotate kernel sample into view space via TBN.
        let sample_dir = tbn * kernel[i].xyz;
        let sample_pos = view_pos + sample_dir * params.radius;

        // Project sample to screen space.
        let proj = params.projection * vec4<f32>(sample_pos, 1.0);
        var sample_uv = proj.xy / proj.w;
        sample_uv = sample_uv * 0.5 + 0.5;
        sample_uv.y = 1.0 - sample_uv.y;

        // Skip samples that project outside the screen — out-of-bounds
        // textureLoad returns depth=0 (near plane), causing false occlusion.
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }

        // Sample depth at projected position.
        let sample_screen = vec2<i32>(sample_uv * tex_size);
        let sample_depth = textureLoad(t_depth, sample_screen, 0);

        // Skip samples that land on the far plane (sky).
        if sample_depth >= 1.0 {
            continue;
        }

        let sample_view = reconstruct_view_pos(sample_uv, sample_depth);

        // Range check: avoid occlusion from distant geometry.
        let range_check = smoothstep(0.0, 1.0, params.radius / abs(view_pos.z - sample_view.z));

        // Occlusion test: is the sample behind the surface?
        if sample_view.z >= sample_pos.z + params.bias {
            occlusion += range_check;
        }
    }

    occlusion = occlusion / f32(sample_count);
    let ao = 1.0 - (occlusion * params.intensity);
    return clamp(ao, 0.0, 1.0);
}
"#;

/// SSAO blur shader (WGSL) — bilateral 4x4 blur on the R8 occlusion texture.
///
/// Depth-aware: rejects samples across depth discontinuities to prevent
/// dark SSAO halos from bleeding across geometry edges (e.g. ground → sky).
pub(crate) const SSAO_BLUR_SHADER: &str = r#"
// ── Bindings ──

@group(0) @binding(0) var t_occlusion: texture_2d<f32>;
@group(0) @binding(1) var s_point: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;
@group(0) @binding(3) var t_depth: texture_depth_2d;

struct BlurParams {
    // We reuse the SsaoParams layout; noise_scale.xy encodes viewport dimensions.
    projection: mat4x4<f32>,
    inv_projection: mat4x4<f32>,
    noise_scale: vec2<f32>,
    radius: f32,
    bias: f32,
    intensity: f32,
    kernel_size: f32,
    _pad: vec2<f32>,
}

// ── Vertex shader — fullscreen triangle ──

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOutput {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    var out: VsOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

// ── Fragment shader — bilateral 4x4 blur ──

@fragment
fn fs_main(in: VsOutput) -> @location(0) f32 {
    let tex_size = vec2<f32>(textureDimensions(t_occlusion));
    let texel_size = 1.0 / tex_size;
    let uv = in.uv;
    let pixel = vec2<i32>(in.position.xy);

    let center_depth = textureLoad(t_depth, pixel, 0);

    // Depth threshold: reject samples that cross a depth discontinuity.
    // Use a relative threshold so it works at all distances.
    let depth_threshold = max(center_depth * 0.02, 0.0002);

    var result = 0.0;
    var total_weight = 0.0;

    for (var x = -2; x <= 2; x++) {
        for (var y = -2; y <= 2; y++) {
            let sample_pixel = pixel + vec2(x, y);
            let sample_depth = textureLoad(t_depth, sample_pixel, 0);
            let depth_diff = abs(center_depth - sample_depth);

            // Weight is 1.0 if depth is similar, 0.0 if across a discontinuity.
            let w = select(0.0, 1.0, depth_diff < depth_threshold);

            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let ao = textureSampleLevel(t_occlusion, s_point, uv + offset, 0.0).r;

            result += ao * w;
            total_weight += w;
        }
    }

    return result / max(total_weight, 1.0);
}
"#;

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssao_params_size() {
        // SsaoParams: 2 mat4x4 (128 bytes) + 2 f32 + f32 + f32 + f32 + f32 + 2 f32 = 160 bytes.
        assert_eq!(size_of::<SsaoParams>(), 160);
    }

    #[test]
    fn ssao_params_is_pod() {
        let p = SsaoParams {
            projection: [[0.0; 4]; 4],
            inv_projection: [[0.0; 4]; 4],
            noise_scale: [1.0, 1.0],
            radius: 0.5,
            bias: 0.025,
            intensity: 1.0,
            kernel_size: 32.0,
            _pad: [0.0; 2],
        };
        let _bytes: &[u8] = bytemuck::bytes_of(&p);
    }

    #[test]
    fn ssao_config_default() {
        let c = SsaoConfig::default();
        assert_eq!(c.radius, 0.5);
        assert_eq!(c.bias, 0.025);
        assert_eq!(c.intensity, 1.0);
        assert_eq!(c.kernel_size, 32);
    }

    #[test]
    fn kernel_generation() {
        let kernel = generate_kernel(32);
        assert_eq!(kernel.len(), 32);

        for (i, sample) in kernel.iter().enumerate() {
            let [x, y, z, w] = *sample;

            // w is unused.
            assert_eq!(w, 0.0, "sample {i}: w should be 0.0");

            // z must be positive (hemisphere).
            assert!(z > 0.0, "sample {i}: z={z} should be positive (hemisphere)");

            // Length should be reasonable (scaled between 0.1 and 1.0).
            let len = (x * x + y * y + z * z).sqrt();
            assert!(
                len > 0.05 && len <= 1.05,
                "sample {i}: length={len} should be in (0.05, 1.05]"
            );
        }
    }

    #[test]
    fn kernel_generation_clamped() {
        let kernel_0 = generate_kernel(0);
        assert_eq!(kernel_0.len(), 1, "kernel size 0 should clamp to 1");

        let kernel_big = generate_kernel(200);
        assert_eq!(
            kernel_big.len(),
            MAX_KERNEL_SIZE,
            "kernel size 200 should clamp to MAX_KERNEL_SIZE"
        );
    }

    #[test]
    fn noise_generation() {
        let noise = generate_noise();
        assert_eq!(noise.len(), (NOISE_DIM * NOISE_DIM * 4) as usize);

        // Every 4th byte (alpha) should be 255.
        for i in (3..noise.len()).step_by(4) {
            assert_eq!(noise[i], 255, "alpha at byte {i} should be 255");
        }
    }

    #[test]
    fn kernel_deterministic() {
        let a = generate_kernel(32);
        let b = generate_kernel(32);
        assert_eq!(a, b, "kernel generation should be deterministic");
    }
}
