//! Post-processing types and texture helpers for the 3D renderer.

use crate::bloom::BloomPass;
use crate::pipeline::GpuContext;
use super::camera::Camera;
use super::render_types::{DEPTH_FORMAT, HDR_FORMAT};
use super::shaders_embedded::COMPOSITE_SHADER_3D;

// ── Post-process config ──

/// Configuration for the 3D post-processing pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PostProcess3DConfig {
    /// Enable bloom (dual-Kawase).
    pub bloom_enabled: bool,
    /// Bloom intensity multiplier.
    pub bloom_intensity: f32,
    /// HDR luminance threshold — only pixels brighter than this bloom.
    /// Set to 0.0 to bloom everything (old behavior). Default: 1.0.
    pub bloom_threshold: f32,
    /// Soft knee width around the bloom threshold (smooth transition).
    /// 0.0 = hard cutoff, 0.5 = gentle ramp. Default: 0.5.
    pub bloom_soft_knee: f32,
    /// Enable ACES tone mapping.
    pub tone_map_enabled: bool,
    /// Enable SSAO.
    pub ssao_enabled: bool,
    /// Enable distance fog.
    pub fog_enabled: bool,
    /// Fog color (linear RGB). Default: warm haze.
    pub fog_color: [f32; 3],
    /// Distance at which fog begins (world units). Default: 50.0.
    pub fog_start: f32,
    /// Distance at which fog is fully opaque (world units). Default: 200.0.
    pub fog_end: f32,
}

impl Default for PostProcess3DConfig {
    fn default() -> Self {
        Self {
            bloom_enabled: true,
            bloom_intensity: 0.3,
            bloom_threshold: 1.0,
            bloom_soft_knee: 0.5,
            tone_map_enabled: true,
            ssao_enabled: false,
            fog_enabled: false,
            fog_color: [0.75, 0.82, 0.90],
            fog_start: 50.0,
            fog_end: 200.0,
        }
    }
}

/// GPU params for the composite pass (48 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct CompositeParams3D {
    pub(super) bloom_intensity: f32,
    pub(super) tone_map: f32,
    pub(super) ssao_enabled: f32,
    pub(super) _pad0: f32,
    /// Fog color (rgb) + enabled flag (w).
    pub(super) fog_color_enabled: [f32; 4],
    /// Fog start (x), fog end (y), camera near (z), camera far (w).
    pub(super) fog_params: [f32; 4],
}

/// Internal state for the 3D post-process pipeline.
pub(super) struct PostProcess3D {
    /// Offscreen HDR color texture (1x, used as resolve target / sampling source).
    #[allow(dead_code)]
    pub(super) color_texture: wgpu::Texture,
    /// View for rendering into (RENDER_ATTACHMENT) — 1x when no MSAA, unused as
    /// direct render target when MSAA is active (resolve writes here instead).
    pub(super) color_view: wgpu::TextureView,
    /// View for sampling (TEXTURE_BINDING) — always 1x.
    pub(super) sample_view: wgpu::TextureView,
    /// MSAA render texture (sample_count > 1). Render pass writes here and
    /// hardware-resolves into `color_view`.
    #[allow(dead_code)]
    pub(super) msaa_color_texture: Option<wgpu::Texture>,
    pub(super) msaa_color_view: Option<wgpu::TextureView>,
    /// Bloom pass (reusing the 2D dual-Kawase implementation).
    pub(super) bloom_pass: BloomPass,
    /// Bloom downsample pipeline.
    pub(super) bloom_down_pipeline: wgpu::RenderPipeline,
    /// Bloom upsample pipeline.
    pub(super) bloom_up_pipeline: wgpu::RenderPipeline,
    /// Fallback black texture for when bloom is disabled.
    #[allow(dead_code)]
    pub(super) bloom_black_texture: wgpu::Texture,
    pub(super) bloom_black_view: wgpu::TextureView,
    /// Composite pipeline (fullscreen triangle: scene + bloom + SSAO -> surface).
    pub(super) composite_pipeline: wgpu::RenderPipeline,
    /// Composite bind group layout.
    pub(super) composite_bind_group_layout: wgpu::BindGroupLayout,
    /// Composite params buffer.
    pub(super) params_buffer: wgpu::Buffer,
    /// Linear sampler for sampling HDR textures.
    pub(super) linear_sampler: wgpu::Sampler,
    /// Current config.
    pub(super) config: PostProcess3DConfig,
    /// Current offscreen dimensions.
    pub(super) width: u32,
    pub(super) height: u32,
}

// ── Depth texture helper ──

pub(super) fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("esox_3d_depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create an HDR offscreen texture (1x) with both RENDER_ATTACHMENT and TEXTURE_BINDING usage.
/// When `sample_count > 1`, also creates an MSAA render texture that resolves into the 1x texture.
pub(super) fn create_hdr_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::TextureView,
    Option<wgpu::Texture>,
    Option<wgpu::TextureView>,
) {
    let size = wgpu::Extent3d {
        width: width.max(1),
        height: height.max(1),
        depth_or_array_layers: 1,
    };

    // 1x resolve / sampling texture (always created).
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("esox_3d_hdr_offscreen"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let color_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("esox_3d_hdr_color_view"),
        ..Default::default()
    });
    let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("esox_3d_hdr_sample_view"),
        ..Default::default()
    });

    // MSAA render texture (only when sample_count > 1).
    let (msaa_texture, msaa_view) = if sample_count > 1 {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("esox_3d_hdr_msaa"),
            size,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("esox_3d_hdr_msaa_view"),
            ..Default::default()
        });
        (Some(tex), Some(view))
    } else {
        (None, None)
    };

    (texture, color_view, sample_view, msaa_texture, msaa_view)
}

// ── Integration with Renderer3D ──

impl super::renderer::Renderer3D {
    /// Enable the post-processing pipeline (offscreen HDR + bloom + tone mapping).
    ///
    /// When enabled, the scene renders to an offscreen `Rgba16Float` texture and
    /// a composite pass blits the result to the surface with bloom and tone mapping.
    /// When disabled (default), rendering goes directly to the surface.
    pub fn enable_postprocess(&mut self, gpu: &GpuContext) {
        if self.postprocess.is_some() {
            return;
        }
        let device = &*gpu.device;
        let w = gpu.config.width.max(1);
        let h = gpu.config.height.max(1);

        let (color_texture, color_view, sample_view, msaa_color_texture, msaa_color_view) =
            create_hdr_texture(device, w, h, self.sample_count);
        let bloom_pass = BloomPass::new(device, w, h, HDR_FORMAT, &sample_view);

        // Create bloom pipelines.
        let bloom_bgl = bloom_pass.bind_group_layout();
        let bloom_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("esox_3d_bloom_pipeline_layout"),
            bind_group_layouts: &[bloom_bgl],
            immediate_size: 0,
        });
        let down_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_bloom_downsample"),
            source: wgpu::ShaderSource::Wgsl(crate::bloom::downsample_shader_source().into()),
        });
        let up_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_bloom_upsample"),
            source: wgpu::ShaderSource::Wgsl(crate::bloom::upsample_shader_source().into()),
        });

        let create_bloom_pipeline = |shader: &wgpu::ShaderModule, label: &str, blend: Option<wgpu::BlendState>| -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&bloom_pipeline_layout),
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
                        format: HDR_FORMAT,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let bloom_down_pipeline = create_bloom_pipeline(&down_shader, "esox_3d_bloom_down", None);
        // Upsample needs additive blending to accumulate onto destination mips.
        let bloom_up_pipeline = create_bloom_pipeline(&up_shader, "esox_3d_bloom_up", Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        }));

        let (bloom_black_texture, bloom_black_view) = crate::bloom::create_black_texture(device, &gpu.queue, HDR_FORMAT);

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esox_3d_composite_bgl"),
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
                    // binding 1: bloom texture
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
                    // binding 2: SSAO texture (R8Unorm — filterable)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 3: linear sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // binding 4: composite params
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<CompositeParams3D>() as u64,
                            ),
                        },
                        count: None,
                    },
                    // binding 5: depth texture (for fog)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
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

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("esox_3d_composite_pipeline_layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                immediate_size: 0,
            });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER_3D.into()),
        });

        let composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("esox_3d_composite_pipeline"),
                layout: Some(&composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &composite_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let params_buffer =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("esox_3d_composite_params"),
                size: size_of::<CompositeParams3D>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("esox_3d_composite_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        self.postprocess = Some(PostProcess3D {
            color_texture,
            color_view,
            sample_view,
            msaa_color_texture,
            msaa_color_view,
            bloom_pass,
            bloom_down_pipeline,
            bloom_up_pipeline,
            bloom_black_texture,
            bloom_black_view,
            composite_pipeline,
            composite_bind_group_layout,
            params_buffer,
            linear_sampler,
            config: PostProcess3DConfig::default(),
            width: w,
            height: h,
        });

        // Material pipelines must target the HDR offscreen format, not the
        // surface format, when postprocessing is enabled.
        self.surface_format = HDR_FORMAT;
        self.rebuild_pipeline_cache(device);
    }

    /// Encode the post-processing chain: depth resolve, SSAO, bloom, and composite.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_postprocess_chain(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        camera: &Camera,
        viewport_width: u32,
        viewport_height: u32,
    ) {
        // MSAA depth resolve (writes resolved 1x depth for SSAO).
        if let Some(resolve) = &self.depth_resolve_pass {
            resolve.encode(encoder, &self.depth_sample_view);
        }

        // SSAO (reads depth buffer, writes occlusion texture).
        if let Some(ssao) = &mut self.ssao_pass {
            let proj = camera.projection_matrix(
                viewport_width as f32 / viewport_height.max(1) as f32,
            );
            ssao.encode(&gpu.device, encoder, &gpu.queue, &self.depth_sample_view, proj);
        }

        // Bloom + composite (when post-processing is enabled).
        if let Some(pp) = &mut self.postprocess {
            let config = pp.config;
            let scene_source = &pp.sample_view;

            // Run bloom on the scene HDR texture.
            if config.bloom_enabled {
                pp.bloom_pass.encode(encoder, &gpu.queue, &pp.bloom_down_pipeline, &pp.bloom_up_pipeline, config.bloom_threshold, config.bloom_soft_knee);
            }

            // SSAO result (or fallback white).
            let ssao_view = if config.ssao_enabled {
                if let Some(ssao) = &self.ssao_pass {
                    ssao.result_view()
                } else {
                    &self.fallback_ssao_view
                }
            } else {
                &self.fallback_ssao_view
            };

            // Upload composite params.
            let params = CompositeParams3D {
                bloom_intensity: if config.bloom_enabled { config.bloom_intensity } else { 0.0 },
                tone_map: if config.tone_map_enabled { 1.0 } else { 0.0 },
                ssao_enabled: if config.ssao_enabled { 1.0 } else { 0.0 },
                _pad0: 0.0,
                fog_color_enabled: [
                    config.fog_color[0],
                    config.fog_color[1],
                    config.fog_color[2],
                    if config.fog_enabled { 1.0 } else { 0.0 },
                ],
                fog_params: [config.fog_start, config.fog_end, camera.near, camera.far],
            };
            gpu.queue.write_buffer(&pp.params_buffer, 0, bytemuck::bytes_of(&params));

            // Build composite bind group.
            let bloom_view = if config.bloom_enabled {
                pp.bloom_pass.result_view()
            } else {
                &pp.bloom_black_view
            };
            let composite_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("esox_3d_composite_bg"),
                layout: &pp.composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(scene_source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(bloom_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(ssao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&pp.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: pp.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&self.depth_sample_view),
                    },
                ],
            });

            // Composite pass -> surface.
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("esox_3d_composite_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
                pass.set_pipeline(&pp.composite_pipeline);
                pass.set_bind_group(0, &composite_bg, &[]);
                pass.draw(0..3, 0..1); // Fullscreen triangle.
            }
        }
    }
}
