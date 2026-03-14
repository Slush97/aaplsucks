use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;

use crate::error::Error;
use crate::primitive::{QuadInstance, ShaderId};

/// Holds the core wgpu state: instance, adapter, device, queue, and surface.
pub struct GpuContext {
    /// The wgpu instance.
    pub instance: wgpu::Instance,
    /// The selected GPU adapter.
    pub adapter: wgpu::Adapter,
    /// The logical GPU device (wrapped in `Arc` for background pipeline compilation).
    pub device: Arc<wgpu::Device>,
    /// The command queue.
    pub queue: wgpu::Queue,
    /// The window surface for presentation.
    pub surface: wgpu::Surface<'static>,
    /// Current surface configuration.
    pub config: wgpu::SurfaceConfiguration,
    /// Display scale factor (HiDPI multiplier).
    pub scale_factor: f64,
    /// MSAA sample count (1 = no MSAA, 4 = 4x MSAA).
    pub sample_count: u32,
    /// Whether HDR output is active (Rgba16Float surface format).
    pub hdr_active: bool,
    /// Depth/stencil texture format used by all scene-pass pipelines.
    pub depth_format: wgpu::TextureFormat,
    /// Whether the GPU supports `MULTI_DRAW_INDIRECT`.
    pub multi_draw_indirect: bool,
}

impl GpuContext {
    /// Initialize GPU state for the given window.
    ///
    /// When `hdr` is `true`, the surface prefers `Rgba16Float` for HDR output.
    /// Falls back to sRGB with a warning if the format is not supported.
    pub async fn new(window: Arc<winit::window::Window>, hdr: bool) -> Result<Self, Error> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| Error::SurfaceConfig(format!("failed to create surface: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| Error::NoAdapter)?;

        let info = adapter.get_info();
        tracing::info!(
            backend = ?info.backend,
            device = %info.name,
            driver = %info.driver,
            "GPU adapter selected"
        );

        // Request optional features if the adapter supports them.
        // INDIRECT_FIRST_INSTANCE is required for multi_draw_indexed_indirect
        // with per-draw instance offsets (first_instance must be 0 without it).
        let desired_features = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::INDIRECT_FIRST_INSTANCE;
        let enabled_features = adapter.features() & desired_features;
        let multi_draw_indirect =
            enabled_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        if multi_draw_indirect {
            tracing::info!("INDIRECT_FIRST_INSTANCE supported — multi-draw-indirect enabled");
        }

        let (raw_device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("esox_gfx device"),
                required_features: enabled_features,
                ..Default::default()
            })
            .await?;
        let device = Arc::new(raw_device);

        let scale_factor = window.scale_factor();
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let srgb_fallback = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .or_else(|| caps.formats.first().copied())
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

        let (format, hdr_active) = if hdr {
            if let Some(&f) = caps
                .formats
                .iter()
                .find(|f| **f == wgpu::TextureFormat::Rgba16Float)
            {
                tracing::info!("HDR enabled: using Rgba16Float surface format");
                (f, true)
            } else {
                tracing::warn!("HDR requested but Rgba16Float not supported; falling back to sRGB");
                (srgb_fallback, false)
            }
        } else {
            (srgb_fallback, false)
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            },
            desired_maximum_frame_latency: 1,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            surface,
            config,
            scale_factor,
            sample_count: 1,
            hdr_active,
            depth_format: wgpu::TextureFormat::Depth24PlusStencil8,
            multi_draw_indirect,
        })
    }

    /// Resize the surface to the new dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Acquire the current surface texture for rendering.
    ///
    /// Returns a [`SurfaceFrame`] that can be passed to both 3D and 2D
    /// render passes before presenting.
    pub fn acquire_surface(&self) -> Result<crate::frame::SurfaceFrame, wgpu::SurfaceError> {
        let texture = self.surface.get_current_texture()?;
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        Ok(crate::frame::SurfaceFrame { texture, view })
    }
}

/// A registered render pipeline handle.
pub struct PipelineHandle {
    /// The compiled render pipeline.
    pub pipeline: wgpu::RenderPipeline,
    /// Human-readable label.
    pub label: String,
}

/// A pipeline that has finished compiling on a background thread.
pub struct ReadyPipeline {
    /// The shader ID this pipeline is registered under.
    pub id: ShaderId,
    /// The compiled pipeline handle.
    pub handle: PipelineHandle,
}

/// Receiving end of the async pipeline compilation channel.
pub type PipelineReceiver = mpsc::Receiver<ReadyPipeline>;

/// Registry of render pipelines, keyed by shader ID.
pub struct PipelineRegistry {
    pipelines: HashMap<ShaderId, PipelineHandle>,
    quad_pipeline: Option<ShaderId>,
    /// Shared bind group layout for all scene-pass pipelines.
    scene_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Shared bind group layout for the post-process pipeline.
    pp_bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl PipelineRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
            quad_pipeline: None,
            scene_bind_group_layout: None,
            pp_bind_group_layout: None,
        }
    }

    /// Get the shared scene-pass bind group layout, if created.
    pub fn scene_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.scene_bind_group_layout.as_ref()
    }

    /// Get the shared post-process bind group layout, if created.
    pub fn pp_bind_group_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        self.pp_bind_group_layout.as_ref()
    }

    /// Create only the scene-pass bind group layout (no shader compilation).
    ///
    /// This is the fast synchronous step needed before `RenderResources::new()`.
    /// Pipeline compilation can then proceed on a background thread via
    /// [`spawn_pipeline_compilation`].
    pub fn create_scene_bind_group_layout(&mut self, gpu: &GpuContext) -> wgpu::BindGroupLayout {
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("quad_bind_group_layout"),
                entries: &bind_group_layout_entries(),
            });
        self.scene_bind_group_layout = Some(layout.clone());
        self.quad_pipeline = Some(crate::primitive::PIPELINE_SDF_2D);
        layout
    }

    /// Create and register the three built-in quad instancing pipelines:
    /// 2D SDF, text, and raymarched 3D.
    pub fn create_quad_pipeline(&mut self, gpu: &GpuContext) -> Result<ShaderId, Error> {
        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("quad_bind_group_layout"),
                    entries: &bind_group_layout_entries(),
                });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quad_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipelines = [
            (
                crate::primitive::PIPELINE_SDF_2D,
                SDF_2D_FRAGMENT_SOURCE,
                "quad_sdf_2d",
            ),
            (
                crate::primitive::PIPELINE_TEXT,
                TEXT_FRAGMENT_SOURCE,
                "quad_text",
            ),
            (
                crate::primitive::PIPELINE_3D,
                RAYMARCHED_3D_FRAGMENT_SOURCE,
                "quad_3d",
            ),
        ];

        for (id, frag_source, label) in pipelines {
            let full_source = format!("{SHADER_PREAMBLE}\n{frag_source}");
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(full_source.into()),
                });

            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[quad_vertex_layout(), instance_vertex_layout()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(depth_stencil_state_read_only()),
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                });

            self.pipelines.insert(
                id,
                PipelineHandle {
                    pipeline,
                    label: label.to_string(),
                },
            );
        }

        // Create the opaque variant of the SDF 2D pipeline (blend: None).
        // Used for cell backgrounds in the OpaqueBackground render phase.
        {
            let full_source = format!("{SHADER_PREAMBLE}\n{SDF_2D_FRAGMENT_SOURCE}");
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("quad_sdf_2d_opaque"),
                    source: wgpu::ShaderSource::Wgsl(full_source.into()),
                });

            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("quad_sdf_2d_opaque"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[quad_vertex_layout(), instance_vertex_layout()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(depth_stencil_state_write()),
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                });

            self.pipelines.insert(
                crate::primitive::PIPELINE_SDF_2D_OPAQUE,
                PipelineHandle {
                    pipeline,
                    label: "quad_sdf_2d_opaque".to_string(),
                },
            );
        }

        // Create blend variant pipelines for SDF 2D (additive, screen, multiply).
        let blend_variants: [(crate::primitive::ShaderId, &str, wgpu::BlendState); 3] = [
            (
                crate::primitive::PIPELINE_SDF_2D_ADDITIVE,
                "quad_sdf_2d_additive",
                wgpu::BlendState {
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
                },
            ),
            (
                crate::primitive::PIPELINE_SDF_2D_SCREEN,
                "quad_sdf_2d_screen",
                wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrc,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
            ),
            (
                crate::primitive::PIPELINE_SDF_2D_MULTIPLY,
                "quad_sdf_2d_multiply",
                wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Dst,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::DstAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
            ),
        ];

        for (id, label, blend) in blend_variants {
            let full_source = format!("{SHADER_PREAMBLE}\n{SDF_2D_FRAGMENT_SOURCE}");
            let shader = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(full_source.into()),
                });

            let pipeline = gpu
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[quad_vertex_layout(), instance_vertex_layout()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(depth_stencil_state_read_only()),
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                });

            self.pipelines.insert(
                id,
                PipelineHandle {
                    pipeline,
                    label: label.to_string(),
                },
            );
        }

        self.scene_bind_group_layout = Some(bind_group_layout);
        self.quad_pipeline = Some(crate::primitive::PIPELINE_SDF_2D);
        tracing::info!(
            "created quad pipelines (sdf_2d, text, 3d, sdf_2d_opaque, \
             sdf_2d_additive, sdf_2d_screen, sdf_2d_multiply)"
        );
        Ok(crate::primitive::PIPELINE_SDF_2D)
    }

    /// Register a user-supplied WGSL fragment shader as a new pipeline.
    ///
    /// The `wgsl_fragment` string should contain a function body that receives
    /// `in: VertexOutput` and returns `vec4<f32>`. The system wraps it as:
    ///
    /// ```wgsl
    /// @fragment
    /// fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    ///     // clip test (auto-inserted)
    ///     <your code>
    /// }
    /// ```
    ///
    /// Custom shaders can read 16 user floats packed into `QuadInstance` fields:
    /// - `in.sdf_params` → `ShaderParams::values[0..4]`
    /// - `in.extra` → `ShaderParams::values[4..8]`
    /// - `in.border_radius` → `ShaderParams::values[8..12]`
    /// - `in.color2` → `ShaderParams::values[12..16]`
    ///
    /// IDs 0–9 are reserved for built-in pipelines. Use
    /// [`USER_SHADER_ID_MIN`](crate::primitive::USER_SHADER_ID_MIN) (10) or
    /// higher for user shaders.
    pub fn register_shader_pipeline(
        &mut self,
        gpu: &GpuContext,
        id: ShaderId,
        wgsl_fragment: &str,
        label: &str,
    ) -> Result<(), Error> {
        if self.pipelines.contains_key(&id) {
            return Err(Error::ShaderIdAlreadyRegistered(id.0));
        }

        if id.0 < crate::primitive::USER_SHADER_ID_MIN {
            tracing::warn!(
                id = id.0,
                "registering shader with reserved ID (<{})",
                crate::primitive::USER_SHADER_ID_MIN
            );
        }

        let full_source = compose_scene_shader(wgsl_fragment);

        // Pre-validate with naga before touching the GPU.
        validate_scene_shader(wgsl_fragment).map_err(Error::ShaderValidation)?;

        self.create_scene_shader_pipeline(gpu, id, &full_source, label)
    }

    /// Replace an existing user shader pipeline with new WGSL source.
    ///
    /// On validation failure the old pipeline remains untouched and an error is
    /// returned. Returns [`Error::PipelineNotFound`] if `id` has not been
    /// registered.
    pub fn reload_shader_pipeline(
        &mut self,
        gpu: &GpuContext,
        id: ShaderId,
        wgsl_fragment: &str,
    ) -> Result<(), Error> {
        let existing = self
            .pipelines
            .get(&id)
            .ok_or_else(|| Error::PipelineNotFound(format!("shader ID {}", id.0)))?;
        let old_label = existing.label.clone();

        // Validate first — failure leaves the old pipeline in place.
        validate_scene_shader(wgsl_fragment).map_err(Error::ShaderValidation)?;

        let full_source = compose_scene_shader(wgsl_fragment);
        // create_scene_shader_pipeline overwrites the HashMap entry.
        self.create_scene_shader_pipeline(gpu, id, &full_source, &old_label)?;
        tracing::info!(id = id.0, "reloaded custom shader pipeline");
        Ok(())
    }

    /// Remove a user shader pipeline from the registry.
    ///
    /// Returns `true` if the pipeline existed and was removed, `false` if it
    /// was not found.
    ///
    /// # Panics
    ///
    /// Panics if `id` is a built-in pipeline (0, 1, 2, 3, or 100). Removing
    /// built-in pipelines is a programmer error.
    pub fn unregister_shader_pipeline(&mut self, id: ShaderId) -> bool {
        assert!(
            !is_builtin_id(id),
            "cannot unregister built-in pipeline ID {}",
            id.0
        );
        self.pipelines.remove(&id).is_some()
    }

    /// Internal helper: create a scene-pass render pipeline and insert it into
    /// the registry (overwrites any existing entry for `id`).
    fn create_scene_shader_pipeline(
        &mut self,
        gpu: &GpuContext,
        id: ShaderId,
        full_source: &str,
        label: &str,
    ) -> Result<(), Error> {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(full_source.into()),
            });

        let bind_group_layout = self
            .scene_bind_group_layout
            .as_ref()
            .expect("scene layout must exist before registering shaders")
            .clone();

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label}_pipeline_layout")),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{label}_pipeline")),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[quad_vertex_layout(), instance_vertex_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    ..Default::default()
                },
                depth_stencil: Some(depth_stencil_state_read_only()),
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        self.pipelines.insert(
            id,
            PipelineHandle {
                pipeline,
                label: label.to_string(),
            },
        );

        tracing::info!(id = id.0, label, "registered custom shader pipeline");
        Ok(())
    }

    /// Create and register the post-process pipeline.
    ///
    /// Uses a fullscreen triangle (no vertex buffer) and the post-process
    /// bind group layout (uniforms + 2D scene texture + sampler).
    ///
    /// If `user_fragment_source` is `Some`, the user's WGSL fragment body is
    /// composed into a complete shader module via [`crate::offscreen::compose_user_shader`].
    /// On compile failure, a warning is logged and the built-in shader is used.
    pub fn create_post_process_pipeline(
        &mut self,
        gpu: &GpuContext,
        pp_layout: &wgpu::BindGroupLayout,
        user_fragment_source: Option<&str>,
    ) -> Result<ShaderId, Error> {
        let id = crate::offscreen::PIPELINE_POST_PROCESS;

        let (full_source, is_user) = if let Some(user_src) = user_fragment_source {
            tracing::info!("composing user post-process shader");
            (crate::offscreen::compose_user_shader(user_src), true)
        } else {
            (
                format!(
                    "{}\n{}",
                    crate::offscreen::POST_PROCESS_VERTEX_SOURCE,
                    crate::offscreen::POST_PROCESS_FRAGMENT,
                ),
                false,
            )
        };

        // Try to create the pipeline. If using a user shader and it fails
        // (wgpu validation error), fall back to the built-in shader.
        let result = self.try_create_pp_pipeline(gpu, pp_layout, &full_source, id);
        if result.is_err() && is_user {
            tracing::warn!("user post-process shader failed to compile; falling back to built-in");
            let builtin = format!(
                "{}\n{}",
                crate::offscreen::POST_PROCESS_VERTEX_SOURCE,
                crate::offscreen::POST_PROCESS_FRAGMENT,
            );
            return self.try_create_pp_pipeline(gpu, pp_layout, &builtin, id);
        }
        result
    }

    /// Attempt to create and register a post-process pipeline from WGSL source.
    fn try_create_pp_pipeline(
        &mut self,
        gpu: &GpuContext,
        pp_layout: &wgpu::BindGroupLayout,
        full_source: &str,
        id: ShaderId,
    ) -> Result<ShaderId, Error> {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("post_process_shader"),
                source: wgpu::ShaderSource::Wgsl(full_source.into()),
            });

        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("post_process_pipeline_layout"),
                bind_group_layouts: &[pp_layout],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("post_process_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[], // No vertex buffers — fullscreen triangle from vertex_index.
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None, // Post-process writes final color directly.
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        self.pipelines.insert(
            id,
            PipelineHandle {
                pipeline,
                label: "post_process".to_string(),
            },
        );

        self.pp_bind_group_layout = Some(pp_layout.clone());
        tracing::info!("created post-process pipeline");
        Ok(id)
    }

    /// Look up a pipeline by its shader ID.
    pub fn get(&self, id: ShaderId) -> Option<&PipelineHandle> {
        self.pipelines.get(&id)
    }

    /// Get the built-in quad pipeline ID, if created.
    pub fn quad_pipeline_id(&self) -> Option<ShaderId> {
        self.quad_pipeline
    }

    /// Drain all ready pipelines from the channel and insert them.
    ///
    /// Returns the number of pipelines inserted this call.
    pub fn poll_ready_pipelines(&mut self, rx: &PipelineReceiver) -> usize {
        let mut count = 0;
        while let Ok(ready) = rx.try_recv() {
            tracing::info!(id = ready.id.0, label = %ready.handle.label, "pipeline ready");
            self.pipelines.insert(ready.id, ready.handle);
            count += 1;
        }
        count
    }
}

/// Configuration for spawning background pipeline compilation.
pub struct PipelineCompileConfig {
    /// The GPU device (shared via `Arc` for cross-thread use).
    pub device: Arc<wgpu::Device>,
    /// Surface texture format.
    pub format: wgpu::TextureFormat,
    /// MSAA sample count.
    pub sample_count: u32,
    /// Scene-pass bind group layout.
    pub scene_bind_group_layout: wgpu::BindGroupLayout,
    /// Post-process bind group layout (if post-process is enabled).
    pub pp_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// User-supplied post-process shader source (if any).
    pub user_shader_source: Option<String>,
    /// Bloom bind group layout (if bloom is enabled).
    pub bloom_bind_group_layout: Option<wgpu::BindGroupLayout>,
}

/// Spawn background pipeline compilation for all built-in scene pipelines
/// and optionally the post-process pipeline.
///
/// Returns a [`PipelineReceiver`] that yields [`ReadyPipeline`] values as
/// each pipeline finishes compiling. The caller should store the receiver and
/// call [`PipelineRegistry::poll_ready_pipelines`] each frame.
///
/// The `wake` closure is called after each pipeline to wake the main thread.
pub fn spawn_pipeline_compilation(
    config: PipelineCompileConfig,
    wake: impl Fn() + Send + 'static,
) -> PipelineReceiver {
    let (tx, rx) = mpsc::channel();

    let PipelineCompileConfig {
        device,
        format,
        sample_count,
        scene_bind_group_layout,
        pp_bind_group_layout,
        user_shader_source,
        bloom_bind_group_layout,
    } = config;

    std::thread::Builder::new()
        .name("pipeline-compile".into())
        .spawn(move || {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quad_pipeline_layout"),
                bind_group_layouts: &[&scene_bind_group_layout],
                immediate_size: 0,
            });

            // Helper: compile one scene-pass pipeline.
            let compile_scene = |frag_source: &str,
                                 label: &str,
                                 blend: Option<wgpu::BlendState>,
                                 depth_write: bool,
                                 sc: u32,
                                 use_depth: bool|
             -> wgpu::RenderPipeline {
                let full_source = format!("{SHADER_PREAMBLE}\n{frag_source}");
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(full_source.into()),
                });

                let ds = if use_depth {
                    Some(if depth_write {
                        depth_stencil_state_write()
                    } else {
                        depth_stencil_state_read_only()
                    })
                } else {
                    None
                };

                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[quad_vertex_layout(), instance_vertex_layout()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        strip_index_format: None,
                        ..Default::default()
                    },
                    depth_stencil: ds,
                    multisample: wgpu::MultisampleState {
                        count: sc,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                })
            };

            let send = |id: ShaderId, label: &str, pipeline: wgpu::RenderPipeline| {
                let _ = tx.send(ReadyPipeline {
                    id,
                    handle: PipelineHandle {
                        pipeline,
                        label: label.to_string(),
                    },
                });
                wake();
            };

            // 3 alpha-blended scene pipelines.
            let alpha_blend = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
            for (id, frag, label) in [
                (
                    crate::primitive::PIPELINE_SDF_2D,
                    SDF_2D_FRAGMENT_SOURCE,
                    "quad_sdf_2d",
                ),
                (
                    crate::primitive::PIPELINE_TEXT,
                    TEXT_FRAGMENT_SOURCE,
                    "quad_text",
                ),
                (
                    crate::primitive::PIPELINE_3D,
                    RAYMARCHED_3D_FRAGMENT_SOURCE,
                    "quad_3d",
                ),
            ] {
                let p = compile_scene(frag, label, alpha_blend, false, sample_count, true);
                send(id, label, p);
            }

            // Opaque SDF 2D (no blend, depth write).
            {
                let label = "quad_sdf_2d_opaque";
                let p = compile_scene(SDF_2D_FRAGMENT_SOURCE, label, None, true, sample_count, true);
                send(crate::primitive::PIPELINE_SDF_2D_OPAQUE, label, p);
            }

            // Blend variant pipelines.
            let blend_variants: [(ShaderId, &str, wgpu::BlendState); 3] = [
                (
                    crate::primitive::PIPELINE_SDF_2D_ADDITIVE,
                    "quad_sdf_2d_additive",
                    wgpu::BlendState {
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
                    },
                ),
                (
                    crate::primitive::PIPELINE_SDF_2D_SCREEN,
                    "quad_sdf_2d_screen",
                    wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrc,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    },
                ),
                (
                    crate::primitive::PIPELINE_SDF_2D_MULTIPLY,
                    "quad_sdf_2d_multiply",
                    wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Dst,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::DstAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    },
                ),
            ];
            for (id, label, blend) in &blend_variants {
                let p = compile_scene(SDF_2D_FRAGMENT_SOURCE, label, Some(blend.clone()), false, sample_count, true);
                send(*id, label, p);
            }

            // Non-MSAA variants (sample_count=1, no depth) for compositing 2D UI
            // on top of a 3D pre-render pass, where the 2D render pass runs
            // without MSAA and without a depth attachment.
            if sample_count > 1 {
                use crate::primitive::NO_MSAA_PIPELINE_OFFSET;

                for (id, frag, label) in [
                    (
                        crate::primitive::PIPELINE_SDF_2D,
                        SDF_2D_FRAGMENT_SOURCE,
                        "quad_sdf_2d_no_msaa",
                    ),
                    (
                        crate::primitive::PIPELINE_TEXT,
                        TEXT_FRAGMENT_SOURCE,
                        "quad_text_no_msaa",
                    ),
                    (
                        crate::primitive::PIPELINE_3D,
                        RAYMARCHED_3D_FRAGMENT_SOURCE,
                        "quad_3d_no_msaa",
                    ),
                ] {
                    let p = compile_scene(frag, label, alpha_blend, false, 1, false);
                    send(ShaderId(id.0 + NO_MSAA_PIPELINE_OFFSET), label, p);
                }

                {
                    let label = "quad_sdf_2d_opaque_no_msaa";
                    let p = compile_scene(SDF_2D_FRAGMENT_SOURCE, label, None, false, 1, false);
                    send(
                        ShaderId(crate::primitive::PIPELINE_SDF_2D_OPAQUE.0 + NO_MSAA_PIPELINE_OFFSET),
                        label,
                        p,
                    );
                }

                for (id, label, blend) in &blend_variants {
                    let no_msaa_label = format!("{label}_no_msaa");
                    let p = compile_scene(SDF_2D_FRAGMENT_SOURCE, &no_msaa_label, Some(blend.clone()), false, 1, false);
                    send(ShaderId(id.0 + NO_MSAA_PIPELINE_OFFSET), &no_msaa_label, p);
                }
            }

            // Post-process pipeline (if layout provided).
            if let Some(pp_layout) = pp_bind_group_layout {
                let (full_source, is_user) = if let Some(ref user_src) = user_shader_source {
                    (crate::offscreen::compose_user_shader(user_src), true)
                } else {
                    (
                        format!(
                            "{}\n{}",
                            crate::offscreen::POST_PROCESS_VERTEX_SOURCE,
                            crate::offscreen::POST_PROCESS_FRAGMENT,
                        ),
                        false,
                    )
                };

                let try_pp = |source: &str| -> Option<wgpu::RenderPipeline> {
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("post_process_shader"),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                    let pp_pipeline_layout =
                        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                            label: Some("post_process_pipeline_layout"),
                            bind_group_layouts: &[&pp_layout],
                            immediate_size: 0,
                        });
                    Some(
                        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some("post_process_pipeline"),
                            layout: Some(&pp_pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some("vs_main"),
                                buffers: &[],
                                compilation_options: wgpu::PipelineCompilationOptions::default(),
                            },
                            fragment: Some(wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some("fs_main"),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format,
                                    blend: None,
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                                compilation_options: wgpu::PipelineCompilationOptions::default(),
                            }),
                            primitive: wgpu::PrimitiveState {
                                topology: wgpu::PrimitiveTopology::TriangleList,
                                ..Default::default()
                            },
                            depth_stencil: None,
                            multisample: wgpu::MultisampleState::default(),
                            multiview_mask: None,
                            cache: None,
                        }),
                    )
                };

                let pipeline = try_pp(&full_source).or_else(|| {
                    if is_user {
                        tracing::warn!("user post-process shader failed; falling back to built-in");
                        let builtin = format!(
                            "{}\n{}",
                            crate::offscreen::POST_PROCESS_VERTEX_SOURCE,
                            crate::offscreen::POST_PROCESS_FRAGMENT,
                        );
                        try_pp(&builtin)
                    } else {
                        None
                    }
                });

                if let Some(p) = pipeline {
                    let id = crate::offscreen::PIPELINE_POST_PROCESS;
                    send(id, "post_process", p);
                }
            }

            // Bloom pipelines (if layout provided).
            if let Some(bloom_layout) = bloom_bind_group_layout {
                let bloom_pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("bloom_pipeline_layout"),
                        bind_group_layouts: &[&bloom_layout],
                        immediate_size: 0,
                    });

                let compile_bloom = |source: &str,
                                     label: &str,
                                     blend: Option<wgpu::BlendState>|
                 -> wgpu::RenderPipeline {
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(label),
                        layout: Some(&bloom_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("vs_main"),
                            buffers: &[],
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some("fs_main"),
                            targets: &[Some(wgpu::ColorTargetState {
                                format,
                                blend,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                            compilation_options: wgpu::PipelineCompilationOptions::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleList,
                            ..Default::default()
                        },
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    })
                };

                // Downsample pipeline (no blend — overwrites target).
                let down_src = crate::bloom::downsample_shader_source();
                let p = compile_bloom(&down_src, "bloom_downsample", None);
                send(
                    crate::bloom::PIPELINE_BLOOM_DOWNSAMPLE,
                    "bloom_downsample",
                    p,
                );

                // Upsample pipeline (additive blend — accumulates onto destination).
                let additive = Some(wgpu::BlendState {
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
                });
                let up_src = crate::bloom::upsample_shader_source();
                let p = compile_bloom(&up_src, "bloom_upsample", additive);
                send(crate::bloom::PIPELINE_BLOOM_UPSAMPLE, "bloom_upsample", p);
            }

            tracing::info!("background pipeline compilation finished");
        })
        .expect("failed to spawn pipeline-compile thread");

    rx
}

impl Default for PipelineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Depth/stencil state for opaque geometry: writes depth, tests LessEqual.
fn depth_stencil_state_write() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::LessEqual,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Depth/stencil state for blended geometry: reads depth (Always pass), no write.
fn depth_stencil_state_read_only() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Compose a complete WGSL shader module from a user-supplied fragment body.
///
/// The result includes the shared preamble (vertex shader, uniforms, bindings)
/// and wraps the user body in an `fs_main` function. Clip testing is handled by
/// the GPU scissor rect (`set_scissor_rect`) set per-batch from `ClipKey`.
fn compose_scene_shader(wgsl_fragment: &str) -> String {
    format!(
        "{SHADER_PREAMBLE}\n\
         @fragment\n\
         fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{\n\
             {wgsl_fragment}\n\
         }}\n"
    )
}

/// Pre-validate a user scene-pass shader body using naga's WGSL front-end and
/// IR validator.
///
/// Returns `Ok(())` if the composed shader parses and validates, or an error
/// description on failure. This catches syntax and type errors before GPU
/// pipeline creation.
pub fn validate_scene_shader(wgsl_fragment: &str) -> Result<(), String> {
    let full_source = compose_scene_shader(wgsl_fragment);
    let module =
        naga::front::wgsl::parse_str(&full_source).map_err(|e| format!("WGSL parse error: {e}"))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error: {e}"))?;
    Ok(())
}

/// Built-in pipeline IDs that must not be unregistered.
const BUILTIN_IDS: [u32; 10] = [
    crate::primitive::PIPELINE_SDF_2D.0,
    crate::primitive::PIPELINE_TEXT.0,
    crate::primitive::PIPELINE_3D.0,
    crate::primitive::PIPELINE_SDF_2D_OPAQUE.0,
    crate::primitive::PIPELINE_SDF_2D_ADDITIVE.0,
    crate::primitive::PIPELINE_SDF_2D_SCREEN.0,
    crate::primitive::PIPELINE_SDF_2D_MULTIPLY.0,
    crate::offscreen::PIPELINE_POST_PROCESS.0,
    crate::bloom::PIPELINE_BLOOM_DOWNSAMPLE.0,
    crate::bloom::PIPELINE_BLOOM_UPSAMPLE.0,
];

/// Check whether a shader ID is a built-in pipeline.
fn is_builtin_id(id: ShaderId) -> bool {
    BUILTIN_IDS.contains(&id.0)
}

/// Shared bind group layout entries used by both the built-in and custom pipelines.
fn bind_group_layout_entries() -> [wgpu::BindGroupLayoutEntry; 5] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        },
    ]
}

/// Vertex buffer layout for the 4-vertex unit quad (buffer slot 0).
fn quad_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
        }],
    }
}

/// Instance vertex attributes (9 vec4 fields, locations 1–9).
const INSTANCE_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 9] = [
    wgpu::VertexAttribute {
        offset: 0,
        shader_location: 1,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 16,
        shader_location: 2,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 32,
        shader_location: 3,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 48,
        shader_location: 4,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 64,
        shader_location: 5,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 80,
        shader_location: 6,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 96,
        shader_location: 7,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 112,
        shader_location: 8,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        offset: 128,
        shader_location: 9,
        format: wgpu::VertexFormat::Float32x4,
    },
];

/// Vertex buffer layout for QuadInstance (buffer slot 1, per-instance).
fn instance_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<QuadInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &INSTANCE_VERTEX_ATTRIBUTES,
    }
}

/// GPU resources for the instanced quad pipeline.
pub struct RenderResources {
    /// Bind group layout for recreating bind groups.
    pub bind_group_layout: wgpu::BindGroupLayout,
    /// 4-vertex unit quad (static, created once).
    pub quad_vertex_buffer: wgpu::Buffer,
    /// Double-buffered instance buffers (ping-pong).
    instance_buffers: [wgpu::Buffer; 2],
    /// Capacity of each instance buffer (in instances).
    instance_capacities: [u32; 2],
    /// Index of the currently active instance buffer (0 or 1).
    current_buffer_index: usize,
    /// Uniform buffer (32 bytes for FrameUniforms).
    pub uniform_buffer: wgpu::Buffer,
    /// Bind group for the pipeline.
    pub bind_group: wgpu::BindGroup,
    /// 1x1 white placeholder texture for non-textured shapes.
    pub placeholder_texture: wgpu::Texture,
    /// Texture view for the placeholder.
    placeholder_view: wgpu::TextureView,
    /// Nearest-neighbor sampler for pixel-perfect rendering.
    sampler: wgpu::Sampler,
    /// Linear sampler for smooth texture sampling.
    linear_sampler: wgpu::Sampler,
    /// Consecutive frames where instance count < capacity/4 (for shrink hysteresis).
    pub underutilization_frames: u32,
    /// Monotonic counter incremented when bind group inputs change.
    bind_group_generation: u64,
    /// Generation at which the current bind group was created.
    bind_group_built_at: u64,
}

impl RenderResources {
    /// Create render resources after the quad pipelines have been registered.
    pub fn new(gpu: &GpuContext, registry: &PipelineRegistry) -> Result<Self, Error> {
        let bind_group_layout = registry
            .scene_bind_group_layout()
            .ok_or_else(|| Error::SurfaceConfig("scene bind group layout must exist".into()))?
            .clone();

        // 4-vertex unit quad: triangle strip (0,0), (1,0), (0,1), (1,1).
        let quad_vertices: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let quad_vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quad_vertex_buffer"),
            size: size_of_val(&quad_vertices) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&quad_vertex_buffer, 0, bytemuck::cast_slice(&quad_vertices));

        // Double-buffered instance buffers (start with capacity for 256 instances each).
        let initial_capacity: u32 = 256;
        let create_instance_buf = |label: &str| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (initial_capacity as usize * size_of::<QuadInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let instance_buffers = [
            create_instance_buf("quad_instance_buffer_0"),
            create_instance_buf("quad_instance_buffer_1"),
        ];

        // Uniform buffer (32 bytes).
        let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_uniform_buffer"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 1x1 white placeholder texture.
        let placeholder_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("placeholder_texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &placeholder_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let placeholder_view = placeholder_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("quad_sampler_nearest"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let linear_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("quad_sampler_linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quad_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&placeholder_view),
                },
            ],
        });

        tracing::info!("created render resources (instance_capacity={initial_capacity})");

        Ok(Self {
            bind_group_layout,
            quad_vertex_buffer,
            instance_buffers,
            instance_capacities: [initial_capacity; 2],
            current_buffer_index: 0,
            uniform_buffer,
            bind_group,
            placeholder_texture,
            placeholder_view,
            sampler,
            linear_sampler,
            underutilization_frames: 0,
            bind_group_generation: 0,
            bind_group_built_at: 0,
        })
    }

    /// Whether the bind group needs rebuilding (inputs changed since last build).
    pub fn bind_group_stale(&self) -> bool {
        self.bind_group_generation != self.bind_group_built_at
    }

    /// Get the currently active instance buffer.
    pub fn current_instance_buffer(&self) -> &wgpu::Buffer {
        &self.instance_buffers[self.current_buffer_index]
    }

    /// Get the capacity of the currently active instance buffer.
    pub fn current_instance_capacity(&self) -> u32 {
        self.instance_capacities[self.current_buffer_index]
    }

    /// Flip to the other instance buffer (ping-pong).
    pub fn flip_instance_buffer(&mut self) {
        self.current_buffer_index = 1 - self.current_buffer_index;
    }

    /// Resize the current instance buffer to the given capacity.
    pub fn resize_current_buffer(&mut self, device: &wgpu::Device, new_cap: u32) {
        let idx = self.current_buffer_index;
        self.instance_buffers[idx] = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(if idx == 0 {
                "quad_instance_buffer_0"
            } else {
                "quad_instance_buffer_1"
            }),
            size: (new_cap as usize * size_of::<QuadInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_capacities[idx] = new_cap;
    }

    /// Rebuild the bind group (e.g., after texture changes).
    pub fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group_generation += 1;
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quad_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.placeholder_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.placeholder_view),
                },
            ],
        });
        self.bind_group_built_at = self.bind_group_generation;
    }

    /// Rebind both the glyph atlas and image atlas textures.
    pub fn bind_textures(
        &mut self,
        device: &wgpu::Device,
        glyph_view: &wgpu::TextureView,
        image_view: &wgpu::TextureView,
    ) {
        self.bind_group_generation += 1;
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quad_bind_group_atlas"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(glyph_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(image_view),
                },
            ],
        });
        self.bind_group_built_at = self.bind_group_generation;
    }
}

/// Shared WGSL preamble: uniforms, bindings, I/O structs, constants, vertex shader.
///
/// Used by both the built-in quad pipeline and custom shader pipelines.
pub const SHADER_PREAMBLE: &str = r"
// ── Uniforms ──

struct FrameUniforms {
    viewport: vec4<f32>,  // width, height, 1/width, 1/height
    time: vec4<f32>,      // time_secs, delta_time, frame_number, 0
}

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var atlas_texture: texture_2d_array<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(4) var image_texture: texture_2d_array<f32>;

// ── Vertex I/O ──

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
    @location(1) inst_rect: vec4<f32>,
    @location(2) inst_uv: vec4<f32>,
    @location(3) inst_color: vec4<f32>,
    @location(4) inst_border_radius: vec4<f32>,
    @location(5) inst_sdf_params: vec4<f32>,
    @location(6) inst_flags: vec4<f32>,
    @location(7) inst_clip_rect: vec4<f32>,
    @location(8) inst_color2: vec4<f32>,
    @location(9) inst_extra: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_pos: vec2<f32>,
    @location(2) rect_size: vec2<f32>,
    @location(3) border_radius: vec4<f32>,
    @location(4) sdf_params: vec4<f32>,
    @location(5) flags: vec4<f32>,
    @location(6) uv: vec2<f32>,
    @location(7) rect_origin: vec2<f32>,
    @location(8) clip_rect: vec4<f32>,
    @location(9) color2: vec4<f32>,
    @location(10) extra: vec4<f32>,
}

// ── Shape type constants ──

const SHAPE_RECT: f32 = 0.0;
const SHAPE_CIRCLE: f32 = 1.0;
const SHAPE_ELLIPSE: f32 = 2.0;
const SHAPE_RING: f32 = 3.0;
const SHAPE_LINE: f32 = 4.0;
const SHAPE_ARC: f32 = 5.0;
const SHAPE_TRIANGLE: f32 = 6.0;
const SHAPE_TEXTURED: f32 = 7.0;
const SHAPE_SPHERE_3D: f32 = 9.0;
const SHAPE_TORUS_3D: f32 = 10.0;
const SHAPE_ROUNDED_BOX_3D: f32 = 11.0;
const SHAPE_POLYGON: f32 = 12.0;
const SHAPE_STAR: f32 = 13.0;
const SHAPE_SECTOR: f32 = 14.0;
const SHAPE_CAPSULE: f32 = 15.0;
const SHAPE_CROSS: f32 = 16.0;
const SHAPE_BEZIER: f32 = 17.0;
const SHAPE_ARBITRARY_TRIANGLE: f32 = 18.0;
const SHAPE_TRAPEZOID: f32 = 19.0;
const SHAPE_SLICED_TORUS_3D: f32 = 20.0;
const SHAPE_MORPH_3D: f32 = 21.0;
const SHAPE_UNDERLINE_CURLY: f32 = 22.0;
const SHAPE_UNDERLINE_DOTTED: f32 = 23.0;
const SHAPE_UNDERLINE_DASHED: f32 = 24.0;
const SHAPE_HEART: f32 = 25.0;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// ── Vertex shader ──

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let pixel_x = in.inst_rect.x + in.quad_pos.x * in.inst_rect.z;
    let pixel_y = in.inst_rect.y + in.quad_pos.y * in.inst_rect.w;

    // Pixel coords to NDC: x in [-1,1], y flipped (pixel y-down to NDC y-up).
    let ndc_x = pixel_x * uniforms.viewport.z * 2.0 - 1.0;
    let ndc_y = 1.0 - pixel_y * uniforms.viewport.w * 2.0;

    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.color = in.inst_color;
    out.local_pos = in.quad_pos;
    out.rect_size = in.inst_rect.zw;
    out.border_radius = in.inst_border_radius;
    out.sdf_params = in.inst_sdf_params;
    out.flags = in.inst_flags;
    out.uv = mix(in.inst_uv.xy, in.inst_uv.zw, in.quad_pos);
    out.rect_origin = in.inst_rect.xy;
    out.clip_rect = in.inst_clip_rect;
    out.color2 = in.inst_color2;
    out.extra = in.inst_extra;

    return out;
}
";

/// Fragment shader for textured/glyph quads (pipeline 1).
///
/// Minimal shader: texture sample + premultiply alpha. Handles 95%+ of terminal
/// pixels with no SDF or raymarching overhead. Clip testing is handled by the
/// GPU scissor rect set per-batch from `ClipKey`.
const TEXT_FRAGMENT_SOURCE: &str = r"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let atlas_layer = u32(in.flags.w);
    let opacity = in.flags.z;

    // Image texture: flags.y == 2.0 → sample from dedicated image atlas.
    if in.flags.y > 1.5 {
        let img_color = textureSample(image_texture, linear_sampler, in.uv, atlas_layer);
        let ia = img_color.a * opacity;
        if ia < 0.001 { discard; }
        return vec4<f32>(img_color.rgb * ia, ia);
    }

    let tex_color = textureSample(atlas_texture, linear_sampler, in.uv, atlas_layer);
    // R8 atlas: coverage is in the red channel.
    let a = tex_color.r;
    if a < 0.001 { discard; }
    // Color emoji: use texture RGBA directly, modulate by opacity only.
    let is_color = in.flags.y > 0.5;
    if is_color {
        return vec4<f32>(tex_color.rgb * a * opacity, a * opacity);
    }
    // Monochrome: premultiply fg color by glyph coverage and node opacity.
    let fa = in.color.a * a * opacity;
    return vec4<f32>(in.color.rgb * fa, fa);
}
";

/// Fragment shader for 2D SDF shapes (pipeline 0).
///
/// Handles all 2D SDF shapes with shadow, gradient, stroke, and anti-aliasing.
const SDF_2D_FRAGMENT_SOURCE: &str = r"
// ── SDF functions (2D) ──

fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, radii: vec4<f32>) -> f32 {
    var r: f32;
    if p.x < 0.0 {
        if p.y < 0.0 { r = radii.x; } else { r = radii.z; }
    } else {
        if p.y < 0.0 { r = radii.y; } else { r = radii.w; }
    }
    r = min(r, min(half_size.x, half_size.y));
    let q = abs(p) - half_size + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

fn sdf_circle(p: vec2<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_ellipse(p: vec2<f32>, ab: vec2<f32>) -> f32 {
    // Approximate ellipse SDF via normalized distance.
    let pn = p / ab;
    let k = length(pn);
    if k < 0.001 { return -min(ab.x, ab.y); }
    return (k - 1.0) * min(ab.x, ab.y);
}

fn sdf_ring(p: vec2<f32>, outer_r: f32, inner_r: f32) -> f32 {
    let d = length(p);
    return max(d - outer_r, inner_r - d);
}

fn sdf_line(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, thickness: f32) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - thickness * 0.5;
}

fn sdf_arc(p: vec2<f32>, radius: f32, thickness: f32, angle_start: f32, angle_sweep: f32) -> f32 {
    // Rotate point so the arc is centered on angle 0.
    let mid_angle = angle_start + angle_sweep * 0.5;
    let ca = cos(-mid_angle);
    let sa = sin(-mid_angle);
    let rp = vec2<f32>(p.x * ca - p.y * sa, p.x * sa + p.y * ca);

    // Angular half-extent.
    let half_sweep = abs(angle_sweep) * 0.5;

    // Radial distance from the arc centerline.
    let d_radial = abs(length(rp) - radius) - thickness * 0.5;

    // Angular test: is the point within the sweep?
    let angle = atan2(rp.y, rp.x);
    if abs(angle) <= half_sweep {
        return d_radial;
    }

    // Outside the sweep — return distance to the nearest endpoint.
    let ep1 = vec2<f32>(cos(half_sweep), sin(half_sweep)) * radius;
    let ep2 = vec2<f32>(cos(half_sweep), -sin(half_sweep)) * radius;
    let d1 = length(rp - ep1) - thickness * 0.5;
    let d2 = length(rp - ep2) - thickness * 0.5;
    return min(d1, d2);
}

fn sdf_triangle(p: vec2<f32>, half_size: vec2<f32>) -> f32 {
    let scale = min(half_size.x, half_size.y);
    if scale < 0.001 { return 0.0; }
    let np = p / scale;
    let k = sqrt(3.0);
    var q = vec2<f32>(abs(np.x) - 1.0, np.y + 1.0 / k);
    if q.x + k * q.y > 0.0 {
        q = vec2<f32>(q.x - k * q.y, -k * q.x - q.y) / 2.0;
    }
    q.x -= clamp(q.x, -2.0, 0.0);
    return -length(q) * sign(q.y) * scale;
}

fn sdf_polygon(p: vec2<f32>, radius: f32, sides: f32) -> f32 {
    // Regular N-gon SDF (Inigo Quilez).
    let n = max(sides, 3.0);
    let an = PI / n;
    let acs = vec2<f32>(cos(an), sin(an));
    // Reduce to first sector.
    var bn = (atan2(p.x, p.y) % (2.0 * an)) - an;
    if bn < 0.0 { bn = -bn; }
    let q = length(p) * vec2<f32>(cos(bn), sin(bn));
    let d = q.x * acs.y - q.y * acs.x;
    return d - radius * acs.y;
}

fn sdf_star(p: vec2<f32>, points: f32, inner_r: f32, outer_r: f32) -> f32 {
    // N-pointed star SDF via sector folding and edge distance.
    let n = max(points, 3.0);
    let an = PI / n;
    // Fold into first sector.
    let angle = atan2(p.y, p.x);
    var bn = (angle % (2.0 * an)) - an;
    if bn < 0.0 { bn = -bn; }
    let q = length(p) * vec2<f32>(cos(bn), sin(bn));
    // Tip vertex at sector center, notch vertex at sector edge.
    let tip = vec2<f32>(outer_r, 0.0);
    let notch = vec2<f32>(inner_r * cos(an), inner_r * sin(an));
    // Distance to edge between tip and notch.
    let edge = notch - tip;
    let qp = q - tip;
    let t = clamp(dot(qp, edge) / dot(edge, edge), 0.0, 1.0);
    let closest = tip + edge * t;
    let d = length(q - closest);
    // Sign: inside if on the correct side of the edge.
    let s = sign(edge.x * qp.y - edge.y * qp.x);
    return d * s;
}

fn sdf_sector(p: vec2<f32>, radius: f32, angle_start: f32, angle_sweep: f32) -> f32 {
    // Pie slice: intersection of circle and angular wedge with finite edge segments.
    let d_circle = length(p) - radius;
    let mid_angle = angle_start + angle_sweep * 0.5;
    let ca = cos(-mid_angle);
    let sa = sin(-mid_angle);
    let rp = vec2<f32>(p.x * ca - p.y * sa, p.x * sa + p.y * ca);
    let half_sweep = abs(angle_sweep) * 0.5;
    let angle = atan2(rp.y, rp.x);
    if abs(angle) <= half_sweep {
        return d_circle;
    }
    // Distance to finite edge segments (origin to endpoint * radius).
    let ep1 = vec2<f32>(cos(half_sweep), sin(half_sweep)) * radius;
    let ep2 = vec2<f32>(cos(half_sweep), -sin(half_sweep)) * radius;
    let d1 = sdf_line(rp, vec2<f32>(0.0), ep1, 0.0);
    let d2 = sdf_line(rp, vec2<f32>(0.0), ep2, 0.0);
    return max(d_circle, min(d1, d2));
}

fn sdf_capsule(p: vec2<f32>, half_size: vec2<f32>) -> f32 {
    // Stadium: rectangle with fully rounded short ends.
    let r = min(half_size.x, half_size.y);
    let q = abs(p) - half_size + vec2<f32>(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
}

fn sdf_cross(p: vec2<f32>, arm_width: f32, arm_length: f32) -> f32 {
    // Union of two rectangles (axis-aligned cross).
    let hw = arm_width * 0.5;
    let d_horiz = sdf_rounded_rect(p, vec2<f32>(arm_length, hw), vec4<f32>(0.0));
    let d_vert = sdf_rounded_rect(p, vec2<f32>(hw, arm_length), vec4<f32>(0.0));
    return sdf_union(d_horiz, d_vert);
}

fn sdf_bezier(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, thickness: f32) -> f32 {
    // Quadratic bezier distance (approximate via segmentation).
    var min_d: f32 = 1e10;
    let segments: i32 = 16;
    var prev = a;
    for (var i: i32 = 1; i <= segments; i = i + 1) {
        let t = f32(i) / f32(segments);
        let mt = 1.0 - t;
        let pt = mt * mt * a + 2.0 * mt * t * b + t * t * c;
        // Distance to line segment prev->pt.
        let pa = p - prev;
        let ba = pt - prev;
        let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        let d = length(pa - ba * h);
        min_d = min(min_d, d);
        prev = pt;
    }
    return min_d - thickness * 0.5;
}

fn sdf_arbitrary_triangle(p: vec2<f32>, v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>) -> f32 {
    // Exact triangle SDF (Inigo Quilez).
    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;
    let v0p = p - v0;
    let v1p = p - v1;
    let v2p = p - v2;
    let pq0 = v0p - e0 * clamp(dot(v0p, e0) / dot(e0, e0), 0.0, 1.0);
    let pq1 = v1p - e1 * clamp(dot(v1p, e1) / dot(e1, e1), 0.0, 1.0);
    let pq2 = v2p - e2 * clamp(dot(v2p, e2) / dot(e2, e2), 0.0, 1.0);
    let s = sign(e0.x * e2.y - e0.y * e2.x);
    let d0 = vec2<f32>(dot(pq0, pq0), s * (v0p.x * e0.y - v0p.y * e0.x));
    let d1 = vec2<f32>(dot(pq1, pq1), s * (v1p.x * e1.y - v1p.y * e1.x));
    let d2 = vec2<f32>(dot(pq2, pq2), s * (v2p.x * e2.y - v2p.y * e2.x));
    let d = min(min(d0, d1), d2);
    return -sqrt(d.x) * sign(d.y);
}

fn sdf_trapezoid(p: vec2<f32>, top_hw: f32, bot_hw: f32, half_h: f32) -> f32 {
    // Symmetric trapezoid SDF (Inigo Quilez).
    let ap = abs(p);
    let k1 = vec2<f32>(bot_hw, half_h);
    let k2 = vec2<f32>(bot_hw - top_hw, 2.0 * half_h);
    let q = vec2<f32>(ap.x, select(ap.y, half_h - ap.y, p.y < 0.0));
    let ca = vec2<f32>(clamp(q.x, 0.0, select(top_hw, bot_hw, q.y < 0.0)), q.y);
    let cb_t = clamp(dot(q - vec2<f32>(bot_hw, 0.0), k2) / dot(k2, k2), 0.0, 1.0);
    let cb = vec2<f32>(bot_hw, 0.0) + k2 * cb_t;
    let s = select(1.0, -1.0, (cb.x < 0.0) && (ca.y < 0.0));
    let da = dot(q - ca, q - ca);
    let db = dot(q - cb, q - cb);
    return sqrt(min(da, db)) * s;
}

// ── SDF Underline Functions ──

fn sdf_underline_curly(p: vec2<f32>, half_size: vec2<f32>, freq: f32, amp: f32, thick: f32) -> f32 {
    // Distance to a sine-wave centerline, minus half-thickness.
    let wave_y = sin(p.x * freq) * amp;
    let d_center = abs(p.y - wave_y) - thick * 0.5;
    // Clip to horizontal extent.
    let d_box = abs(p.x) - half_size.x;
    return max(d_center, d_box);
}

fn sdf_underline_dotted(p: vec2<f32>, half_size: vec2<f32>, dot_r: f32, spacing: f32) -> f32 {
    // Repeat along x with the given spacing, then distance to a circle.
    var rx = p.x;
    if spacing > 0.001 {
        rx = p.x - round(p.x / spacing) * spacing;
    }
    let d_dot = length(vec2<f32>(rx, p.y)) - dot_r;
    // Clip to horizontal extent.
    let d_box = abs(p.x) - half_size.x;
    return max(d_dot, d_box);
}

fn sdf_underline_dashed(p: vec2<f32>, half_size: vec2<f32>, dash_w: f32, gap_w: f32, thick: f32) -> f32 {
    // Repeat along x with (dash_w + gap_w) period, distance to a rounded rect.
    let period = dash_w + gap_w;
    var rx = p.x;
    if period > 0.001 {
        rx = p.x - round(p.x / period) * period;
    }
    let half_dash = vec2<f32>(dash_w * 0.5, thick * 0.5);
    let q = abs(vec2<f32>(rx, p.y)) - half_dash;
    let d_dash = min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0)));
    // Clip to horizontal extent.
    let d_box = abs(p.x) - half_size.x;
    return max(d_dash, d_box);
}

fn sdf_heart(p: vec2<f32>, scale: f32) -> f32 {
    // Inigo Quilez heart SDF, normalized to fit within [-scale, scale].
    let q = p / scale;
    // Mirror horizontally and shift origin to heart center.
    let px = abs(q.x);
    // Flip y so the point faces up (heart convention: y-up).
    let py = -q.y + 0.5;
    let d = select(
        length(vec2<f32>(px, py) - vec2<f32>(0.25, 0.75))
            - sqrt(2.0) / 4.0,
        select(
            length(vec2<f32>(px - 0.5 * max(px + py, 0.0), py - 0.5 * max(px + py, 0.0))),
            length(vec2<f32>(px, py) - vec2<f32>(0.0, 1.0)) - 1.0,
            px + py > 1.0
        ),
        py < px * (-2.0) + 2.0
    );
    return d * scale;
}

// ── SDF Boolean Operations ──

fn sdf_union(a: f32, b: f32) -> f32 {
    return min(a, b);
}

fn sdf_intersection(a: f32, b: f32) -> f32 {
    return max(a, b);
}

fn sdf_subtraction(a: f32, b: f32) -> f32 {
    return max(a, -b);
}

fn sdf_smooth_union(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn sdf_smooth_intersection(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}

fn sdf_smooth_subtraction(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
    return mix(b, -a, h) + k * h * (1.0 - h);
}

// ── Fragment shader (2D SDF only) ──

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let shape_type = in.flags.x;
    let stroke_width = in.flags.y;
    let gradient_type = in.extra.x;
    let gradient_param = in.extra.y;
    let shadow_blur = in.extra.z;
    let shadow_offset_packed = in.extra.w;

    let p = (in.local_pos - 0.5) * in.rect_size;
    let half_size = in.rect_size * 0.5;

    var d: f32 = 0.0;

    if shape_type == SHAPE_RECT {
        d = sdf_rounded_rect(p, half_size, in.border_radius);
    } else if shape_type == SHAPE_CIRCLE {
        d = sdf_circle(p, in.sdf_params.x);
    } else if shape_type == SHAPE_ELLIPSE {
        d = sdf_ellipse(p, in.sdf_params.xy);
    } else if shape_type == SHAPE_RING {
        d = sdf_ring(p, in.sdf_params.x, in.sdf_params.y);
    } else if shape_type == SHAPE_LINE {
        let pixel_pos = in.rect_origin + in.local_pos * in.rect_size;
        d = sdf_line(pixel_pos, in.sdf_params.xy, in.sdf_params.zw, stroke_width);
    } else if shape_type == SHAPE_ARC {
        d = sdf_arc(p, in.sdf_params.x, in.sdf_params.y, in.sdf_params.z, in.sdf_params.w);
    } else if shape_type == SHAPE_TRIANGLE {
        d = sdf_triangle(p, half_size);
    } else if shape_type == SHAPE_POLYGON {
        d = sdf_polygon(p, half_size.x, in.sdf_params.x);
        d = d - in.border_radius.x;
    } else if shape_type == SHAPE_STAR {
        d = sdf_star(p, in.sdf_params.x, in.sdf_params.y, in.sdf_params.z);
    } else if shape_type == SHAPE_SECTOR {
        d = sdf_sector(p, in.sdf_params.x, in.sdf_params.y, in.sdf_params.z);
    } else if shape_type == SHAPE_CAPSULE {
        d = sdf_capsule(p, half_size);
    } else if shape_type == SHAPE_CROSS {
        d = sdf_cross(p, in.sdf_params.x, in.sdf_params.y);
    } else if shape_type == SHAPE_BEZIER {
        let pixel_pos = in.rect_origin + in.local_pos * in.rect_size;
        let start_pt = in.uv;
        let ctrl_pt = in.sdf_params.xy;
        let end_pt = in.extra.xy;
        d = sdf_bezier(pixel_pos, start_pt, ctrl_pt, end_pt, stroke_width);
    } else if shape_type == SHAPE_ARBITRARY_TRIANGLE {
        let pixel_pos = in.rect_origin + in.local_pos * in.rect_size;
        let v0 = in.sdf_params.xy;
        let v1 = in.sdf_params.zw;
        let v2 = in.extra.xy;
        d = sdf_arbitrary_triangle(pixel_pos, v0, v1, v2);
    } else if shape_type == SHAPE_TRAPEZOID {
        d = sdf_trapezoid(p, in.sdf_params.x, in.sdf_params.y, in.sdf_params.z);
    } else if shape_type == SHAPE_UNDERLINE_CURLY {
        d = sdf_underline_curly(p, half_size, in.sdf_params.x, in.sdf_params.y, in.sdf_params.z);
    } else if shape_type == SHAPE_UNDERLINE_DOTTED {
        d = sdf_underline_dotted(p, half_size, in.sdf_params.x, in.sdf_params.y);
    } else if shape_type == SHAPE_UNDERLINE_DASHED {
        d = sdf_underline_dashed(p, half_size, in.sdf_params.x, in.sdf_params.y, in.sdf_params.z);
    } else if shape_type == SHAPE_HEART {
        d = sdf_heart(p, in.sdf_params.x);
    } else {
        let fo = in.color.a * in.flags.z;
        return vec4<f32>(in.color.rgb * fo, fo);
    }

    let opacity = in.flags.z;

    // ── Shadow / Glow ──
    var shadow_color_out = vec4<f32>(0.0);
    if shadow_blur > 0.001 {
        let packed = shadow_offset_packed;
        let sdx = floor(packed / 256.0) - 128.0;
        let sdy = packed - floor(packed / 256.0) * 256.0 - 128.0;
        let shadow_p = p - vec2<f32>(sdx, sdy);

        var shadow_d = d;
        if shape_type == SHAPE_RECT {
            shadow_d = sdf_rounded_rect(shadow_p, half_size, in.border_radius);
        } else if shape_type == SHAPE_CIRCLE {
            shadow_d = sdf_circle(shadow_p, in.sdf_params.x);
        } else if shape_type == SHAPE_ELLIPSE {
            shadow_d = sdf_ellipse(shadow_p, in.sdf_params.xy);
        } else if shape_type == SHAPE_RING {
            shadow_d = sdf_ring(shadow_p, in.sdf_params.x, in.sdf_params.y);
        } else if shape_type == SHAPE_LINE {
            let shadow_pixel_pos = in.rect_origin + in.local_pos * in.rect_size
                - vec2<f32>(sdx, sdy);
            shadow_d = sdf_line(shadow_pixel_pos, in.sdf_params.xy, in.sdf_params.zw,
                stroke_width);
        } else if shape_type == SHAPE_ARC {
            shadow_d = sdf_arc(shadow_p, in.sdf_params.x, in.sdf_params.y,
                in.sdf_params.z, in.sdf_params.w);
        } else if shape_type == SHAPE_TRIANGLE {
            shadow_d = sdf_triangle(shadow_p, half_size);
        } else if shape_type == SHAPE_POLYGON {
            shadow_d = sdf_polygon(shadow_p, half_size.x, in.sdf_params.x)
                - in.border_radius.x;
        } else if shape_type == SHAPE_STAR {
            shadow_d = sdf_star(shadow_p, in.sdf_params.x, in.sdf_params.y,
                in.sdf_params.z);
        } else if shape_type == SHAPE_SECTOR {
            shadow_d = sdf_sector(shadow_p, in.sdf_params.x, in.sdf_params.y,
                in.sdf_params.z);
        } else if shape_type == SHAPE_CAPSULE {
            shadow_d = sdf_capsule(shadow_p, half_size);
        } else if shape_type == SHAPE_CROSS {
            shadow_d = sdf_cross(shadow_p, in.sdf_params.x, in.sdf_params.y);
        } else if shape_type == SHAPE_TRAPEZOID {
            shadow_d = sdf_trapezoid(shadow_p, in.sdf_params.x, in.sdf_params.y,
                in.sdf_params.z);
        } else if shape_type == SHAPE_HEART {
            shadow_d = sdf_heart(shadow_p, in.sdf_params.x);
        }

        let shadow_alpha = smoothstep(shadow_blur, -shadow_blur * 0.1, shadow_d);
        var sc = in.color2;
        if sc.a < 0.001 {
            sc = vec4<f32>(0.0, 0.0, 0.0, 0.5);
        }
        let shadow_fa = shadow_alpha * sc.a * opacity;
        shadow_color_out = vec4<f32>(sc.rgb * shadow_fa, shadow_fa);
    }

    // Anti-aliasing via fwidth + smoothstep.
    let aa = fwidth(d) * 0.75;

    var alpha: f32;
    var fill_color = in.color;

    let has_radius = (in.border_radius.x + in.border_radius.y
                    + in.border_radius.z + in.border_radius.w) > 0.001;
    if shape_type == SHAPE_RECT && !has_radius && stroke_width <= 0.0
        && shadow_blur <= 0.0 {
        alpha = select(0.0, 1.0, d < 0.0);
    } else if stroke_width > 0.0 && shape_type != SHAPE_LINE && shape_type != SHAPE_BEZIER {
        if in.color2.a > 0.001 && gradient_type < 0.5 {
            let fill_alpha = smoothstep(aa, -aa, d + stroke_width);
            let stroke_alpha = smoothstep(aa, -aa, d) - fill_alpha;
            alpha = fill_alpha + stroke_alpha;
            let border_color = in.color2;
            fill_color = vec4<f32>(
                (fill_color.rgb * fill_alpha + border_color.rgb * stroke_alpha),
                fill_color.a * fill_alpha + border_color.a * stroke_alpha,
            );
        } else {
            let outer = smoothstep(aa, -aa, d);
            let inner = smoothstep(aa, -aa, d + stroke_width);
            alpha = outer - inner;
        }
    } else {
        alpha = smoothstep(aa, -aa, d);
    }

    if alpha < 0.001 && shadow_color_out.a < 0.001 { discard; }

    // ── Gradient ──
    var result = fill_color;
    if gradient_type > 0.5 {
        let centered_uv = in.local_pos - 0.5;
        var t: f32 = 0.0;
        if gradient_type < 1.5 {
            let dir = vec2<f32>(cos(gradient_param), sin(gradient_param));
            t = dot(centered_uv, dir) + 0.5;
        } else if gradient_type < 2.5 {
            t = length(centered_uv) * 2.0;
        } else {
            let angle = atan2(centered_uv.y, centered_uv.x);
            t = fract((angle - gradient_param) / TAU + 0.5);
        }
        t = clamp(t, 0.0, 1.0);
        let color2_premul = in.color2;
        result = mix(result, color2_premul, t);
    }

    let fa = result.a * alpha * opacity;
    result = vec4<f32>(result.rgb * fa, fa);

    if shadow_color_out.a > 0.001 {
        result = result + shadow_color_out * (1.0 - result.a);
    }

    return result;
}
";

/// Fragment shader for raymarched 3D shapes (pipeline 2).
///
/// Contains 3D SDF functions, raymarching, lighting, and the fragment shader
/// for Sphere3D, Torus3D, RoundedBox3D, SlicedTorus3D, and Morph3D.
const RAYMARCHED_3D_FRAGMENT_SOURCE: &str = r"
// ── 3D SDF functions ──

fn sdf3d_sphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn sdf3d_torus(p: vec3<f32>, R: f32, r: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - R, p.y);
    return length(q) - r;
}

fn sdf3d_rounded_box(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

fn map_scene(p: vec3<f32>, shape_type: f32, sdf_params: vec4<f32>) -> f32 {
    if shape_type == SHAPE_SPHERE_3D {
        return sdf3d_sphere(p, sdf_params.x);
    } else if shape_type == SHAPE_TORUS_3D {
        return sdf3d_torus(p, sdf_params.x, sdf_params.y);
    } else {
        return sdf3d_rounded_box(p, sdf_params.xyz, sdf_params.w);
    }
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>, shape_type: f32, sdf_params: vec4<f32>) -> vec2<f32> {
    var t: f32 = 0.0;
    var closest: f32 = 1e10;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let p = ro + rd * t;
        let d = map_scene(p, shape_type, sdf_params);
        closest = min(closest, d);
        if d < 0.0002 {
            return vec2<f32>(t, d);
        }
        t = t + d;
        if t > 20.0 {
            break;
        }
    }
    return vec2<f32>(-1.0, closest);
}

fn calc_normal(p: vec3<f32>, shape_type: f32, sdf_params: vec4<f32>) -> vec3<f32> {
    let e = 0.0005;
    let ex = vec3<f32>(e, 0.0, 0.0);
    let ey = vec3<f32>(0.0, e, 0.0);
    let ez = vec3<f32>(0.0, 0.0, e);
    return normalize(vec3<f32>(
        map_scene(p + ex, shape_type, sdf_params) - map_scene(p - ex, shape_type, sdf_params),
        map_scene(p + ey, shape_type, sdf_params) - map_scene(p - ey, shape_type, sdf_params),
        map_scene(p + ez, shape_type, sdf_params) - map_scene(p - ez, shape_type, sdf_params),
    ));
}

fn sdf2d_iso_triangle(p: vec2<f32>, q: vec2<f32>) -> f32 {
    // iq's isosceles triangle SDF. Apex at origin, base from (-q.x, q.y) to (q.x, q.y).
    let px = abs(p.x);
    let pp = vec2<f32>(px, p.y);
    let a = pp - q * clamp(dot(pp, q) / dot(q, q), 0.0, 1.0);
    let b = pp - q * vec2<f32>(clamp(px / q.x, 0.0, 1.0), 1.0);
    let s = -sign(q.y);
    let d = min(
        vec2<f32>(dot(a, a), s * (px * q.y - p.y * q.x)),
        vec2<f32>(dot(b, b), s * (p.y - q.y))
    );
    return -sqrt(d.x) * sign(d.y);
}

fn sdf2d_arch_logo(p: vec2<f32>, scale: f32, wall: f32, notch_r: f32) -> f32 {
    let ps = p / scale;

    // ── Outer triangle ──
    // Minkowski rounding bows the straight edges outward (convex), matching
    // the real Arch Linux logo's characteristic organic silhouette.
    let rounding = 0.04;
    let outer_hw = 0.44;
    let outer_h = 1.28;
    let p_tri = ps + vec2<f32>(0.0, 0.56);
    let outer = sdf2d_iso_triangle(p_tri, vec2<f32>(outer_hw, outer_h)) - rounding;

    // ── Inner arch cutout (pointed gothic/ogive arch) ──
    // Two large circles whose intersection forms a pointed opening — the
    // real Arch logo has a lancet arch, not a capsule.  The arch peak sits
    // at arch_peak_y and widens to half-width = arch_hw at arch_base_y.
    // Below arch_base_y the y coordinate is clamped so the cutout continues
    // at constant width straight down through the base.
    let arch_peak_y = 0.0;
    let arch_base_y = 0.50;
    let arch_hw = wall;
    let arch_h = arch_base_y - arch_peak_y;
    // Gothic arch geometry: d = (H²-W²)/(2W), R = d + W.
    let d_arch = (arch_h * arch_h - arch_hw * arch_hw) / (2.0 * arch_hw);
    let r_arch = d_arch + arch_hw;
    let cy = min(ps.y, arch_base_y);
    let dy = cy - arch_base_y;
    let c1 = length(vec2<f32>(ps.x + d_arch, dy)) - r_arch;
    let c2 = length(vec2<f32>(ps.x - d_arch, dy)) - r_arch;
    let inner = max(c1, c2);

    var d: f32 = max(outer, -inner);

    // ── Concave base ──
    // A large circle centered below the triangle base carves the straight
    // base into a smooth upward arc between the two feet.
    let base_y = -0.56 + outer_h;
    let base_r = 2.0;
    let base_center_y = base_y + base_r - 0.15;
    let base_carve = length(vec2<f32>(ps.x, ps.y - base_center_y)) - base_r;
    d = max(d, base_carve);

    // ── Head notch (upper slit) ──
    // Thin angled slit near the apex separating the head from the body,
    // matching the Arch Linux logo characteristic diagonal split.
    let n1_cx = 0.08;
    let n1_cy = -0.26;
    let n1_a = -0.50;
    let n1_c = cos(n1_a);
    let n1_s = sin(n1_a);
    let np1 = vec2<f32>(
        (ps.x - n1_cx) * n1_c + (ps.y - n1_cy) * n1_s,
        -(ps.x - n1_cx) * n1_s + (ps.y - n1_cy) * n1_c
    );
    let notch1 = max(abs(np1.x) - 0.012, abs(np1.y) - 0.15);
    d = max(d, -notch1);

    // ── Foot notch (lower-right slit) ──
    let n2_cx = 0.26;
    let n2_cy = 0.44;
    let n2_a = 0.50;
    let n2_c = cos(n2_a);
    let n2_s = sin(n2_a);
    let np2 = vec2<f32>(
        (ps.x - n2_cx) * n2_c + (ps.y - n2_cy) * n2_s,
        -(ps.x - n2_cx) * n2_s + (ps.y - n2_cy) * n2_c
    );
    let notch2 = max(abs(np2.x) - 0.010, abs(np2.y) - 0.12);
    d = max(d, -notch2);

    return d * scale;
}

fn sdf3d_arch_logo(p: vec3<f32>, scale: f32, thickness: f32, wall: f32, notch_r: f32) -> f32 {
    // 2D logo in xy plane (facing camera), extruded along z.
    let d2d = sdf2d_arch_logo(p.xy, scale, wall, notch_r);
    return max(d2d, abs(p.z) - thickness);
}

fn eval_morph_sdf(p: vec3<f32>, shape_id: f32, params: vec2<f32>, extra: vec3<f32>) -> f32 {
    if shape_id < 0.5 {
        return sdf3d_sphere(p, params.x);
    } else if shape_id < 1.5 {
        return sdf3d_torus(p, params.x, params.y);
    } else if shape_id < 2.5 {
        return sdf3d_rounded_box(p, vec3<f32>(params.x, extra.x, extra.y), params.y);
    } else {
        return sdf3d_arch_logo(p, params.x, params.y, extra.x, extra.y);
    }
}

fn map_morph_scene(p: vec3<f32>, sdf_params: vec4<f32>,
                    border_radius: vec4<f32>, extra: vec4<f32>) -> f32 {
    let morph = border_radius.x;
    let shape_a = border_radius.y;
    let shape_b = border_radius.z;
    let da = eval_morph_sdf(p, shape_a, sdf_params.xy, vec3<f32>(0.0));
    let db = eval_morph_sdf(p, shape_b, sdf_params.zw, extra.xyz);
    return mix(da, db, morph);
}

fn raymarch_morph(ro: vec3<f32>, rd: vec3<f32>, sdf_params: vec4<f32>,
                  border_radius: vec4<f32>, extra: vec4<f32>) -> vec2<f32> {
    var t: f32 = 0.0;
    var closest: f32 = 1e10;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let p = ro + rd * t;
        let d = map_morph_scene(p, sdf_params, border_radius, extra);
        closest = min(closest, d);
        if d < 0.0002 {
            return vec2<f32>(t, d);
        }
        t = t + d;
        if t > 20.0 {
            break;
        }
    }
    return vec2<f32>(-1.0, closest);
}

fn calc_normal_morph(p: vec3<f32>, sdf_params: vec4<f32>,
                     border_radius: vec4<f32>, extra: vec4<f32>) -> vec3<f32> {
    let e = 0.0005;
    let ex = vec3<f32>(e, 0.0, 0.0);
    let ey = vec3<f32>(0.0, e, 0.0);
    let ez = vec3<f32>(0.0, 0.0, e);
    return normalize(vec3<f32>(
        map_morph_scene(p + ex, sdf_params, border_radius, extra) - map_morph_scene(p - ex, sdf_params, border_radius, extra),
        map_morph_scene(p + ey, sdf_params, border_radius, extra) - map_morph_scene(p - ey, sdf_params, border_radius, extra),
        map_morph_scene(p + ez, sdf_params, border_radius, extra) - map_morph_scene(p - ez, sdf_params, border_radius, extra),
    ));
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - vec3<f32>(3.0));
    return v * mix(vec3<f32>(1.0), clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

fn phong(pos: vec3<f32>, normal: vec3<f32>, view_dir: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(vec3<f32>(0.8, 1.0, -0.6));
    let ambient = 0.15;
    let diffuse = max(dot(normal, light_dir), 0.0);
    let half_dir = normalize(light_dir + view_dir);
    let specular = pow(max(dot(normal, half_dir), 0.0), 32.0);
    return base_color * (ambient + diffuse * 0.75) + vec3<f32>(1.0) * specular * 0.4;
}

fn rot_y(angle: f32) -> mat3x3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat3x3<f32>(
        vec3<f32>(c, 0.0, s),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(-s, 0.0, c),
    );
}

fn rot_x(angle: f32) -> mat3x3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, c, -s),
        vec3<f32>(0.0, s, c),
    );
}

// ── Fragment shader (3D raymarched) ──

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let shape_type = in.flags.x;

    let ndc = in.local_pos * 2.0 - vec2<f32>(1.0);
    let focal_len = 1.8;
    let ro = vec3<f32>(0.0, 0.0, -2.5);
    let rd_raw = normalize(vec3<f32>(ndc.x, ndc.y, focal_len));

    let angle_y = uniforms.time.x * 0.8;
    let angle_x = uniforms.time.x * 0.5;
    let rotation = rot_x(angle_x) * rot_y(angle_y);
    let rd = rotation * rd_raw;
    let ro_rot = rotation * ro;

    if shape_type == SHAPE_SLICED_TORUS_3D {
        // Use per-instance time from extra.w for rotation and rainbow hue,
        // so the animation stays in sync with the Rust-side clock and can
        // freeze smoothly after the cycle completes.
        let st_time = in.extra.w;
        let ov_rot = rot_x(st_time * 0.5) * rot_y(st_time * 0.8);
        let st_rd = ov_rot * rd_raw;
        let st_ro = ov_rot * ro;

        // Offset ray origin by piece translation (stored in extra.xyz).
        let piece_offset = in.extra.xyz;
        let ro_piece = st_ro - piece_offset;

        let result = raymarch(ro_piece, st_rd, SHAPE_TORUS_3D, in.sdf_params);
        let t_hit = result.x;
        if t_hit < 0.0 { discard; }

        let hit = ro_piece + st_rd * t_hit;

        // Clip planes: two fixed normals.
        let CUT_A = normalize(vec3<f32>(0.15, 1.0, 0.0));
        let CUT_B = normalize(vec3<f32>(0.0, 0.1, 1.0));
        let dot_a = dot(CUT_A, hit);
        let dot_b = dot(CUT_B, hit);

        let a_min = in.border_radius.x;
        let a_max = in.border_radius.y;
        let b_min = in.border_radius.z;
        let b_max = in.border_radius.w;
        if dot_a < a_min || dot_a > a_max || dot_b < b_min || dot_b > b_max {
            discard;
        }

        let normal = calc_normal(hit, SHAPE_TORUS_3D, in.sdf_params);
        let view_dir = normalize(-st_rd);

        var base_color = in.color.rgb;
        if in.flags.w > 0.5 {
            let major_angle = atan2(hit.z, hit.x) / TAU + 0.5;
            let center_on_ring = normalize(vec2<f32>(hit.x, hit.z))
                * in.sdf_params.x;
            let tube_vec = hit
                - vec3<f32>(center_on_ring.x, 0.0, center_on_ring.y);
            let tube_angle = atan2(
                tube_vec.y,
                length(vec2<f32>(hit.x, hit.z)) - in.sdf_params.x,
            ) / TAU + 0.5;
            let hue = major_angle + tube_angle * 0.3
                + st_time * 0.15;
            base_color = hsv_to_rgb(fract(hue), 0.5, 1.0);
        }
        let lit = phong(hit, normal, view_dir, base_color);

        let fresnel = 1.0 - abs(dot(normal, view_dir));
        let pixel_size = fwidth(ndc.x) * 1.5;
        let edge_alpha = 1.0
            - smoothstep(1.0 - pixel_size * 8.0, 1.0, fresnel);
        let a = in.color.a * edge_alpha * in.flags.z;
        return vec4<f32>(lit * a, a);
    } else if shape_type == SHAPE_MORPH_3D {
        let result = raymarch_morph(ro_rot, rd, in.sdf_params, in.border_radius, in.extra);
        let t_hit = result.x;
        if t_hit < 0.0 { discard; }

        let hit = ro_rot + rd * t_hit;
        let normal = calc_normal_morph(hit, in.sdf_params, in.border_radius, in.extra);
        let view_dir = normalize(-rd);

        var base_color = in.color.rgb;
        if in.flags.w > 0.5 {
            let major_angle = atan2(hit.z, hit.x) / TAU + 0.5;
            let height_contrib = hit.y * 0.3 + 0.5;
            let hue = major_angle + height_contrib * 0.3 + uniforms.time.x * 0.15;
            base_color = hsv_to_rgb(fract(hue), 0.5, 1.0);
        }
        let lit = phong(hit, normal, view_dir, base_color);

        let fresnel = 1.0 - abs(dot(normal, view_dir));
        let pixel_size = fwidth(ndc.x) * 1.5;
        let edge_alpha = 1.0 - smoothstep(1.0 - pixel_size * 8.0, 1.0, fresnel);
        let a = in.color.a * edge_alpha * in.flags.z;
        return vec4<f32>(lit * a, a);
    } else {
        // Standard 3D shapes: Sphere3D, Torus3D, RoundedBox3D.
        let result = raymarch(ro_rot, rd, shape_type, in.sdf_params);
        let t_hit = result.x;

        if t_hit < 0.0 { discard; }

        let hit = ro_rot + rd * t_hit;
        let normal = calc_normal(hit, shape_type, in.sdf_params);
        let view_dir = normalize(-rd);

        let rainbow = in.flags.w > 0.5;
        var base_color = in.color.rgb;
        if rainbow {
            let major_angle = atan2(hit.z, hit.x) / TAU + 0.5;
            let center_on_ring = normalize(vec2<f32>(hit.x, hit.z)) * in.sdf_params.x;
            let tube_vec = hit - vec3<f32>(center_on_ring.x, 0.0, center_on_ring.y);
            let tube_angle = atan2(tube_vec.y, length(vec2<f32>(hit.x, hit.z)) - in.sdf_params.x) / TAU + 0.5;
            let hue = major_angle + tube_angle * 0.3 + uniforms.time.x * 0.15;
            base_color = hsv_to_rgb(fract(hue), 0.5, 1.0);
        }
        let lit = phong(hit, normal, view_dir, base_color);

        let fresnel = 1.0 - abs(dot(normal, view_dir));
        let pixel_size = fwidth(ndc.x) * 1.5;
        let edge_alpha = 1.0 - smoothstep(1.0 - pixel_size * 8.0, 1.0, fresnel);
        let a = in.color.a * edge_alpha * in.flags.z;
        return vec4<f32>(lit * a, a);
    }
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_scene_shader_valid_identity() {
        assert!(validate_scene_shader("return in.color;").is_ok());
    }

    #[test]
    fn validate_scene_shader_invalid_syntax() {
        assert!(validate_scene_shader("let x = ;;").is_err());
    }

    #[test]
    fn validate_scene_shader_wrong_return_type() {
        assert!(validate_scene_shader("return 42;").is_err());
    }

    #[test]
    fn validate_scene_shader_empty_body() {
        assert!(
            validate_scene_shader("").is_err(),
            "empty body should fail validation (missing return)"
        );
    }

    #[test]
    fn compose_scene_shader_omits_clip_test() {
        let source = compose_scene_shader("return in.color;");
        assert!(!source.contains("has_clip"));
    }

    #[test]
    fn compose_scene_shader_contains_body() {
        let body = "return in.color;";
        let source = compose_scene_shader(body);
        assert!(source.contains(body));
    }

    #[test]
    fn register_duplicate_detected_by_contains_key() {
        // The duplicate check in register_shader_pipeline uses
        // `self.pipelines.contains_key(&id)` before touching the GPU.
        // We verify the HashMap logic without needing real wgpu handles.
        let mut map: HashMap<ShaderId, String> = HashMap::new();
        map.insert(ShaderId(42), "existing".to_string());
        assert!(map.contains_key(&ShaderId(42)));
    }

    #[test]
    fn reload_nonexistent_detected_by_get() {
        // reload_shader_pipeline checks `self.pipelines.get(&id)` first.
        let reg = PipelineRegistry::new();
        assert!(reg.get(ShaderId(999)).is_none());
    }

    #[test]
    fn unregister_nonexistent_returns_false() {
        let mut reg = PipelineRegistry::new();
        assert!(!reg.unregister_shader_pipeline(ShaderId(42)));
    }

    #[test]
    #[should_panic(expected = "cannot unregister built-in pipeline ID")]
    fn unregister_builtin_panics() {
        let mut reg = PipelineRegistry::new();
        reg.unregister_shader_pipeline(crate::primitive::PIPELINE_SDF_2D);
    }

    #[test]
    fn is_builtin_id_detects_all_builtins() {
        assert!(is_builtin_id(crate::primitive::PIPELINE_SDF_2D));
        assert!(is_builtin_id(crate::primitive::PIPELINE_TEXT));
        assert!(is_builtin_id(crate::primitive::PIPELINE_3D));
        assert!(is_builtin_id(crate::primitive::PIPELINE_SDF_2D_OPAQUE));
        assert!(is_builtin_id(crate::primitive::PIPELINE_SDF_2D_ADDITIVE));
        assert!(is_builtin_id(crate::primitive::PIPELINE_SDF_2D_SCREEN));
        assert!(is_builtin_id(crate::primitive::PIPELINE_SDF_2D_MULTIPLY));
        assert!(is_builtin_id(crate::offscreen::PIPELINE_POST_PROCESS));
        assert!(!is_builtin_id(ShaderId(42)));
    }
}
