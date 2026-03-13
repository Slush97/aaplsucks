//! MSAA depth resolve pass — resolves multisampled depth to a 1x texture.
//!
//! When MSAA is enabled (`sample_count > 1`), the main render pass writes to a
//! multisampled depth texture. Post-processing passes (SSAO, motion blur, SDF)
//! need a regular `texture_depth_2d`, so this pass resolves the MSAA depth by
//! taking the minimum depth across all samples (closest geometry per pixel).

/// Embedded WGSL source for the depth resolve shader.
pub(crate) const DEPTH_RESOLVE_SHADER: &str = include_str!("../../shaders/depth_resolve.wgsl");

/// MSAA depth resolve pass.
pub struct DepthResolvePass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    sample_count: u32,
}

impl DepthResolvePass {
    /// Create a new depth resolve pass.
    ///
    /// `msaa_depth_view` is the multisampled depth texture view from the main render pass.
    /// `sample_count` must be > 1.
    pub fn new(
        device: &wgpu::Device,
        msaa_depth_view: &wgpu::TextureView,
        sample_count: u32,
    ) -> Self {
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esox_depth_resolve_bg_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                }],
            });

        let bind_group = Self::create_bind_group(device, &bind_group_layout, msaa_depth_view);

        let pipeline = Self::create_pipeline(device, &bind_group_layout, DEPTH_RESOLVE_SHADER, sample_count);

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            sample_count,
        }
    }

    /// Rebuild the bind group after depth textures are recreated (e.g. on resize).
    pub fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        msaa_depth_view: &wgpu::TextureView,
    ) {
        self.bind_group =
            Self::create_bind_group(device, &self.bind_group_layout, msaa_depth_view);
    }

    /// Rebuild the pipeline with new shader source (for hot-reload).
    #[cfg(feature = "hot-reload")]
    pub fn rebuild_pipeline(
        &mut self,
        device: &wgpu::Device,
        src: &str,
        sample_count: u32,
    ) {
        self.pipeline =
            Self::create_pipeline(device, &self.bind_group_layout, src, sample_count);
        self.sample_count = sample_count;
    }

    /// Run the depth resolve pass, writing resolved depth into `resolved_depth_view`.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resolved_depth_view: &wgpu::TextureView,
    ) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("esox_depth_resolve_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: resolved_depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1); // fullscreen triangle
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        msaa_depth_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_depth_resolve_bg"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(msaa_depth_view),
            }],
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        shader_src: &str,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_depth_resolve_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("esox_depth_resolve_pipeline_layout"),
                bind_group_layouts: &[bind_group_layout],
                immediate_size: 0,
            });

        let constants = [("SAMPLE_COUNT", sample_count as f64)];

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("esox_depth_resolve_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }
}
