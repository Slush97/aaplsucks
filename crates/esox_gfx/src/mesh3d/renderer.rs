//! 3D mesh renderer — pipeline, depth buffer, frame encoding.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::pipeline::GpuContext;

use super::bounds::{Aabb, Frustum};
use super::bvh::Bvh;
use super::camera::Camera;
use super::ibl::IblState;
use super::instance::InstanceData;
use super::light::{LightEnvironment, LightUniforms};
use super::material::{
    BlendMode3D, Material, MaterialDescriptor, MaterialHandle, MaterialType, MaterialUniforms,
    PipelineKey, create_pipeline, create_pipeline_with_shader,
};
use super::mesh::{MegaBuffer, Mesh, MeshData, MeshHandle, MeshRegion};
use super::depth_resolve::DepthResolvePass;
#[cfg(feature = "hot-reload")]
use super::shader_library::ShaderSlot;
use super::shader_library::ShaderLibrary;
use super::shadow::{ShadowConfig, ShadowPass, ShadowState, ShadowUniforms};
use super::skinning::{SkinningPipeline, SkinnedMesh};
use super::ssao::SsaoPass;
use super::texture::{Texture3D, TextureHandle};

use super::render_types::{
    Uniforms, DrawCmd, BatchStats3D,
    INITIAL_INSTANCE_CAPACITY, MAX_INSTANCES,
    SKINNED_MESH_BIT, INDIRECT_ARGS_SIZE, INITIAL_INDIRECT_CAPACITY,
    pipeline_key_sort_tuple, instance_translation,
};
use super::postprocess::{
    PostProcess3DConfig, PostProcess3D,
    create_depth_texture, create_hdr_texture,
};
use super::shaders_embedded::{SHADER_PREAMBLE, compile_shader_modules};

// ── Renderer ──

/// 3D mesh renderer.
///
/// Manages render pipelines, materials, lighting, depth buffer, instance buffer,
/// and frame encoding. Shares a [`GpuContext`] with the 2D esox renderer.
pub struct Renderer3D {
    // Bind group layouts.
    #[allow(dead_code)]
    scene_bind_group_layout: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    light_bind_group_layout: wgpu::BindGroupLayout,
    material_bind_group_layout: wgpu::BindGroupLayout,

    // Shared pipeline layout (3 groups: scene, light, material).
    pipeline_layout: wgpu::PipelineLayout,

    // Scene uniforms (group 0).
    scene_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,

    // Lighting (group 1).
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    pub(super) light_env: LightEnvironment,

    // Pipeline cache: key -> pipeline.
    pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
    shader_modules: HashMap<MaterialType, wgpu::ShaderModule>,
    pub(super) surface_format: wgpu::TextureFormat,
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    pub(super) shader_library: ShaderLibrary,

    // Materials.
    materials: Vec<Material>,

    // Textures.
    textures: Vec<Texture3D>,
    fallback_albedo: Texture3D,
    fallback_normal: Texture3D,
    fallback_mr: Texture3D,
    shared_sampler: wgpu::Sampler,

    // Instancing.
    pub(super) instance_buffer: wgpu::Buffer,
    instance_capacity: u64,

    // Depth — render target (MSAA when sample_count > 1).
    #[allow(dead_code)]
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    // Depth — 1x sampling view for post-processing (SSAO).
    // When sample_count == 1 this is the same texture/view as above.
    #[allow(dead_code)]
    depth_sample_texture: Option<wgpu::Texture>,
    pub(super) depth_sample_view: wgpu::TextureView,
    depth_width: u32,
    depth_height: u32,

    // Meshes — mega-buffer shared VB/IB for static meshes.
    pub(crate) mega_buffer: MegaBuffer,
    pub(crate) mesh_regions: Vec<MeshRegion>,
    /// Legacy per-mesh buffers (kept for skinned meshes that write their own VB).
    pub(crate) meshes: Vec<Mesh>,

    // Skinning.
    pub(crate) skinning_pipeline: Option<SkinningPipeline>,
    pub(crate) skinned_meshes: Vec<SkinnedMesh>,

    // Particles.
    pub(crate) particle_pipeline: Option<super::particles::ParticlePipeline>,
    pub(crate) particle_pools: Vec<super::particles::ParticlePool>,
    pub(crate) particle_draw_cmds: Vec<super::particles::ParticleDrawCmd>,
    pub(crate) particle_quad_mesh: Option<MeshHandle>,

    // Per-frame state.
    draw_cmds: Vec<DrawCmd>,
    instance_staging: Vec<InstanceData>,

    // BVH culling threshold — use BVH when draw count exceeds this.
    bvh_threshold: usize,
    // Scratch buffer for world-space AABBs (reused each frame).
    world_aabbs_scratch: Vec<Aabb>,

    // Multi-draw-indirect support.
    multi_draw_indirect: bool,
    indirect_buffer: Option<wgpu::Buffer>,
    indirect_capacity: u32,

    // ── Phase 4: Visual Quality ──

    // Post-processing (offscreen HDR + bloom + tone mapping + SSAO composite).
    pub(super) postprocess: Option<PostProcess3D>,

    // Shadow maps (CSM, point, spot — fallback textures, uniform buffers, samplers).
    pub(super) shadow_state: super::shadow::ShadowState,

    // SSAO.
    pub(super) ssao_pass: Option<SsaoPass>,
    pub(super) fallback_ssao_view: wgpu::TextureView,

    #[allow(dead_code)]
    fallback_ssao_texture: wgpu::Texture,

    // IBL (image-based lighting).
    ibl_state: IblState,
    ibl_sampler: wgpu::Sampler,

    // MSAA depth resolve.
    pub(super) depth_resolve_pass: Option<DepthResolvePass>,

    // MSAA sample count (1 = off, 4 = 4x).
    pub(super) sample_count: u32,
}

impl Renderer3D {
    /// Create a new 3D renderer using the given GPU context.
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &*gpu.device;

        // ── Bind group layouts ──

        // Scene bind group (group 0): uniforms + shadow uniforms + shadow depth array + comparison sampler.
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esox_3d_scene_layout"),
                entries: &[
                    // binding 0: scene uniforms
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
                    // binding 1: shadow uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<ShadowUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    // binding 2: shadow depth texture array
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 3: comparison sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                    // binding 4: omni shadow uniforms (point + spot shadow VP matrices)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<super::shadow::OmniShadowUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                    // binding 5: point light shadow depth texture array (24 layers)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 6: spot light shadow depth texture array (4 layers)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Light bind group (group 1): light uniforms + IBL textures + IBL sampler.
        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esox_3d_light_layout"),
                entries: &[
                    // binding 0: light uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: irradiance cubemap
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2: prefiltered env cubemap
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 3: BRDF LUT
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 4: IBL sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esox_3d_material_layout"),
                entries: &[
                    // binding 0: MaterialUniforms buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: albedo texture (sRGB)
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
                    // binding 2: normal texture (linear)
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
                    // binding 3: metallic-roughness texture (linear)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 4: emissive texture (sRGB)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 5: shared sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // ── Shared pipeline layout ──
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("esox_3d_pipeline_layout"),
            bind_group_layouts: &[
                &scene_bind_group_layout,
                &light_bind_group_layout,
                &material_bind_group_layout,
            ],
            immediate_size: 0,
        });

        // ── Buffers ──

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_uniforms"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let light_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_light_uniforms"),
            size: size_of::<LightUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_state = ShadowState::new(device, &gpu.queue);

        let instance_capacity = INITIAL_INSTANCE_CAPACITY;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_instance_buffer"),
            size: instance_capacity * size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Fallback resources ──

        // Fallback 1x1 R8Unorm white texture for when SSAO is disabled (AO = 1.0).
        let fallback_ssao_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("esox_3d_fallback_ssao"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_ssao_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let fallback_ssao_view =
            fallback_ssao_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // IBL fallback (1x1 white cubemaps + BRDF LUT).
        let ibl_state = IblState::fallback(device, &gpu.queue);

        let ibl_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("esox_3d_ibl_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // ── Bind groups ──

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_scene_bg"),
            layout: &scene_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: shadow_state.shadow_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_state.fallback_shadow_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&shadow_state.comparison_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: shadow_state.omni_shadow_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&shadow_state.fallback_point_shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&shadow_state.fallback_spot_shadow_view),
                },
            ],
        });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_light_bg"),
            layout: &light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ibl_state.irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ibl_state.prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&ibl_state.brdf_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&ibl_sampler),
                },
            ],
        });

        // ── Shader library + modules ──

        let shader_dir = option_env!("CARGO_MANIFEST_DIR")
            .map(|d| PathBuf::from(d).join("shaders"));
        let shader_library = ShaderLibrary::new(shader_dir);
        let shader_modules = compile_shader_modules(device, &shader_library);

        // ── Pipeline cache — eagerly create 3 opaque pipelines ──

        let mut pipeline_cache = HashMap::new();
        let format = gpu.config.format;
        for &mat_type in &[MaterialType::Unlit, MaterialType::Lit, MaterialType::PBR, MaterialType::Toon] {
            let key = PipelineKey {
                material_type: mat_type,
                blend_mode: BlendMode3D::Opaque,
                cull_mode: super::material::CullMode3D::Back,
                depth_write: true,
            };
            let pipeline =
                create_pipeline(device, format, &pipeline_layout, &shader_modules, &key, gpu.sample_count);
            pipeline_cache.insert(key, pipeline);
        }

        // ── Shared sampler + fallback textures ──

        let shared_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("esox_3d_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let fallback_albedo = Texture3D::fallback_white(device, &gpu.queue);
        let fallback_normal = Texture3D::fallback_normal(device, &gpu.queue);
        let fallback_mr = Texture3D::fallback_metallic_roughness(device, &gpu.queue);

        // ── Default material (handle 0, white Lit) ──

        let default_desc = MaterialDescriptor::default();
        let default_uniforms = default_desc.to_uniforms();
        let default_mat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_material_0"),
            size: size_of::<MaterialUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &default_mat_buffer,
            0,
            bytemuck::bytes_of(&default_uniforms),
        );

        let default_mat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_material_bg_0"),
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: default_mat_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&fallback_albedo.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&fallback_normal.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&fallback_mr.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&fallback_albedo.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&shared_sampler),
                },
            ],
        });

        let materials = vec![Material {
            pipeline_key: default_desc.pipeline_key(),
            uniform_buffer: default_mat_buffer,
            bind_group: default_mat_bg,
            texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            emissive_texture: None,
            descriptor: default_desc.clone(),
        }];

        // ── Depth texture ──

        let (depth_texture, depth_view) =
            create_depth_texture(device, gpu.config.width, gpu.config.height, gpu.sample_count);
        // Separate 1x depth for sampling by SSAO when MSAA is active.
        let (depth_sample_texture, depth_sample_view) = if gpu.sample_count > 1 {
            let (t, v) = create_depth_texture(device, gpu.config.width, gpu.config.height, 1);
            (Some(t), v)
        } else {
            let v = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            (None, v)
        };

        // ── MSAA depth resolve pass ──
        let depth_resolve_pass = if gpu.sample_count > 1 {
            Some(DepthResolvePass::new(device, &depth_view, gpu.sample_count))
        } else {
            None
        };

        // ── Mega-buffer ──

        let mega_buffer = MegaBuffer::new(device);

        // ── Multi-draw-indirect feature detection ──

        // multi_draw_indexed_indirect is always available in wgpu 28+.
        let multi_draw_indirect = gpu.multi_draw_indirect;

        Self {
            scene_bind_group_layout,
            light_bind_group_layout,
            material_bind_group_layout,
            pipeline_layout,
            scene_bind_group,
            uniform_buffer,
            light_buffer,
            light_bind_group,
            light_env: LightEnvironment::default(),
            pipeline_cache,
            shader_modules,
            surface_format: format,
            shader_library,
            materials,
            textures: Vec::new(),
            fallback_albedo,
            fallback_normal,
            fallback_mr,
            shared_sampler,
            instance_buffer,
            instance_capacity,
            depth_texture,
            depth_view,
            depth_sample_texture,
            depth_sample_view,
            depth_width: gpu.config.width,
            depth_height: gpu.config.height,
            mega_buffer,
            mesh_regions: Vec::new(),
            meshes: Vec::new(),
            skinning_pipeline: None,
            skinned_meshes: Vec::new(),
            particle_pipeline: None,
            particle_pools: Vec::new(),
            particle_draw_cmds: Vec::new(),
            particle_quad_mesh: None,
            draw_cmds: Vec::new(),
            instance_staging: Vec::new(),
            bvh_threshold: 4096,
            world_aabbs_scratch: Vec::new(),
            multi_draw_indirect,
            indirect_buffer: None,
            indirect_capacity: 0,
            postprocess: None,
            shadow_state,
            ssao_pass: None,
            fallback_ssao_view,
            fallback_ssao_texture,
            ibl_state,
            ibl_sampler,
            depth_resolve_pass,
            sample_count: gpu.sample_count,
        }
    }

    /// Upload mesh geometry to the shared mega-buffer and return a handle.
    ///
    /// Computes the AABB at upload time for frustum culling.
    pub fn upload_mesh(&mut self, gpu: &GpuContext, data: &MeshData) -> MeshHandle {
        let aabb = data.compute_aabb();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("esox_3d_mega_upload"),
            });
        let region = self.mega_buffer.append(&gpu.device, &gpu.queue, &mut encoder, data, aabb);
        gpu.queue.submit(std::iter::once(encoder.finish()));
        let handle = MeshHandle(self.mesh_regions.len() as u32);
        self.mesh_regions.push(region);
        handle
    }

    /// Upload mesh geometry as a standalone buffer (for skinned meshes).
    ///
    /// Skinned meshes have their vertex buffer written by compute shaders,
    /// so they can't use the shared mega-buffer.
    pub fn upload_mesh_standalone(&mut self, gpu: &GpuContext, data: &MeshData) -> MeshHandle {
        let mesh = Mesh::upload(&gpu.device, data);
        let handle = MeshHandle(self.meshes.len() as u32 | SKINNED_MESH_BIT);
        self.meshes.push(mesh);
        handle
    }

    /// Upload an RGBA8 texture (sRGB) and return a handle.
    ///
    /// Returns `None` if `data.len() != width * height * 4`.
    pub fn upload_texture(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Option<TextureHandle> {
        let tex = Texture3D::upload(&gpu.device, &gpu.queue, width, height, data)?;
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(tex);
        Some(handle)
    }

    /// Upload an RGBA8 texture (linear) and return a handle.
    ///
    /// Use for data textures like normal maps and metallic-roughness maps.
    pub fn upload_texture_linear(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Option<TextureHandle> {
        let tex = Texture3D::upload_linear(&gpu.device, &gpu.queue, width, height, data)?;
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(tex);
        Some(handle)
    }

    /// Upload a texture from encoded image bytes (PNG/JPEG).
    #[cfg(feature = "mesh3d")]
    pub fn upload_texture_from_bytes(
        &mut self,
        gpu: &GpuContext,
        data: &[u8],
        srgb: bool,
    ) -> Option<TextureHandle> {
        let tex = Texture3D::upload_from_bytes(&gpu.device, &gpu.queue, data, srgb)?;
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(tex);
        Some(handle)
    }

    /// Resolve a texture handle to a view, falling back to the given fallback texture.
    fn resolve_texture_view<'a>(
        &'a self,
        handle: Option<TextureHandle>,
        fallback: &'a Texture3D,
    ) -> &'a wgpu::TextureView {
        match handle {
            Some(h) => {
                let idx = h.0 as usize;
                if idx < self.textures.len() {
                    &self.textures[idx].view
                } else {
                    &fallback.view
                }
            }
            None => &fallback.view,
        }
    }

    /// Build a material bind group with uniform buffer, 4 textures, and sampler.
    fn create_material_bind_group(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        albedo_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        mr_view: &wgpu::TextureView,
        emissive_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_material_bg"),
            layout: &self.material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(mr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(emissive_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.shared_sampler),
                },
            ],
        })
    }

    /// Resolve all 4 texture views for a material descriptor and create the bind group.
    fn create_material_bind_group_from_desc(
        &self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        desc: &MaterialDescriptor,
    ) -> wgpu::BindGroup {
        let albedo_view = self.resolve_texture_view(desc.texture, &self.fallback_albedo);
        let normal_view = self.resolve_texture_view(desc.normal_texture, &self.fallback_normal);
        let mr_view =
            self.resolve_texture_view(desc.metallic_roughness_texture, &self.fallback_mr);
        let emissive_view =
            self.resolve_texture_view(desc.emissive_texture, &self.fallback_albedo);
        self.create_material_bind_group(device, buffer, albedo_view, normal_view, mr_view, emissive_view)
    }

    /// Create a material from a descriptor and return a handle.
    pub fn create_material(
        &mut self,
        gpu: &GpuContext,
        desc: &MaterialDescriptor,
    ) -> MaterialHandle {
        let uniforms = desc.to_uniforms();
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_material"),
            size: size_of::<MaterialUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&uniforms));

        let bind_group = self.create_material_bind_group_from_desc(&gpu.device, &buffer, desc);

        let key = desc.pipeline_key();

        // Ensure pipeline exists for this key.
        if !self.pipeline_cache.contains_key(&key) {
            let pipeline = create_pipeline(
                &gpu.device,
                self.surface_format,
                &self.pipeline_layout,
                &self.shader_modules,
                &key,
                self.sample_count,
            );
            self.pipeline_cache.insert(key, pipeline);
        }

        let handle = MaterialHandle(self.materials.len() as u32);
        self.materials.push(Material {
            pipeline_key: key,
            uniform_buffer: buffer,
            bind_group,
            texture: desc.texture,
            normal_texture: desc.normal_texture,
            metallic_roughness_texture: desc.metallic_roughness_texture,
            emissive_texture: desc.emissive_texture,
            descriptor: desc.clone(),
        });
        handle
    }

    /// Update an existing material's uniform data.
    pub fn update_material(
        &mut self,
        gpu: &GpuContext,
        handle: MaterialHandle,
        desc: &MaterialDescriptor,
    ) {
        let idx = handle.0 as usize;
        if idx >= self.materials.len() {
            tracing::warn!("invalid material handle {}", handle.0);
            return;
        }
        let uniforms = desc.to_uniforms();
        gpu.queue.write_buffer(
            &self.materials[idx].uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );

        let new_key = desc.pipeline_key();
        let textures_changed = self.materials[idx].texture != desc.texture
            || self.materials[idx].normal_texture != desc.normal_texture
            || self.materials[idx].metallic_roughness_texture != desc.metallic_roughness_texture
            || self.materials[idx].emissive_texture != desc.emissive_texture;

        if self.materials[idx].pipeline_key != new_key {
            if !self.pipeline_cache.contains_key(&new_key) {
                let pipeline = create_pipeline(
                    &gpu.device,
                    self.surface_format,
                    &self.pipeline_layout,
                    &self.shader_modules,
                    &new_key,
                    self.sample_count,
                );
                self.pipeline_cache.insert(new_key, pipeline);
            }
            self.materials[idx].pipeline_key = new_key;
        }

        if textures_changed {
            self.materials[idx].bind_group = self.create_material_bind_group_from_desc(
                &gpu.device,
                &self.materials[idx].uniform_buffer,
                desc,
            );
            self.materials[idx].texture = desc.texture;
            self.materials[idx].normal_texture = desc.normal_texture;
            self.materials[idx].metallic_roughness_texture = desc.metallic_roughness_texture;
            self.materials[idx].emissive_texture = desc.emissive_texture;
        }

        self.materials[idx].descriptor = desc.clone();
    }

    /// Get the descriptor for an existing material.
    pub fn material_descriptor(&self, handle: MaterialHandle) -> Option<&MaterialDescriptor> {
        self.materials.get(handle.0 as usize).map(|m| &m.descriptor)
    }

    /// Create a material with a custom WGSL fragment shader.
    ///
    /// The shader must define `fn fs_main(in: VertexOutput) -> @location(0) vec4<f32>`.
    /// Returns an error if the WGSL fails to compile.
    pub fn create_custom_material(
        &mut self,
        gpu: &GpuContext,
        wgsl: &str,
        desc: &MaterialDescriptor,
    ) -> Result<MaterialHandle, String> {
        let full_source = format!("{SHADER_PREAMBLE}\n{wgsl}");
        let module = naga::front::wgsl::parse_str(&full_source)
            .map_err(|e| format!("WGSL parse error: {e}"))?;
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .map_err(|e| format!("WGSL validation error: {e}"))?;
        let _ = info;

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("esox_3d_custom_shader"),
                source: wgpu::ShaderSource::Wgsl(full_source.into()),
            });

        let uniforms = desc.to_uniforms();
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_custom_material"),
            size: size_of::<MaterialUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(&uniforms));

        let bind_group = self.create_material_bind_group_from_desc(&gpu.device, &buffer, desc);

        let key = desc.pipeline_key();
        let pipeline = create_pipeline_with_shader(
            &gpu.device,
            self.surface_format,
            &self.pipeline_layout,
            &shader,
            &key,
            self.sample_count,
        );
        self.pipeline_cache.insert(key, pipeline);

        let handle = MaterialHandle(self.materials.len() as u32);
        self.materials.push(Material {
            pipeline_key: key,
            uniform_buffer: buffer,
            bind_group,
            texture: desc.texture,
            normal_texture: desc.normal_texture,
            metallic_roughness_texture: desc.metallic_roughness_texture,
            emissive_texture: desc.emissive_texture,
            descriptor: desc.clone(),
        });
        Ok(handle)
    }

    /// Set the light environment for subsequent frames.
    pub fn set_lights(&mut self, env: &LightEnvironment) {
        self.light_env = env.clone();
    }

    /// Poll for shader file changes and rebuild affected pipelines.
    #[cfg(feature = "hot-reload")]
    pub fn poll_shader_reload(&mut self, gpu: &GpuContext) {
        let changed = self.shader_library.poll_changes();
        if changed.is_empty() {
            return;
        }

        let device = &*gpu.device;

        // Determine what needs rebuilding.
        let mut rebuild_materials = false;
        let mut rebuild_composite = false;
        let mut rebuild_shadow = false;
        let mut rebuild_ssao = false;
        let mut rebuild_skinning = false;
        let mut rebuild_depth_resolve = false;
        let mut rebuild_bloom = false;

        for slot in &changed {
            match slot {
                ShaderSlot::Preamble | ShaderSlot::FsUnlit | ShaderSlot::FsLit | ShaderSlot::FsPbr | ShaderSlot::FsToon => {
                    rebuild_materials = true;
                }
                ShaderSlot::Composite => rebuild_composite = true,
                ShaderSlot::ShadowVertex => rebuild_shadow = true,
                ShaderSlot::Ssao | ShaderSlot::SsaoBlur => rebuild_ssao = true,
                ShaderSlot::Skinning => rebuild_skinning = true,
                ShaderSlot::DepthResolve => rebuild_depth_resolve = true,
                ShaderSlot::BloomDownsample | ShaderSlot::BloomUpsample => rebuild_bloom = true,
            }
        }

        if rebuild_materials {
            self.shader_modules = compile_shader_modules(device, &self.shader_library);
            self.rebuild_pipeline_cache(device);
            tracing::info!("Rebuilt material shader pipelines");
        }

        if rebuild_composite {
            if let Some(pp) = &mut self.postprocess {
                let src = self.shader_library.get(ShaderSlot::Composite);
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("esox_3d_composite_shader"),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("esox_3d_composite_pipeline_layout"),
                    bind_group_layouts: &[&pp.composite_bind_group_layout],
                    immediate_size: 0,
                });
                pp.composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("esox_3d_composite_pipeline"),
                    layout: Some(&layout),
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
                tracing::info!("Rebuilt composite pipeline");
            }
        }

        if rebuild_shadow {
            if let Some(sp) = &mut self.shadow_state.shadow_pass {
                let src = self.shader_library.get(ShaderSlot::ShadowVertex);
                sp.rebuild_pipeline(device, src);
                tracing::info!("Rebuilt shadow pipeline");
            }
        }

        if rebuild_ssao {
            if let Some(ssao) = &mut self.ssao_pass {
                let ssao_src = self.shader_library.get(ShaderSlot::Ssao);
                let blur_src = self.shader_library.get(ShaderSlot::SsaoBlur);
                ssao.rebuild_pipelines(device, ssao_src, blur_src);
                tracing::info!("Rebuilt SSAO pipelines");
            }
        }

        if rebuild_skinning {
            if let Some(sp) = &mut self.skinning_pipeline {
                let src = self.shader_library.get(ShaderSlot::Skinning);
                sp.rebuild_pipeline(device, src);
                tracing::info!("Rebuilt skinning pipeline");
            }
        }

        if rebuild_depth_resolve {
            if let Some(resolve) = &mut self.depth_resolve_pass {
                let src = self.shader_library.get(ShaderSlot::DepthResolve);
                resolve.rebuild_pipeline(device, src, self.sample_count);
                tracing::info!("Rebuilt depth resolve pipeline");
            }
        }

        if rebuild_bloom {
            if let Some(pp) = &mut self.postprocess {
                let down_src = self.shader_library.get(ShaderSlot::BloomDownsample);
                let up_src = self.shader_library.get(ShaderSlot::BloomUpsample);
                let bloom_bgl = pp.bloom_pass.bind_group_layout();
                let bloom_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("esox_3d_bloom_pipeline_layout"),
                    bind_group_layouts: &[bloom_bgl],
                    immediate_size: 0,
                });
                let create_bloom = |src: &str, label: &str, blend: Option<wgpu::BlendState>| -> wgpu::RenderPipeline {
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
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
                pp.bloom_down_pipeline = create_bloom(down_src, "esox_3d_bloom_down", None);
                pp.bloom_up_pipeline = create_bloom(up_src, "esox_3d_bloom_up", Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                }));
                tracing::info!("Rebuilt bloom pipelines");
            }
        }
    }

    /// Rebuild all cached material pipelines for the current `surface_format`.
    pub(super) fn rebuild_pipeline_cache(&mut self, device: &wgpu::Device) {
        let sample_count = self.sample_count;
        let new_cache: HashMap<PipelineKey, wgpu::RenderPipeline> = self
            .pipeline_cache
            .keys()
            .map(|key| {
                let pipeline = create_pipeline(
                    device,
                    self.surface_format,
                    &self.pipeline_layout,
                    &self.shader_modules,
                    key,
                    sample_count,
                );
                (*key, pipeline)
            })
            .collect();
        self.pipeline_cache = new_cache;
    }

    /// Get the current post-process configuration.
    pub fn postprocess_config(&self) -> Option<PostProcess3DConfig> {
        self.postprocess.as_ref().map(|pp| pp.config)
    }

    /// Get the current shadow configuration.
    pub fn shadow_config(&self) -> Option<ShadowConfig> {
        self.shadow_state.shadow_pass.as_ref().map(|sp| sp.config)
    }

    /// Set the post-process configuration.
    pub fn set_postprocess(&mut self, config: PostProcess3DConfig) {
        if let Some(pp) = &mut self.postprocess {
            pp.config = config;
        }
    }

    /// Rebuild the scene bind group (Group 0) — called when shadow passes change.
    fn rebuild_scene_bind_group(&mut self, device: &wgpu::Device) {
        let csm_view = self
            .shadow_state.shadow_pass
            .as_ref()
            .map(|sp| &sp.depth_view)
            .unwrap_or(&self.shadow_state.fallback_shadow_depth_view);
        let point_view = self
            .shadow_state.point_shadow_pass
            .as_ref()
            .map(|p| &p.depth_view)
            .unwrap_or(&self.shadow_state.fallback_point_shadow_view);
        let spot_view = self
            .shadow_state.spot_shadow_pass
            .as_ref()
            .map(|s| &s.depth_view)
            .unwrap_or(&self.shadow_state.fallback_spot_shadow_view);

        self.scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_scene_bg"),
            layout: &self.scene_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.shadow_state.shadow_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(csm_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_state.comparison_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.shadow_state.omni_shadow_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(point_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(spot_view),
                },
            ],
        });
    }

    /// Enable cascaded shadow maps.
    pub fn enable_shadows(&mut self, gpu: &GpuContext) {
        if self.shadow_state.shadow_pass.is_some() {
            return;
        }
        self.shadow_state.shadow_pass = Some(ShadowPass::new(&gpu.device));
        self.rebuild_scene_bind_group(&gpu.device);
    }

    /// Enable point light shadow maps (cube map atlas, up to 4 lights).
    pub fn enable_point_shadows(&mut self, gpu: &GpuContext) {
        if self.shadow_state.point_shadow_pass.is_some() {
            return;
        }
        // The shadow pass pipeline shares the same bind group layout as the CSM pass.
        // We need to get the layout from either the existing shadow pass or create one.
        let shadow_pass = self
            .shadow_state.shadow_pass
            .get_or_insert_with(|| ShadowPass::new(&gpu.device));
        let layout = &shadow_pass.bind_group_layout;
        self.shadow_state.point_shadow_pass = Some(super::shadow::PointShadowPass::new(&gpu.device, layout));
        self.rebuild_scene_bind_group(&gpu.device);
    }

    /// Enable spot light shadow maps (up to 4 lights).
    pub fn enable_spot_shadows(&mut self, gpu: &GpuContext) {
        if self.shadow_state.spot_shadow_pass.is_some() {
            return;
        }
        let shadow_pass = self
            .shadow_state.shadow_pass
            .get_or_insert_with(|| ShadowPass::new(&gpu.device));
        let layout = &shadow_pass.bind_group_layout;
        self.shadow_state.spot_shadow_pass = Some(super::shadow::SpotShadowPass::new(&gpu.device, layout));
        self.rebuild_scene_bind_group(&gpu.device);
    }

    /// Set the shadow configuration.
    pub fn set_shadow_config(&mut self, config: ShadowConfig) {
        if let Some(sp) = &mut self.shadow_state.shadow_pass {
            sp.config = config;
        }
    }

    /// Enable SSAO (requires post-processing to be enabled).
    pub fn enable_ssao(&mut self, gpu: &GpuContext) {
        if self.ssao_pass.is_some() {
            return;
        }
        let w = gpu.config.width.max(1);
        let h = gpu.config.height.max(1);
        self.ssao_pass = Some(SsaoPass::new(&gpu.device, w, h));
    }

    /// Set the SSAO configuration.
    pub fn set_ssao_config(&mut self, config: super::ssao::SsaoConfig) {
        if let Some(sp) = &mut self.ssao_pass {
            sp.config = config;
        }
    }

    /// Load an equirectangular HDR environment map for IBL.
    ///
    /// `hdr_data` is a flat array of f32 RGB pixels (width * height * 3).
    pub fn load_environment_map(
        &mut self,
        gpu: &GpuContext,
        hdr_data: &[f32],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        self.ibl_state = IblState::from_equirect(&gpu.device, &gpu.queue, hdr_data, width, height)?;
        self.rebuild_light_bind_group(gpu);
        Ok(())
    }

    /// Generate procedural sky IBL from the current directional light.
    ///
    /// Uses the directional light's direction, color, and intensity to create
    /// a sky environment map, then runs the full IBL precomputation pipeline.
    pub fn generate_procedural_ibl(&mut self, gpu: &GpuContext) {
        let dir_light = &self.light_env.directional;
        let sun_dir = glam::Vec3::from(dir_light.direction).normalize() * -1.0;
        let sun_color = glam::Vec3::from(dir_light.color);
        let sun_intensity = dir_light.intensity;
        let sky_color = glam::Vec3::new(0.4, 0.6, 1.0);
        let ground_color = glam::Vec3::new(0.15, 0.12, 0.1);

        self.ibl_state = IblState::from_procedural_sky(
            &gpu.device,
            &gpu.queue,
            sun_dir,
            sun_color,
            sun_intensity,
            sky_color,
            ground_color,
        );
        self.rebuild_light_bind_group(gpu);
    }

    /// Rebuild the light bind group after IBL textures change.
    fn rebuild_light_bind_group(&mut self, gpu: &GpuContext) {
        self.light_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_light_bg"),
            layout: &self.light_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ibl_state.irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ibl_state.prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.ibl_state.brdf_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.ibl_sampler),
                },
            ],
        });
    }

    /// Get the local-space AABB for a mesh handle (mega-buffer meshes only).
    pub fn mesh_local_aabb(&self, handle: MeshHandle) -> Option<Aabb> {
        let idx = handle.0 as usize;
        if (handle.0 & SKINNED_MESH_BIT) != 0 || idx >= self.mesh_regions.len() {
            return None;
        }
        Some(self.mesh_regions[idx].aabb)
    }

    /// Queue a draw command with a specific material.
    pub fn draw_with_material(
        &mut self,
        mesh: MeshHandle,
        material: MaterialHandle,
        instances: &[InstanceData],
    ) {
        if instances.is_empty() {
            return;
        }
        let offset = self.instance_staging.len() as u32;
        let count = instances.len() as u32;
        self.instance_staging.extend_from_slice(instances);
        self.draw_cmds.push(DrawCmd {
            mesh,
            material,
            instance_offset: offset,
            instance_count: count,
        });
    }

    /// Queue a draw command using the default material (handle 0, white Lit).
    pub fn draw(&mut self, mesh: MeshHandle, instances: &[InstanceData]) {
        self.draw_with_material(mesh, MaterialHandle(0), instances);
    }

    /// Encode and submit the 3D render pass, returning the command buffer and batch stats.
    ///
    /// Renders into `target` (which could be the swapchain texture or an offscreen layer).
    /// Clears the draw list after encoding.
    ///
    /// Phase 3 optimizations applied:
    /// - Frustum culling (linear or BVH-accelerated for large scenes)
    /// - Mega-buffer: single VB/IB bind per frame
    /// - Multi-draw-indirect when supported (batches draw calls per pipeline+material)
    ///
    /// Phase 4 additions:
    /// - Shadow pass (cascaded shadow maps) before scene rendering
    /// - Optional offscreen HDR rendering with bloom and SSAO
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        gpu: &GpuContext,
        target: &wgpu::TextureView,
        camera: &Camera,
        viewport_width: u32,
        viewport_height: u32,
        elapsed: f32,
        delta: f32,
        clear_color: wgpu::Color,
    ) -> (wgpu::CommandBuffer, BatchStats3D) {
        self.resize_if_needed(gpu, viewport_width, viewport_height);
        self.upload_instances(gpu);
        self.upload_scene_uniforms(gpu, camera, viewport_width, viewport_height, elapsed, delta);

        let mut stats = BatchStats3D::default();
        let shadow_opaque_cmds = self.collect_shadow_opaque_cmds();

        let aspect = viewport_width as f32 / viewport_height.max(1) as f32;
        let vp = camera.view_projection(aspect);
        self.frustum_cull(&vp, &mut stats);

        let (ordered, opaque_count) = self.partition_and_sort(camera);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("esox_3d_encoder"),
            });

        self.encode_shadow_passes(gpu, &mut encoder, camera, viewport_width, viewport_height, &shadow_opaque_cmds);
        self.encode_scene_pass(gpu, &mut encoder, target, &ordered, opaque_count, &mut stats, clear_color);
        self.encode_postprocess_chain(gpu, &mut encoder, target, camera, viewport_width, viewport_height);

        self.draw_cmds.clear();
        self.instance_staging.clear();

        (encoder.finish(), stats)
    }

    // ── Private encode sub-methods ──

    /// Resize depth, HDR, and SSAO textures when the viewport dimensions change.
    fn resize_if_needed(&mut self, gpu: &GpuContext, viewport_width: u32, viewport_height: u32) {
        // Ensure depth texture matches viewport.
        if viewport_width != self.depth_width || viewport_height != self.depth_height {
            let (tex, view) = create_depth_texture(&gpu.device, viewport_width, viewport_height, self.sample_count);
            self.depth_texture = tex;
            self.depth_view = view;
            if self.sample_count > 1 {
                let (st, sv) = create_depth_texture(&gpu.device, viewport_width, viewport_height, 1);
                self.depth_sample_texture = Some(st);
                self.depth_sample_view = sv;
            } else {
                self.depth_sample_texture = None;
                self.depth_sample_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            }
            self.depth_width = viewport_width;
            self.depth_height = viewport_height;

            // Rebuild depth resolve bind group with new MSAA depth view.
            if let Some(resolve) = &mut self.depth_resolve_pass {
                resolve.rebuild_bind_group(&gpu.device, &self.depth_view);
            }

            // Resize post-process passes that depend on viewport dimensions.
            if let Some(ssao) = &mut self.ssao_pass {
                ssao.resize(&gpu.device, viewport_width, viewport_height);
            }
        }

        // Resize offscreen HDR target if needed.
        if let Some(pp) = &mut self.postprocess {
            if viewport_width != pp.width || viewport_height != pp.height {
                let (tex, cv, sv, msaa_tex, msaa_v) =
                    create_hdr_texture(&gpu.device, viewport_width, viewport_height, self.sample_count);
                pp.color_texture = tex;
                pp.color_view = cv;
                pp.sample_view = sv;
                pp.msaa_color_texture = msaa_tex;
                pp.msaa_color_view = msaa_v;
                pp.bloom_pass.resize(&gpu.device, viewport_width, viewport_height, &pp.sample_view);
                pp.width = viewport_width;
                pp.height = viewport_height;
            }
        }
    }

    /// Clamp instance count, grow the instance buffer if needed, and upload staging data.
    fn upload_instances(&mut self, gpu: &GpuContext) {
        // Clamp total instances.
        let total_instances = (self.instance_staging.len() as u32).min(MAX_INSTANCES);
        if (self.instance_staging.len() as u32) > MAX_INSTANCES {
            tracing::warn!(
                "instance count {} exceeds limit {MAX_INSTANCES}, truncating",
                self.instance_staging.len()
            );
            self.instance_staging.truncate(MAX_INSTANCES as usize);
        }

        // Grow instance buffer if needed.
        if total_instances as u64 > self.instance_capacity {
            let new_cap = (total_instances as u64)
                .next_power_of_two()
                .max(INITIAL_INSTANCE_CAPACITY);
            self.instance_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("esox_3d_instance_buffer"),
                size: new_cap * size_of::<InstanceData>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
            tracing::debug!("grew 3D instance buffer to {new_cap} instances");
        }

        // Upload instances.
        if !self.instance_staging.is_empty() {
            gpu.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.instance_staging),
            );
        }
    }

    /// Build and upload the scene `Uniforms` and `LightUniforms` to the GPU.
    #[allow(clippy::too_many_arguments)]
    fn upload_scene_uniforms(
        &self,
        gpu: &GpuContext,
        camera: &Camera,
        viewport_width: u32,
        viewport_height: u32,
        elapsed: f32,
        delta: f32,
    ) {
        // Upload scene uniforms.
        let aspect = viewport_width as f32 / viewport_height.max(1) as f32;
        let vp = camera.view_projection(aspect);
        let uniforms = Uniforms {
            view_projection: vp.to_cols_array_2d(),
            camera_position: [camera.position.x, camera.position.y, camera.position.z, 0.0],
            viewport: [
                viewport_width as f32,
                viewport_height as f32,
                1.0 / viewport_width.max(1) as f32,
                1.0 / viewport_height.max(1) as f32,
            ],
            time: [elapsed, delta, 0.0, 0.0],
            camera_forward: {
                let fwd = camera.forward();
                [fwd.x, fwd.y, fwd.z, 0.0]
            },
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Upload light uniforms.
        let light_uniforms = self.light_env.to_uniforms();
        gpu.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::bytes_of(&light_uniforms));
    }

    /// Pre-cull: collect all opaque mega-buffer draw data for shadow passes.
    ///
    /// Point/spot light shadow maps must render ALL opaque geometry, not just
    /// what the camera can see, otherwise shadows disappear when the light
    /// source leaves the camera frustum.
    fn collect_shadow_opaque_cmds(&self) -> Vec<(u32, u32, u32)> {
        self.draw_cmds
            .iter()
            .filter(|cmd| {
                let mesh_idx = cmd.mesh.0 as usize;
                if (cmd.mesh.0 & SKINNED_MESH_BIT) != 0 {
                    return false;
                }
                if mesh_idx >= self.mesh_regions.len() {
                    return false;
                }
                let mat_idx = cmd.material.0 as usize;
                if mat_idx >= self.materials.len() {
                    return false;
                }
                matches!(self.materials[mat_idx].pipeline_key.blend_mode, BlendMode3D::Opaque)
            })
            .map(|cmd| (cmd.mesh.0, cmd.instance_offset, cmd.instance_count))
            .collect()
    }

    /// Frustum culling — BVH-accelerated for large scenes, linear for smaller ones.
    fn frustum_cull(&mut self, vp: &glam::Mat4, stats: &mut BatchStats3D) {
        let frustum = Frustum::from_view_projection(vp);

        if self.draw_cmds.len() > self.bvh_threshold {
            // BVH-accelerated culling for large scenes.
            self.world_aabbs_scratch.clear();
            for cmd in &self.draw_cmds {
                let mesh_idx = cmd.mesh.0 as usize;
                if (cmd.mesh.0 & SKINNED_MESH_BIT) != 0 || mesh_idx >= self.mesh_regions.len() {
                    // Skinned or invalid — use a huge AABB (never culled).
                    self.world_aabbs_scratch.push(Aabb::new(
                        glam::Vec3::splat(-1e10),
                        glam::Vec3::splat(1e10),
                    ));
                    continue;
                }
                let region = &self.mesh_regions[mesh_idx];
                let inst = &self.instance_staging[cmd.instance_offset as usize];
                let model = glam::Mat4::from_cols_array_2d(&inst.model);
                self.world_aabbs_scratch.push(region.aabb.transformed(&model));
            }

            let bvh = Bvh::build(&self.world_aabbs_scratch);
            let mut visible_indices = Vec::new();
            bvh.query_frustum(&frustum, &mut visible_indices);
            visible_indices.sort_unstable();

            let total_before = self.draw_cmds.len() as u32;
            let visible_set: std::collections::HashSet<u32> = visible_indices.into_iter().collect();
            let mut write = 0;
            for read in 0..self.draw_cmds.len() {
                if visible_set.contains(&(read as u32)) {
                    self.draw_cmds.swap(write, read);
                    write += 1;
                }
            }
            self.draw_cmds.truncate(write);
            stats.culled_draws = total_before - write as u32;
        } else {
            // Linear frustum culling for smaller scenes.
            let pre_cull_count = self.draw_cmds.len() as u32;
            self.draw_cmds.retain(|cmd| {
                let mesh_idx = cmd.mesh.0 as usize;
                if (cmd.mesh.0 & SKINNED_MESH_BIT) != 0 {
                    return true; // Skinned meshes always pass.
                }
                if mesh_idx >= self.mesh_regions.len() {
                    return false;
                }
                let region = &self.mesh_regions[mesh_idx];
                let inst = &self.instance_staging[cmd.instance_offset as usize];
                let model = glam::Mat4::from_cols_array_2d(&inst.model);
                let world_aabb = region.aabb.transformed(&model);
                frustum.test_aabb_visible(&world_aabb)
            });
            stats.culled_draws = pre_cull_count - self.draw_cmds.len() as u32;
        }
    }

    /// Partition draw commands into opaque/transparent, sort each set, and return
    /// the ordered index list plus the opaque boundary.
    fn partition_and_sort(&self, camera: &Camera) -> (Vec<usize>, usize) {
        let mut opaque_cmds: Vec<usize> = Vec::new();
        let mut transparent_cmds: Vec<usize> = Vec::new();
        for (i, cmd) in self.draw_cmds.iter().enumerate() {
            let mat_idx = cmd.material.0 as usize;
            if mat_idx >= self.materials.len() {
                continue;
            }
            match self.materials[mat_idx].pipeline_key.blend_mode {
                BlendMode3D::Opaque => opaque_cmds.push(i),
                BlendMode3D::AlphaBlend | BlendMode3D::Additive => transparent_cmds.push(i),
            }
        }

        // Sort opaque by (pipeline_key, material, mesh) for maximum batching.
        let materials = &self.materials;
        let draw_cmds = &self.draw_cmds;
        opaque_cmds.sort_by(|&a, &b| {
            let ca = &draw_cmds[a];
            let cb = &draw_cmds[b];
            let key_a = &materials[ca.material.0 as usize].pipeline_key;
            let key_b = &materials[cb.material.0 as usize].pipeline_key;
            pipeline_key_sort_tuple(key_a, ca.material.0, ca.mesh.0).cmp(
                &pipeline_key_sort_tuple(key_b, cb.material.0, cb.mesh.0),
            )
        });

        // Sort transparent back-to-front by centroid distance to camera.
        let cam_pos = glam::Vec3::new(camera.position.x, camera.position.y, camera.position.z);
        let instance_staging = &self.instance_staging;
        transparent_cmds.sort_by(|&a, &b| {
            let ca = &draw_cmds[a];
            let cb = &draw_cmds[b];
            let pos_a = instance_translation(instance_staging, ca.instance_offset);
            let pos_b = instance_translation(instance_staging, cb.instance_offset);
            let dist_a = cam_pos.distance_squared(pos_a);
            let dist_b = cam_pos.distance_squared(pos_b);
            dist_b
                .partial_cmp(&dist_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build ordered draw list: opaque first, then transparent.
        let opaque_count = opaque_cmds.len();
        let ordered: Vec<usize> = opaque_cmds
            .iter()
            .chain(transparent_cmds.iter())
            .copied()
            .collect();

        (ordered, opaque_count)
    }

    /// Encode the main scene render pass with MDI/fallback draw paths and particles.
    #[allow(clippy::too_many_arguments)]
    fn encode_scene_pass(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        ordered: &[usize],
        opaque_count: usize,
        stats: &mut BatchStats3D,
        clear_color: wgpu::Color,
    ) {
        // Determine scene color target: offscreen HDR or direct to surface.
        // When MSAA is active, render into the multisampled texture and resolve
        // into the 1x texture that post-processing will sample from.
        let (scene_color_target, scene_resolve_target): (&wgpu::TextureView, Option<&wgpu::TextureView>) =
            if let Some(pp) = &self.postprocess {
                if let Some(msaa_view) = &pp.msaa_color_view {
                    (msaa_view, Some(&pp.color_view))
                } else {
                    (&pp.color_view, None)
                }
            } else {
                (target, None)
            };

        // MSAA requires StoreOp::Discard on the multisampled attachment (the
        // resolved data lives in the resolve target).
        let color_store = if scene_resolve_target.is_some() {
            wgpu::StoreOp::Discard
        } else {
            wgpu::StoreOp::Store
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("esox_3d_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_color_target,
                    resolve_target: scene_resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: color_store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_bind_group(0, Some(&self.scene_bind_group), &[]);
            pass.set_bind_group(1, Some(&self.light_bind_group), &[]);
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

            // Bind mega-buffer once for static meshes.
            pass.set_vertex_buffer(0, self.mega_buffer.vertex_buffer.slice(..));
            pass.set_index_buffer(
                self.mega_buffer.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            let mut current_is_mega = true;
            let mut current_skinned_idx: Option<usize> = None;

            let mut current_pipeline_key: Option<PipelineKey> = None;
            let mut current_material: Option<u32> = None;

            // ── Multi-draw-indirect path ──

            if self.multi_draw_indirect && !ordered.is_empty() {
                // Build indirect args grouped by (pipeline, material).
                // Each group becomes one multi_draw_indexed_indirect call.

                // Ensure indirect buffer is large enough.
                let needed = ordered.len() as u32;
                if needed > self.indirect_capacity {
                    let new_cap = (needed as u64)
                        .next_power_of_two()
                        .max(INITIAL_INDIRECT_CAPACITY as u64) as u32;
                    self.indirect_buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("esox_3d_indirect"),
                        size: new_cap as u64 * INDIRECT_ARGS_SIZE,
                        usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                    self.indirect_capacity = new_cap;
                }

                let indirect_buf = self.indirect_buffer.as_ref().unwrap();

                // Build groups: contiguous runs in `ordered` with same pipeline+material+buffer source.
                // Skinned meshes each get their own group since they use separate vertex/index buffers.
                // (pipeline_key, material_idx, start_in_indirect, count, skinned_mesh_idx or u32::MAX for mega-buffer)
                let mut groups: Vec<(PipelineKey, u32, u32, u32, u32)> = Vec::new();
                let mut indirect_args: Vec<[u32; 5]> = Vec::with_capacity(ordered.len());

                for &oi_idx in ordered {
                    let cmd = &self.draw_cmds[oi_idx];
                    let mat_idx = cmd.material.0 as usize;
                    if mat_idx >= self.materials.len() {
                        continue;
                    }
                    let key = self.materials[mat_idx].pipeline_key;
                    let is_skinned = (cmd.mesh.0 & SKINNED_MESH_BIT) != 0;

                    // Resolve index count and base vertex/index for this draw.
                    let (index_count, index_offset, base_vertex, skinned_idx) = if is_skinned {
                        let si = (cmd.mesh.0 & !SKINNED_MESH_BIT) as usize;
                        if si < self.meshes.len() {
                            (self.meshes[si].index_count, 0u32, 0i32, si as u32)
                        } else {
                            continue;
                        }
                    } else {
                        let mesh_idx = cmd.mesh.0 as usize;
                        if mesh_idx < self.mesh_regions.len() {
                            let r = &self.mesh_regions[mesh_idx];
                            (r.index_count, r.index_offset, r.vertex_offset as i32, u32::MAX)
                        } else {
                            continue;
                        }
                    };

                    let arg_idx = indirect_args.len() as u32;
                    indirect_args.push([
                        index_count,
                        cmd.instance_count,
                        index_offset,
                        base_vertex as u32,
                        cmd.instance_offset,
                    ]);

                    // Extend current group or start new one.
                    // Skinned meshes break groups because they need different vertex/index buffers.
                    let mat_key = cmd.material.0;
                    if let Some(last) = groups.last_mut() {
                        if last.0 == key && last.1 == mat_key && last.4 == skinned_idx {
                            last.3 += 1;
                            continue;
                        }
                    }
                    groups.push((key, mat_key, arg_idx, 1, skinned_idx));
                }

                // Upload indirect args.
                if !indirect_args.is_empty() {
                    gpu.queue.write_buffer(
                        indirect_buf,
                        0,
                        bytemuck::cast_slice(&indirect_args),
                    );
                }

                // Issue one multi_draw_indexed_indirect per group, rebinding buffers for skinned meshes.
                for (key, mat_key, start, count, skinned_idx) in &groups {
                    if current_pipeline_key != Some(*key) {
                        if let Some(pipeline) = self.pipeline_cache.get(key) {
                            pass.set_pipeline(pipeline);
                            current_pipeline_key = Some(*key);
                            stats.pipeline_switches += 1;
                            current_material = None;
                        } else {
                            continue;
                        }
                    }
                    if current_material != Some(*mat_key) {
                        let mat = &self.materials[*mat_key as usize];
                        pass.set_bind_group(2, Some(&mat.bind_group), &[]);
                        current_material = Some(*mat_key);
                        stats.material_switches += 1;
                    }

                    // Rebind vertex/index buffers when switching between mega-buffer and skinned meshes.
                    if *skinned_idx != u32::MAX {
                        let mesh = &self.meshes[*skinned_idx as usize];
                        if current_is_mega {
                            current_is_mega = false;
                        }
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                    } else if !current_is_mega {
                        pass.set_vertex_buffer(0, self.mega_buffer.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            self.mega_buffer.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        current_is_mega = true;
                    }

                    pass.multi_draw_indexed_indirect(
                        indirect_buf,
                        *start as u64 * INDIRECT_ARGS_SIZE,
                        *count,
                    );
                    stats.draw_calls += 1;
                }

                // Compute total instances/triangles from args.
                for arg in &indirect_args {
                    stats.total_instances += arg[1];
                    stats.total_triangles += arg[1] * (arg[0] / 3);
                }
            } else {
                // ── Fallback: individual draw_indexed calls ──

                let mut oi = 0;
                while oi < ordered.len() {
                    let cmd_idx = ordered[oi];
                    let cmd = &self.draw_cmds[cmd_idx];
                    let mat_idx = cmd.material.0 as usize;
                    if mat_idx >= self.materials.len() {
                        tracing::warn!("invalid material handle {}", cmd.material.0);
                        oi += 1;
                        continue;
                    }

                    let is_skinned = (cmd.mesh.0 & SKINNED_MESH_BIT) != 0;

                    let mat = &self.materials[mat_idx];
                    let key = mat.pipeline_key;

                    // Set pipeline if changed.
                    if current_pipeline_key != Some(key) {
                        if let Some(pipeline) = self.pipeline_cache.get(&key) {
                            pass.set_pipeline(pipeline);
                            current_pipeline_key = Some(key);
                            stats.pipeline_switches += 1;
                            current_material = None;
                        } else {
                            tracing::warn!("no pipeline for key {:?}", key);
                            oi += 1;
                            continue;
                        }
                    }

                    // Set material bind group if changed.
                    if current_material != Some(cmd.material.0) {
                        pass.set_bind_group(2, Some(&mat.bind_group), &[]);
                        current_material = Some(cmd.material.0);
                        stats.material_switches += 1;
                    }

                    if is_skinned {
                        // Skinned mesh: use individual buffers.
                        let skinned_idx = (cmd.mesh.0 & !SKINNED_MESH_BIT) as usize;
                        if skinned_idx >= self.meshes.len() {
                            oi += 1;
                            continue;
                        }
                        if current_is_mega || current_skinned_idx != Some(skinned_idx) {
                            let mesh = &self.meshes[skinned_idx];
                            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            current_is_mega = false;
                            current_skinned_idx = Some(skinned_idx);
                        }
                        let mesh = &self.meshes[skinned_idx];
                        pass.draw_indexed(
                            0..mesh.index_count,
                            0,
                            cmd.instance_offset..cmd.instance_offset + cmd.instance_count,
                        );
                        stats.draw_calls += 1;
                        stats.total_instances += cmd.instance_count;
                        stats.total_triangles += cmd.instance_count * (mesh.index_count / 3);
                        oi += 1;
                    } else {
                        // Static mesh: use mega-buffer with region offsets.
                        let mesh_idx = cmd.mesh.0 as usize;
                        if mesh_idx >= self.mesh_regions.len() {
                            tracing::warn!("invalid mesh handle {}", cmd.mesh.0);
                            oi += 1;
                            continue;
                        }

                        // Rebind mega-buffer if we were on skinned.
                        if !current_is_mega {
                            pass.set_vertex_buffer(0, self.mega_buffer.vertex_buffer.slice(..));
                            pass.set_index_buffer(
                                self.mega_buffer.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            current_is_mega = true;
                            current_skinned_idx = None;
                        }

                        let r = &self.mesh_regions[mesh_idx];

                        // Try to merge adjacent opaque draws with same material + mesh.
                        let merged_offset = cmd.instance_offset;
                        let mut merged_count = cmd.instance_count;
                        let is_opaque = oi < opaque_count;
                        if is_opaque {
                            let mut j = oi + 1;
                            while j < opaque_count {
                                let next = &self.draw_cmds[ordered[j]];
                                if next.material.0 == cmd.material.0
                                    && next.mesh.0 == cmd.mesh.0
                                    && next.instance_offset == merged_offset + merged_count
                                {
                                    merged_count += next.instance_count;
                                    j += 1;
                                } else {
                                    break;
                                }
                            }
                            oi = j;
                        } else {
                            oi += 1;
                        }

                        pass.draw_indexed(
                            r.index_offset..r.index_offset + r.index_count,
                            r.vertex_offset as i32,
                            merged_offset..merged_offset + merged_count,
                        );
                        stats.draw_calls += 1;
                        stats.total_instances += merged_count;
                        stats.total_triangles += merged_count * (r.index_count / 3);
                    }
                }
            }

            // ── Particle indirect draws ──
            if let Some(quad_handle) = self.particle_quad_mesh {
                let quad_idx = quad_handle.0 as usize;
                if quad_idx < self.mesh_regions.len() {
                    // Rebind mega-buffer for particle quad if needed.
                    if !current_is_mega {
                        pass.set_vertex_buffer(0, self.mega_buffer.vertex_buffer.slice(..));
                        pass.set_index_buffer(
                            self.mega_buffer.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                    }

                    for pcmd in &self.particle_draw_cmds {
                        let pool_idx = pcmd.pool.0 as usize;
                        if pool_idx >= self.particle_pools.len() {
                            continue;
                        }
                        let pool = &self.particle_pools[pool_idx];

                        // Set material pipeline + bind group.
                        let mat_idx = pcmd.material.0 as usize;
                        if mat_idx >= self.materials.len() {
                            continue;
                        }
                        let mat = &self.materials[mat_idx];
                        let key = mat.pipeline_key;
                        if current_pipeline_key != Some(key) {
                            if let Some(pipeline) = self.pipeline_cache.get(&key) {
                                pass.set_pipeline(pipeline);
                                current_pipeline_key = Some(key);
                                stats.pipeline_switches += 1;
                            } else {
                                continue;
                            }
                        }
                        pass.set_bind_group(2, Some(&mat.bind_group), &[]);

                        // Bind particle instance output as VB slot 1.
                        pass.set_vertex_buffer(1, pool.instance_output.slice(..));

                        // Indirect draw: instance_count was set by finalize_main.
                        pass.draw_indexed_indirect(
                            &pool.indirect_args_buffer,
                            0,
                        );
                        stats.draw_calls += 1;
                    }
                }
            }
            self.particle_draw_cmds.clear();
        }
    }

}

