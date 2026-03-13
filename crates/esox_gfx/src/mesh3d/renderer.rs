//! 3D mesh renderer — pipeline, depth buffer, frame encoding.

use std::collections::HashMap;

use crate::bloom::BloomPass;
use crate::pipeline::GpuContext;

use super::bounds::{Aabb, Frustum};
use super::bvh::Bvh;
use super::camera::Camera;
use super::ibl::IblState;
use super::instance::InstanceData;
use super::light::{LightEnvironment, LightUniforms, SpotLightGpu, MAX_SPOT_LIGHTS};
use super::lod::LodGroup;
use super::material::{
    BlendMode3D, Material, MaterialDescriptor, MaterialHandle, MaterialType, MaterialUniforms,
    PipelineKey, create_pipeline, create_pipeline_with_shader,
};
use super::mesh::{MegaBuffer, Mesh, MeshData, MeshHandle, MeshRegion};
use super::motion_blur::MotionBlurPass;
use super::sdf_pass::{SdfEffectDescriptor, SdfEffectHandle, SdfPass};
use super::shadow::{ShadowConfig, ShadowPass, ShadowUniforms};
use super::skinning::{SkinningPipeline, SkinnedMesh};
use super::ssao::SsaoPass;
use super::texture::{Texture3D, TextureHandle};

// ── Uniforms ──

/// GPU uniform data: view-projection matrix, camera position, viewport, time.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Uniforms {
    /// Combined view-projection matrix (column-major).
    view_projection: [[f32; 4]; 4],
    /// Camera world-space position (w unused).
    camera_position: [f32; 4],
    /// Viewport: [width, height, 1/width, 1/height].
    viewport: [f32; 4],
    /// Time: [elapsed_seconds, delta_seconds, 0, 0].
    time: [f32; 4],
}

// ── Draw command ──

/// A queued draw command (mesh + material + instances).
struct DrawCmd {
    mesh: MeshHandle,
    material: MaterialHandle,
    instance_offset: u32,
    instance_count: u32,
}

// ── Batch stats ──

/// Statistics from a single frame's draw batching.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchStats3D {
    /// Number of draw calls issued.
    pub draw_calls: u32,
    /// Number of pipeline switches.
    pub pipeline_switches: u32,
    /// Number of material bind group switches.
    pub material_switches: u32,
    /// Total instances rendered.
    pub total_instances: u32,
    /// Total triangles rendered (instances * triangles-per-mesh).
    pub total_triangles: u32,
    /// Draw commands culled by frustum.
    pub culled_draws: u32,
    /// Instances culled by frustum.
    pub culled_instances: u32,
}

// ── Constants ──

/// Initial instance buffer capacity.
const INITIAL_INSTANCE_CAPACITY: u64 = 4096;

/// Maximum instances per frame (safety limit).
const MAX_INSTANCES: u32 = 500_000;

/// Depth texture format.
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// HDR format for offscreen rendering (when post-processing is enabled).
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Bit flag to distinguish skinned (standalone-buffer) mesh handles from mega-buffer handles.
const SKINNED_MESH_BIT: u32 = 0x8000_0000;

/// Size of `DrawIndexedIndirectArgs` (5 × u32 = 20 bytes).
const INDIRECT_ARGS_SIZE: u64 = 20;

/// Initial indirect buffer capacity.
const INITIAL_INDIRECT_CAPACITY: u32 = 1024;

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
    /// Enable motion blur.
    pub motion_blur_enabled: bool,
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
            motion_blur_enabled: false,
        }
    }
}

/// GPU params for the composite pass (16 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CompositeParams3D {
    bloom_intensity: f32,
    tone_map: f32,
    ssao_enabled: f32,
    _pad: f32,
}

/// Internal state for the 3D post-process pipeline.
struct PostProcess3D {
    /// Offscreen HDR color texture (1x, used as resolve target / sampling source).
    #[allow(dead_code)]
    color_texture: wgpu::Texture,
    /// View for rendering into (RENDER_ATTACHMENT) — 1x when no MSAA, unused as
    /// direct render target when MSAA is active (resolve writes here instead).
    color_view: wgpu::TextureView,
    /// View for sampling (TEXTURE_BINDING) — always 1x.
    sample_view: wgpu::TextureView,
    /// MSAA render texture (sample_count > 1). Render pass writes here and
    /// hardware-resolves into `color_view`.
    #[allow(dead_code)]
    msaa_color_texture: Option<wgpu::Texture>,
    msaa_color_view: Option<wgpu::TextureView>,
    /// Bloom pass (reusing the 2D dual-Kawase implementation).
    bloom_pass: BloomPass,
    /// Bloom downsample pipeline.
    bloom_down_pipeline: wgpu::RenderPipeline,
    /// Bloom upsample pipeline.
    bloom_up_pipeline: wgpu::RenderPipeline,
    /// Fallback black texture for when bloom is disabled.
    #[allow(dead_code)]
    bloom_black_texture: wgpu::Texture,
    bloom_black_view: wgpu::TextureView,
    /// Composite pipeline (fullscreen triangle: scene + bloom + SSAO -> surface).
    composite_pipeline: wgpu::RenderPipeline,
    /// Composite bind group layout.
    composite_bind_group_layout: wgpu::BindGroupLayout,
    /// Composite params buffer.
    params_buffer: wgpu::Buffer,
    /// Linear sampler for sampling HDR textures.
    linear_sampler: wgpu::Sampler,
    /// Current config.
    config: PostProcess3DConfig,
    /// Current offscreen dimensions.
    width: u32,
    height: u32,
}

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
    light_env: LightEnvironment,

    // Pipeline cache: key -> pipeline.
    pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
    shader_modules: HashMap<MaterialType, wgpu::ShaderModule>,
    surface_format: wgpu::TextureFormat,

    // Materials.
    materials: Vec<Material>,

    // Textures.
    textures: Vec<Texture3D>,
    fallback_albedo: Texture3D,
    fallback_normal: Texture3D,
    fallback_mr: Texture3D,
    shared_sampler: wgpu::Sampler,

    // Instancing.
    instance_buffer: wgpu::Buffer,
    instance_capacity: u64,

    // Depth — render target (MSAA when sample_count > 1).
    #[allow(dead_code)]
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    // Depth — 1x sampling view for post-processing (SSAO, SDF, motion blur).
    // When sample_count == 1 this is the same texture/view as above.
    #[allow(dead_code)]
    depth_sample_texture: Option<wgpu::Texture>,
    depth_sample_view: wgpu::TextureView,
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

    // Per-frame state.
    draw_cmds: Vec<DrawCmd>,
    instance_staging: Vec<InstanceData>,

    // LOD scratch buffers (reused each frame to avoid allocation).
    lod_staging: Vec<InstanceData>,
    #[allow(dead_code)]
    lod_offsets: Vec<(u32, u32)>,

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
    postprocess: Option<PostProcess3D>,

    // Shadow maps.
    shadow_pass: Option<ShadowPass>,
    shadow_uniform_buffer: wgpu::Buffer,
    comparison_sampler: wgpu::Sampler,
    fallback_shadow_depth_view: wgpu::TextureView,
    #[allow(dead_code)]
    fallback_shadow_depth_texture: wgpu::Texture,

    // SSAO.
    ssao_pass: Option<SsaoPass>,
    fallback_ssao_view: wgpu::TextureView,
    #[allow(dead_code)]
    fallback_ssao_texture: wgpu::Texture,

    // IBL (image-based lighting).
    ibl_state: IblState,
    ibl_sampler: wgpu::Sampler,

    // Motion blur.
    motion_blur_pass: Option<MotionBlurPass>,
    prev_view_projection: Option<glam::Mat4>,

    // SDF effects.
    sdf_pass: Option<SdfPass>,

    // MSAA sample count (1 = off, 4 = 4x).
    sample_count: u32,
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

        let shadow_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_shadow_uniforms"),
            size: size_of::<ShadowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_capacity = INITIAL_INSTANCE_CAPACITY;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esox_3d_instance_buffer"),
            size: instance_capacity * size_of::<InstanceData>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Fallback resources ──

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
                depth_or_array_layers: super::shadow::MAX_SHADOW_CASCADES as u32,
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
                    resource: shadow_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&fallback_shadow_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&comparison_sampler),
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

        // ── Shader modules ──

        let shader_modules = compile_shader_modules(device);

        // ── Pipeline cache — eagerly create 3 opaque pipelines ──

        let mut pipeline_cache = HashMap::new();
        let format = gpu.config.format;
        for &mat_type in &[MaterialType::Unlit, MaterialType::Lit, MaterialType::PBR] {
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
        }];

        // ── Depth texture ──

        let (depth_texture, depth_view) =
            create_depth_texture(device, gpu.config.width, gpu.config.height, gpu.sample_count);
        // Separate 1x depth for sampling by SSAO/SDF/motion blur when MSAA is active.
        let (depth_sample_texture, depth_sample_view) = if gpu.sample_count > 1 {
            let (t, v) = create_depth_texture(device, gpu.config.width, gpu.config.height, 1);
            (Some(t), v)
        } else {
            let v = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            (None, v)
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
            draw_cmds: Vec::new(),
            instance_staging: Vec::new(),
            lod_staging: Vec::new(),
            lod_offsets: Vec::new(),
            bvh_threshold: 4096,
            world_aabbs_scratch: Vec::new(),
            multi_draw_indirect,
            indirect_buffer: None,
            indirect_capacity: 0,
            postprocess: None,
            shadow_pass: None,
            shadow_uniform_buffer,
            comparison_sampler,
            fallback_shadow_depth_view,
            fallback_shadow_depth_texture,
            ssao_pass: None,
            fallback_ssao_view,
            fallback_ssao_texture,
            ibl_state,
            ibl_sampler,
            motion_blur_pass: None,
            prev_view_projection: None,
            sdf_pass: None,
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
        });
        Ok(handle)
    }

    /// Set the light environment for subsequent frames.
    pub fn set_lights(&mut self, env: &LightEnvironment) {
        self.light_env = env.clone();
    }

    /// Enable the post-processing pipeline (offscreen HDR + bloom + tone mapping).
    ///
    /// When enabled, the scene renders to an offscreen `Rgba16Float` texture and
    /// a composite pass blits the result to the surface with bloom and tone mapping.
    /// When disabled (default), rendering goes directly to the surface.
    pub fn enable_postprocess(&mut self, gpu: &GpuContext) {
        if self.postprocess.is_some() {
            return;
        }
        if self.sample_count > 1 {
            tracing::warn!(
                "MSAA ({}x) is active but no depth resolve pass exists — \
                 depth-dependent post-processing (SSAO, motion blur) will \
                 read a blank depth buffer",
                self.sample_count,
            );
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

        let create_bloom_pipeline = |shader: &wgpu::ShaderModule, label: &str| -> wgpu::RenderPipeline {
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
            })
        };
        let bloom_down_pipeline = create_bloom_pipeline(&down_shader, "esox_3d_bloom_down");
        let bloom_up_pipeline = create_bloom_pipeline(&up_shader, "esox_3d_bloom_up");

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

    /// Rebuild all cached material pipelines for the current `surface_format`.
    fn rebuild_pipeline_cache(&mut self, device: &wgpu::Device) {
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

    /// Set the post-process configuration.
    pub fn set_postprocess(&mut self, config: PostProcess3DConfig) {
        if let Some(pp) = &mut self.postprocess {
            pp.config = config;
        }
    }

    /// Enable cascaded shadow maps.
    pub fn enable_shadows(&mut self, gpu: &GpuContext) {
        if self.shadow_pass.is_some() {
            return;
        }
        let shadow_pass = ShadowPass::new(&gpu.device);

        // Rebuild scene bind group with real shadow depth texture.
        self.scene_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("esox_3d_scene_bg"),
            layout: &self.scene_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.shadow_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_pass.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.comparison_sampler),
                },
            ],
        });

        self.shadow_pass = Some(shadow_pass);
    }

    /// Set the shadow configuration.
    pub fn set_shadow_config(&mut self, config: ShadowConfig) {
        if let Some(sp) = &mut self.shadow_pass {
            sp.config = config;
        }
    }

    /// Enable SSAO (requires post-processing to be enabled).
    pub fn enable_ssao(&mut self, gpu: &GpuContext) {
        if self.ssao_pass.is_some() {
            return;
        }
        if self.sample_count > 1 {
            tracing::warn!(
                "enabling SSAO with MSAA ({}x) — depth buffer is unresolved, \
                 SSAO will produce incorrect results",
                self.sample_count,
            );
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

    /// Enable motion blur post-processing.
    pub fn enable_motion_blur(&mut self, gpu: &GpuContext) {
        if self.motion_blur_pass.is_some() {
            return;
        }
        if self.sample_count > 1 {
            tracing::warn!(
                "enabling motion blur with MSAA ({}x) — depth buffer is \
                 unresolved, motion blur will produce incorrect results",
                self.sample_count,
            );
        }
        let w = gpu.config.width.max(1);
        let h = gpu.config.height.max(1);
        self.motion_blur_pass = Some(MotionBlurPass::new(&gpu.device, w, h));
    }

    /// Set the motion blur configuration.
    pub fn set_motion_blur_config(&mut self, config: super::motion_blur::MotionBlurConfig) {
        if let Some(mb) = &mut self.motion_blur_pass {
            mb.config = config;
        }
    }

    /// Register an SDF effect. Returns a handle to enable/disable it.
    pub fn register_sdf_effect(
        &mut self,
        gpu: &GpuContext,
        desc: &SdfEffectDescriptor,
    ) -> Result<SdfEffectHandle, String> {
        let sdf_pass = self.sdf_pass.get_or_insert_with(|| SdfPass::new(&gpu.device));
        let depth_view = &self.depth_sample_view;
        sdf_pass.register_effect(&gpu.device, &gpu.queue, depth_view, desc)
    }

    /// Enable or disable an SDF effect.
    pub fn set_sdf_enabled(&mut self, handle: SdfEffectHandle, enabled: bool) {
        if let Some(sdf) = &mut self.sdf_pass {
            sdf.set_enabled(handle, enabled);
        }
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

    /// Queue LOD-selected draw commands.
    ///
    /// Buckets instances by selected mesh handle, issues one `draw_with_material`
    /// per non-empty bucket. Uses scratch buffers to avoid steady-state allocation.
    pub fn draw_lod(
        &mut self,
        lod: &LodGroup,
        material: MaterialHandle,
        instances: &[InstanceData],
        camera_pos: glam::Vec3,
    ) {
        if instances.is_empty() {
            return;
        }

        // Assign each instance to its LOD mesh handle.
        // Collect unique meshes seen (typically 3-4 levels).
        let mut seen_meshes: Vec<MeshHandle> = Vec::with_capacity(lod.level_count());
        let mut assignments: Vec<MeshHandle> = Vec::with_capacity(instances.len());
        for inst in instances {
            let pos = glam::Vec3::new(inst.model[3][0], inst.model[3][1], inst.model[3][2]);
            let dist = camera_pos.distance(pos);
            let mesh = lod.select(dist);
            assignments.push(mesh);
            if !seen_meshes.contains(&mesh) {
                seen_meshes.push(mesh);
            }
        }

        // For each unique mesh, batch its instances into a single draw command.
        for mesh in &seen_meshes {
            self.lod_staging.clear();
            for (inst, assigned) in instances.iter().zip(assignments.iter()) {
                if assigned == mesh {
                    self.lod_staging.push(*inst);
                }
            }
            if !self.lod_staging.is_empty() {
                let offset = self.instance_staging.len() as u32;
                let count = self.lod_staging.len() as u32;
                self.instance_staging.extend_from_slice(&self.lod_staging);
                self.draw_cmds.push(DrawCmd {
                    mesh: *mesh,
                    material,
                    instance_offset: offset,
                    instance_count: count,
                });
            }
        }
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
    /// - Optional offscreen HDR rendering with bloom, SSAO, motion blur, SDF effects
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

            // Resize post-process passes that depend on viewport dimensions.
            if let Some(ssao) = &mut self.ssao_pass {
                ssao.resize(&gpu.device, viewport_width, viewport_height);
            }
            if let Some(mb) = &mut self.motion_blur_pass {
                mb.resize(&gpu.device, viewport_width, viewport_height);
            }
            if let Some(sdf) = &mut self.sdf_pass {
                sdf.rebuild_bind_groups(&gpu.device, &self.depth_sample_view);
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
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Upload light uniforms.
        let light_uniforms = self.light_env.to_uniforms();
        gpu.queue
            .write_buffer(&self.light_buffer, 0, bytemuck::bytes_of(&light_uniforms));

        let mut stats = BatchStats3D::default();

        // ── Frustum culling ──
        let frustum = Frustum::from_view_projection(&vp);

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

        // ── Partition and sort ──

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
        let ordered: Vec<usize> = opaque_cmds
            .iter()
            .chain(transparent_cmds.iter())
            .copied()
            .collect();
        let opaque_count = opaque_cmds.len();

        // ── Encode render pass ──

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("esox_3d_encoder"),
            });

        // ── Shadow pass (before scene) ──
        if let Some(shadow_pass) = &self.shadow_pass {
            if shadow_pass.config.enabled {
                let aspect = viewport_width as f32 / viewport_height.max(1) as f32;
                let shadow_uniforms = shadow_pass.update_cascades(
                    &gpu.queue,
                    camera,
                    self.light_env.directional.direction,
                    aspect,
                );
                gpu.queue.write_buffer(
                    &self.shadow_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&shadow_uniforms),
                );

                // Render depth from light's perspective for each cascade.
                let cascade_count = shadow_pass.config.cascade_count.clamp(2, super::shadow::MAX_SHADOW_CASCADES);
                for cascade in 0..cascade_count {
                    let mut pass = shadow_pass.begin_cascade_pass(&mut encoder, cascade);

                    // Bind mega-buffer and instance buffer, draw all opaque meshes.
                    pass.set_vertex_buffer(0, self.mega_buffer.vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        self.mega_buffer.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

                    for &oi_idx in &opaque_cmds {
                        let cmd = &self.draw_cmds[oi_idx];
                        let is_skinned = (cmd.mesh.0 & SKINNED_MESH_BIT) != 0;
                        if is_skinned {
                            continue; // Skip skinned meshes in shadow pass for simplicity.
                        }
                        let mesh_idx = cmd.mesh.0 as usize;
                        if mesh_idx >= self.mesh_regions.len() {
                            continue;
                        }
                        let r = &self.mesh_regions[mesh_idx];
                        pass.draw_indexed(
                            r.index_offset..r.index_offset + r.index_count,
                            r.vertex_offset as i32,
                            cmd.instance_offset..cmd.instance_offset + cmd.instance_count,
                        );
                    }
                }
            } else {
                // Write zeroed shadow uniforms (shadow_config.w = 0 -> shadow_factor returns 1).
                let zeroed = ShadowUniforms {
                    light_vp: [[[0.0; 4]; 4]; super::shadow::MAX_SHADOW_CASCADES],
                    splits_count: [0.0; 4],
                    shadow_config: [0.0, 0.0, 0.0, 0.0],
                };
                gpu.queue.write_buffer(
                    &self.shadow_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&zeroed),
                );
            }
        } else {
            // No shadow pass: write zeroed shadow uniforms.
            let zeroed = ShadowUniforms {
                light_vp: [[[0.0; 4]; 4]; super::shadow::MAX_SHADOW_CASCADES],
                splits_count: [0.0; 4],
                shadow_config: [0.0, 0.0, 0.0, 0.0],
            };
            gpu.queue.write_buffer(
                &self.shadow_uniform_buffer,
                0,
                bytemuck::bytes_of(&zeroed),
            );
        }

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

                // Build groups: contiguous runs in `ordered` with same pipeline+material.
                // Each group: (pipeline_key, material_idx, start_in_indirect, count, is_skinned_group).
                let mut groups: Vec<(PipelineKey, u32, u32, u32)> = Vec::new();
                let mut indirect_args: Vec<[u32; 5]> = Vec::with_capacity(ordered.len());

                for &oi_idx in &ordered {
                    let cmd = &self.draw_cmds[oi_idx];
                    let mat_idx = cmd.material.0 as usize;
                    if mat_idx >= self.materials.len() {
                        continue;
                    }
                    let key = self.materials[mat_idx].pipeline_key;
                    let is_skinned = (cmd.mesh.0 & SKINNED_MESH_BIT) != 0;

                    // Resolve index count and base vertex/index for this draw.
                    let (index_count, index_offset, base_vertex) = if is_skinned {
                        let skinned_idx = (cmd.mesh.0 & !SKINNED_MESH_BIT) as usize;
                        if skinned_idx < self.meshes.len() {
                            (self.meshes[skinned_idx].index_count, 0u32, 0i32)
                        } else {
                            continue;
                        }
                    } else {
                        let mesh_idx = cmd.mesh.0 as usize;
                        if mesh_idx < self.mesh_regions.len() {
                            let r = &self.mesh_regions[mesh_idx];
                            (r.index_count, r.index_offset, r.vertex_offset as i32)
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
                    let mat_key = cmd.material.0;
                    if let Some(last) = groups.last_mut() {
                        if last.0 == key && last.1 == mat_key {
                            last.3 += 1;
                            continue;
                        }
                    }
                    groups.push((key, mat_key, arg_idx, 1));
                }

                // Upload indirect args.
                if !indirect_args.is_empty() {
                    gpu.queue.write_buffer(
                        indirect_buf,
                        0,
                        bytemuck::cast_slice(&indirect_args),
                    );
                }

                // Issue one multi_draw_indexed_indirect per group.
                for (key, mat_key, start, count) in &groups {
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
                        if current_is_mega {
                            let mesh = &self.meshes[skinned_idx];
                            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            current_is_mega = false;
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
        }

        // ── Post-processing chain ──

        // SDF effects (render onto the scene color target with LoadOp::Load).
        if let Some(sdf) = &self.sdf_pass {
            let inv_vp = vp.inverse();
            sdf.encode(
                &mut encoder,
                &gpu.queue,
                scene_color_target,
                &self.depth_sample_view,
                inv_vp,
                camera.position,
                [
                    viewport_width as f32,
                    viewport_height as f32,
                    1.0 / viewport_width.max(1) as f32,
                    1.0 / viewport_height.max(1) as f32,
                ],
                elapsed,
                delta,
            );
        }

        // SSAO (reads depth buffer, writes occlusion texture).
        if let Some(ssao) = &mut self.ssao_pass {
            let proj = glam::Mat4::perspective_rh(
                camera.fov_y,
                viewport_width as f32 / viewport_height.max(1) as f32,
                camera.near,
                camera.far,
            );
            ssao.encode(&gpu.device, &mut encoder, &gpu.queue, &self.depth_sample_view, proj);
        }

        // Motion blur (reads depth + scene HDR, writes blurred HDR).
        if let Some(mb) = &mut self.motion_blur_pass {
            if let (Some(pp), Some(prev_vp)) = (&self.postprocess, &self.prev_view_projection) {
                let inv_vp = vp.inverse();
                mb.rebuild_bind_groups(&gpu.device, &self.depth_sample_view, &pp.sample_view);
                mb.encode(
                    &mut encoder,
                    &gpu.queue,
                    &self.depth_sample_view,
                    &pp.sample_view,
                    inv_vp,
                    *prev_vp,
                    viewport_width,
                    viewport_height,
                );
            }
        }
        self.prev_view_projection = Some(vp);

        // Bloom + composite (when post-processing is enabled).
        if let Some(pp) = &mut self.postprocess {
            let config = pp.config;

            // Determine which scene texture to read from (motion blur output or raw scene).
            let scene_source = if config.motion_blur_enabled {
                if let Some(mb) = &self.motion_blur_pass {
                    mb.result_view()
                } else {
                    &pp.sample_view
                }
            } else {
                &pp.sample_view
            };

            // Run bloom on the scene HDR texture.
            if config.bloom_enabled {
                pp.bloom_pass.encode(&mut encoder, &gpu.queue, &pp.bloom_down_pipeline, &pp.bloom_up_pipeline, config.bloom_threshold, config.bloom_soft_knee);
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
                _pad: 0.0,
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

        // Clear draw state for next frame.
        self.draw_cmds.clear();
        self.instance_staging.clear();

        (encoder.finish(), stats)
    }
}

/// Sort key for draw commands: (pipeline key hash, material index, mesh index).
fn pipeline_key_sort_tuple(
    key: &PipelineKey,
    material_idx: u32,
    mesh_idx: u32,
) -> (u8, u8, bool, u32, u32) {
    let mt = match key.material_type {
        MaterialType::Unlit => 0,
        MaterialType::Lit => 1,
        MaterialType::PBR => 2,
    };
    let bm = match key.blend_mode {
        BlendMode3D::Opaque => 0,
        BlendMode3D::AlphaBlend => 1,
        BlendMode3D::Additive => 2,
    };
    (mt, bm, key.depth_write, material_idx, mesh_idx)
}

/// Extract the translation (column 3) from the first instance of a draw command.
fn instance_translation(staging: &[InstanceData], offset: u32) -> glam::Vec3 {
    let inst = &staging[offset as usize];
    glam::Vec3::new(inst.model[3][0], inst.model[3][1], inst.model[3][2])
}

// ── Depth texture helper ──

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    // MSAA depth textures cannot have TEXTURE_BINDING usage.
    let usage = if sample_count > 1 {
        wgpu::TextureUsages::RENDER_ATTACHMENT
    } else {
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
    };
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
fn create_hdr_texture(
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

// ── Shader sources ──

fn compile_shader_modules(device: &wgpu::Device) -> HashMap<MaterialType, wgpu::ShaderModule> {
    let mut modules = HashMap::new();

    let unlit_src = format!("{SHADER_PREAMBLE}\n{FS_UNLIT}");
    let lit_src = format!("{SHADER_PREAMBLE}\n{FS_LIT}");
    let pbr_src = format!("{SHADER_PREAMBLE}\n{FS_PBR}");

    modules.insert(
        MaterialType::Unlit,
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_shader_unlit"),
            source: wgpu::ShaderSource::Wgsl(unlit_src.into()),
        }),
    );
    modules.insert(
        MaterialType::Lit,
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_shader_lit"),
            source: wgpu::ShaderSource::Wgsl(lit_src.into()),
        }),
    );
    modules.insert(
        MaterialType::PBR,
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("esox_3d_shader_pbr"),
            source: wgpu::ShaderSource::Wgsl(pbr_src.into()),
        }),
    );

    modules
}

/// Shared shader preamble: struct definitions, bind groups, vertex shader.
const SHADER_PREAMBLE: &str = r"
struct Uniforms {
    view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,
    time: vec4<f32>,
}

struct ShadowUniforms {
    light_vp: array<mat4x4<f32>, 4>,
    splits_count: vec4<f32>,
    shadow_config: vec4<f32>,
}

struct PointLightGpu {
    position_range: vec4<f32>,
    color_intensity: vec4<f32>,
}

struct SpotLightGpu {
    position_range: vec4<f32>,
    direction_inner: vec4<f32>,
    color_intensity: vec4<f32>,
    outer_pad: vec4<f32>,
}

struct LightUniforms {
    ambient: vec4<f32>,
    directional_dir_intensity: vec4<f32>,
    directional_color_count: vec4<f32>,
    spot_count_pad: vec4<f32>,
    point_lights: array<PointLightGpu, 8>,
    spot_lights: array<SpotLightGpu, 4>,
}

struct MaterialUniforms {
    albedo: vec4<f32>,
    emissive_metallic: vec4<f32>,
    roughness_opacity_flags: vec4<f32>,
    texture_flags: vec4<f32>,
    extra: vec4<f32>,
}

// Group 0: Scene uniforms + shadow.
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(2) var shadow_depth: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;

// Group 1: Lights + IBL.
@group(1) @binding(0) var<uniform> lights: LightUniforms;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(3) var brdf_lut: texture_2d<f32>;
@group(1) @binding(4) var ibl_sampler: sampler;

// Group 2: Material.
@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var albedo_tex: texture_2d<f32>;
@group(2) @binding(2) var normal_tex: texture_2d<f32>;
@group(2) @binding(3) var mr_tex: texture_2d<f32>;
@group(2) @binding(4) var emissive_tex: texture_2d<f32>;
@group(2) @binding(5) var mat_sampler: sampler;

struct VertexInput {
    // Per-vertex (slot 0)
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) tangent: vec4<f32>,
    // Per-instance (slot 1)
    @location(5) model_0: vec4<f32>,
    @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>,
    @location(8) model_3: vec4<f32>,
    @location(9) inst_color: vec4<f32>,
    @location(10) inst_params: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) params: vec4<f32>,
    @location(5) world_tangent: vec4<f32>,
    @location(6) view_depth: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let world_pos = model * vec4<f32>(in.position, 1.0);

    // Normal matrix: normalize columns of mat3(model).
    let normal_mat = mat3x3<f32>(
        normalize(model[0].xyz),
        normalize(model[1].xyz),
        normalize(model[2].xyz),
    );

    let clip = uniforms.view_projection * world_pos;

    var out: VertexOutput;
    out.clip_position = clip;
    out.world_position = world_pos.xyz;
    out.world_normal = normalize(normal_mat * in.normal);
    out.uv = in.uv;
    out.color = in.color * in.inst_color;
    out.params = in.inst_params;
    out.view_depth = length(world_pos.xyz - uniforms.camera_position.xyz);

    // Transform tangent to world space (w = bitangent handedness, preserved).
    let wt = normalize(normal_mat * in.tangent.xyz);
    out.world_tangent = vec4<f32>(wt, in.tangent.w);

    return out;
}

// Sample shadow for a single cascade. Returns 0.0 (shadowed) to 1.0 (lit).
fn shadow_sample_cascade(biased_pos: vec3<f32>, cascade: i32) -> f32 {
    let light_pos = shadow.light_vp[cascade] * vec4<f32>(biased_pos, 1.0);
    var proj = light_pos.xyz / light_pos.w;

    // NDC to UV: [-1,1] -> [0,1], flip Y.
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 1.0 - (proj.y * 0.5 + 0.5));

    // Out of shadow map bounds -> fully lit.
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z < 0.0 || proj.z > 1.0 {
        return 1.0;
    }

    let depth_bias = shadow.shadow_config.x;
    let compare_depth = proj.z - depth_bias;

    // 3x3 PCF (percentage-closer filtering).
    let tex_size = vec2<f32>(textureDimensions(shadow_depth));
    let texel_size = 1.0 / tex_size;
    var total = 0.0;
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel_size;
            total += textureSampleCompareLevel(
                shadow_depth,
                shadow_sampler,
                uv + offset,
                cascade,
                compare_depth,
            );
        }
    }

    return total / 9.0;
}

// Shadow factor: returns 0.0 (fully shadowed) to 1.0 (fully lit).
// Applies normal bias and blends between adjacent cascades at split boundaries.
fn shadow_factor(world_pos: vec3<f32>, normal: vec3<f32>, view_depth: f32) -> f32 {
    let cascade_count = i32(shadow.shadow_config.w);
    if cascade_count == 0 {
        return 1.0;
    }

    // Apply normal bias: offset along surface normal to reduce acne at
    // grazing angles.
    let normal_bias = shadow.shadow_config.y;
    let biased_pos = world_pos + normal * normal_bias;

    // Select cascade by view depth.
    var cascade = cascade_count - 1;
    for (var i = 0; i < cascade_count; i = i + 1) {
        if view_depth < shadow.splits_count[i] {
            cascade = i;
            break;
        }
    }

    let sf = shadow_sample_cascade(biased_pos, cascade);

    // Blend with next cascade near the split boundary to hide seams.
    let next = cascade + 1;
    if next < cascade_count {
        let split_far = shadow.splits_count[cascade];
        // Blend zone: last 10% of current cascade's range.
        let split_near = select(0.0, shadow.splits_count[cascade - 1], cascade > 0);
        let blend_start = mix(split_near, split_far, 0.75);
        if view_depth > blend_start {
            let t = (view_depth - blend_start) / max(split_far - blend_start, 0.001);
            let sf_next = shadow_sample_cascade(biased_pos, next);
            return mix(sf, sf_next, t);
        }
    }

    return sf;
}

// Spot light attenuation.
fn spot_attenuation(cos_theta: f32, cos_inner: f32, cos_outer: f32) -> f32 {
    return smoothstep(cos_outer, cos_inner, cos_theta);
}

// Helper: apply normal map using TBN matrix.
// Returns perturbed world normal. If tangent is zero-length, returns geometric normal.
fn apply_normal_map(geo_normal: vec3<f32>, world_tangent: vec4<f32>, uv: vec2<f32>, normal_scale: f32) -> vec3<f32> {
    let has_normal = material.texture_flags.y;
    let tang_len = length(world_tangent.xyz);
    if has_normal < 0.5 || tang_len < 0.001 {
        return geo_normal;
    }

    let t = normalize(world_tangent.xyz);
    let n = normalize(geo_normal);
    let b = cross(n, t) * world_tangent.w;

    let tbn = mat3x3<f32>(t, b, n);

    let sampled = textureSample(normal_tex, mat_sampler, uv).xyz;
    var ts_normal = sampled * 2.0 - 1.0;
    ts_normal.x = ts_normal.x * normal_scale;
    ts_normal.y = ts_normal.y * normal_scale;

    return normalize(tbn * ts_normal);
}
";

/// Fragment shader: Unlit — flat color + emissive.
const FS_UNLIT: &str = r"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(albedo_tex, mat_sampler, in.uv);
    let has_tex = material.texture_flags.x;
    let tex_color = mix(vec4<f32>(1.0), texel, has_tex);
    let base = in.color * material.albedo * tex_color;

    var emissive = material.emissive_metallic.xyz;
    let has_emissive_tex = material.texture_flags.w;
    if has_emissive_tex > 0.5 {
        let emissive_texel = textureSample(emissive_tex, mat_sampler, in.uv).rgb;
        emissive = emissive * emissive_texel;
    }

    return vec4<f32>(base.rgb + emissive, base.a);
}
";

/// Fragment shader: Lit — Lambertian diffuse + ambient + point lights + spot lights + shadows + normal mapping.
const FS_LIT: &str = r"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(albedo_tex, mat_sampler, in.uv);
    let has_tex = material.texture_flags.x;
    let tex_color = mix(vec4<f32>(1.0), texel, has_tex);
    let base = in.color * material.albedo * tex_color;

    var emissive = material.emissive_metallic.xyz;
    let has_emissive_tex = material.texture_flags.w;
    if has_emissive_tex > 0.5 {
        let emissive_texel = textureSample(emissive_tex, mat_sampler, in.uv).rgb;
        emissive = emissive * emissive_texel;
    }

    let normal_scale = material.extra.x;
    let n = apply_normal_map(in.world_normal, in.world_tangent, in.uv, normal_scale);

    // Ambient.
    let ambient = lights.ambient.rgb * lights.ambient.w;

    // Shadow factor for directional light.
    let sf = shadow_factor(in.world_position, n, in.view_depth);

    // Directional light (with shadows).
    // Negate: stored direction is light travel; shading needs surface-to-light.
    let light_dir = -normalize(lights.directional_dir_intensity.xyz);
    let dir_intensity = lights.directional_dir_intensity.w;
    let dir_color = lights.directional_color_count.xyz;
    let ndotl = max(dot(n, light_dir), 0.0);
    var diffuse = dir_color * dir_intensity * ndotl * sf;

    // Point lights.
    let point_count = i32(lights.directional_color_count.w);
    for (var i = 0; i < point_count; i = i + 1) {
        let pl = lights.point_lights[i];
        let pl_pos = pl.position_range.xyz;
        let pl_range = pl.position_range.w;
        let pl_color = pl.color_intensity.xyz;
        let pl_intensity = pl.color_intensity.w;

        let to_light = pl_pos - in.world_position;
        let dist = length(to_light);
        if dist < pl_range {
            let l = to_light / max(dist, 0.001);
            let atten = pl_intensity / max(dist * dist, 0.01);
            let pl_ndotl = max(dot(n, l), 0.0);
            diffuse = diffuse + pl_color * atten * pl_ndotl;
        }
    }

    // Spot lights.
    let spot_count = i32(lights.spot_count_pad.x);
    for (var i = 0; i < spot_count; i = i + 1) {
        let sl = lights.spot_lights[i];
        let sl_pos = sl.position_range.xyz;
        let sl_range = sl.position_range.w;
        let sl_dir = sl.direction_inner.xyz;
        let sl_cos_inner = sl.direction_inner.w;
        let sl_color = sl.color_intensity.xyz;
        let sl_intensity = sl.color_intensity.w;
        let sl_cos_outer = sl.outer_pad.x;

        let to_light = sl_pos - in.world_position;
        let dist = length(to_light);
        if dist < sl_range {
            let l = to_light / max(dist, 0.001);
            let cos_theta = dot(normalize(-sl_dir), l);
            let spot_atten = spot_attenuation(cos_theta, sl_cos_inner, sl_cos_outer);
            let dist_atten = sl_intensity / max(dist * dist, 0.01);
            let sl_ndotl = max(dot(n, l), 0.0);
            diffuse = diffuse + sl_color * dist_atten * spot_atten * sl_ndotl;
        }
    }

    let lit = base.rgb * (ambient + diffuse) + emissive;
    return vec4<f32>(lit, base.a);
}
";

/// Fragment shader: PBR — Cook-Torrance microfacet BRDF with shadows, spot lights, IBL.
const FS_PBR: &str = r"
const PI: f32 = 3.14159265358979323846;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn cook_torrance_brdf(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_v = max(dot(n, v), 0.001);
    let n_dot_l = max(dot(n, l), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);

    let numerator = d * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let specular = numerator / denominator;

    let k_s = f;
    let k_d = (vec3<f32>(1.0) - k_s) * (1.0 - metallic);

    return (k_d * albedo / PI + specular) * n_dot_l;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(albedo_tex, mat_sampler, in.uv);
    let has_tex = material.texture_flags.x;
    let tex_color = mix(vec4<f32>(1.0), texel, has_tex);
    let base = in.color * material.albedo * tex_color;
    let albedo = base.rgb;

    var emissive = material.emissive_metallic.xyz;
    let has_emissive_tex = material.texture_flags.w;
    if has_emissive_tex > 0.5 {
        let emissive_texel = textureSample(emissive_tex, mat_sampler, in.uv).rgb;
        emissive = emissive * emissive_texel;
    }

    // Read metallic/roughness from uniform or texture.
    var metallic = material.emissive_metallic.w;
    var roughness = max(material.roughness_opacity_flags.x, 0.04);

    let has_mr_tex = material.texture_flags.z;
    if has_mr_tex > 0.5 {
        let mr_sample = textureSample(mr_tex, mat_sampler, in.uv);
        // glTF channel packing: G=roughness, B=metallic
        roughness = max(roughness * mr_sample.g, 0.04);
        metallic = metallic * mr_sample.b;
    }

    // Normal mapping.
    let normal_scale = material.extra.x;
    let n = apply_normal_map(in.world_normal, in.world_tangent, in.uv, normal_scale);
    let v = normalize(uniforms.camera_position.xyz - in.world_position);
    let n_dot_v = max(dot(n, v), 0.001);

    // Shadow factor for directional light.
    let sf = shadow_factor(in.world_position, n, in.view_depth);

    // Directional light (with shadows).
    // Negate: stored direction is light travel; BRDF expects surface-to-light.
    let light_dir = -normalize(lights.directional_dir_intensity.xyz);
    let dir_intensity = lights.directional_dir_intensity.w;
    let dir_color = lights.directional_color_count.xyz;
    var lo = dir_color * dir_intensity * sf * cook_torrance_brdf(n, v, light_dir, albedo, metallic, roughness);

    // Point lights.
    let point_count = i32(lights.directional_color_count.w);
    for (var i = 0; i < point_count; i = i + 1) {
        let pl = lights.point_lights[i];
        let pl_pos = pl.position_range.xyz;
        let pl_range = pl.position_range.w;
        let pl_color = pl.color_intensity.xyz;
        let pl_intensity = pl.color_intensity.w;

        let to_light = pl_pos - in.world_position;
        let dist = length(to_light);
        if dist < pl_range {
            let l = to_light / max(dist, 0.001);
            let atten = pl_intensity / max(dist * dist, 0.01);
            lo = lo + pl_color * atten * cook_torrance_brdf(n, v, l, albedo, metallic, roughness);
        }
    }

    // Spot lights.
    let spot_count = i32(lights.spot_count_pad.x);
    for (var i = 0; i < spot_count; i = i + 1) {
        let sl = lights.spot_lights[i];
        let sl_pos = sl.position_range.xyz;
        let sl_range = sl.position_range.w;
        let sl_dir = sl.direction_inner.xyz;
        let sl_cos_inner = sl.direction_inner.w;
        let sl_color = sl.color_intensity.xyz;
        let sl_intensity = sl.color_intensity.w;
        let sl_cos_outer = sl.outer_pad.x;

        let to_light = sl_pos - in.world_position;
        let dist = length(to_light);
        if dist < sl_range {
            let l = to_light / max(dist, 0.001);
            let cos_theta = dot(normalize(-sl_dir), l);
            let s_atten = spot_attenuation(cos_theta, sl_cos_inner, sl_cos_outer);
            let d_atten = sl_intensity / max(dist * dist, 0.01);
            lo = lo + sl_color * d_atten * s_atten * cook_torrance_brdf(n, v, l, albedo, metallic, roughness);
        }
    }

    // IBL ambient (split-sum approximation).
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f_ibl = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let k_d_ibl = (vec3<f32>(1.0) - f_ibl) * (1.0 - metallic);

    let diffuse_ibl = textureSample(irradiance_map, ibl_sampler, n).rgb * albedo;
    let r = reflect(-v, n);
    let prefiltered = textureSampleLevel(prefiltered_map, ibl_sampler, r, roughness * 4.0).rgb;
    let brdf_sample = textureSample(brdf_lut, ibl_sampler, vec2<f32>(n_dot_v, roughness)).rg;
    let specular_ibl = prefiltered * (f_ibl * brdf_sample.x + brdf_sample.y);

    let ambient_ibl = (k_d_ibl * diffuse_ibl + specular_ibl) * lights.ambient.w;

    let color = ambient_ibl + lo + emissive;
    return vec4<f32>(color, base.a);
}
";

/// Composite shader: fullscreen triangle blending scene HDR + bloom + SSAO, with ACES tone mapping.
const COMPOSITE_SHADER_3D: &str = r"
struct CompositeParams {
    bloom_intensity: f32,
    tone_map: f32,
    ssao_enabled: f32,
    _pad: f32,
}

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var ssao_tex: texture_2d<f32>;
@group(0) @binding(3) var linear_samp: sampler;
@group(0) @binding(4) var<uniform> params: CompositeParams;

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

// ACES filmic tone mapping (simple fit).
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var color = textureSample(scene_tex, linear_samp, in.uv).rgb;

    // Add bloom.
    let bloom = textureSample(bloom_tex, linear_samp, in.uv).rgb;
    color = color + bloom * params.bloom_intensity;

    // Apply SSAO.
    if params.ssao_enabled > 0.5 {
        let ao = textureSample(ssao_tex, linear_samp, in.uv).r;
        color = color * ao;
    }

    // Tone mapping.
    if params.tone_map > 0.5 {
        color = aces_filmic(color);
    }

    return vec4<f32>(color, 1.0);
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::material::{BlendMode3D, CullMode3D, MaterialHandle, MaterialType};

    #[test]
    fn draw_cmd_sort_order() {
        let keys = vec![
            PipelineKey {
                material_type: MaterialType::Lit,
                blend_mode: BlendMode3D::Opaque,
                cull_mode: CullMode3D::Back,
                depth_write: true,
            },
            PipelineKey {
                material_type: MaterialType::PBR,
                blend_mode: BlendMode3D::Opaque,
                cull_mode: CullMode3D::Back,
                depth_write: true,
            },
        ];

        let mut cmds = vec![
            DrawCmd {
                mesh: MeshHandle(1),
                material: MaterialHandle(1),
                instance_offset: 0,
                instance_count: 1,
            },
            DrawCmd {
                mesh: MeshHandle(0),
                material: MaterialHandle(0),
                instance_offset: 1,
                instance_count: 1,
            },
        ];

        cmds.sort_by(|a, b| {
            let key_a = &keys[a.material.0 as usize];
            let key_b = &keys[b.material.0 as usize];
            pipeline_key_sort_tuple(key_a, a.material.0, a.mesh.0).cmp(
                &pipeline_key_sort_tuple(key_b, b.material.0, b.mesh.0),
            )
        });

        assert_eq!(cmds[0].material.0, 0); // Lit
        assert_eq!(cmds[1].material.0, 1); // PBR
    }

    #[test]
    fn draw_cmd_merge_adjacent() {
        let cmds = vec![
            DrawCmd {
                mesh: MeshHandle(0),
                material: MaterialHandle(0),
                instance_offset: 0,
                instance_count: 5,
            },
            DrawCmd {
                mesh: MeshHandle(0),
                material: MaterialHandle(0),
                instance_offset: 5,
                instance_count: 3,
            },
            DrawCmd {
                mesh: MeshHandle(1),
                material: MaterialHandle(0),
                instance_offset: 8,
                instance_count: 2,
            },
        ];

        let mut merged = Vec::new();
        let mut i = 0;
        while i < cmds.len() {
            let mut count = cmds[i].instance_count;
            let mut j = i + 1;
            while j < cmds.len()
                && cmds[j].material.0 == cmds[i].material.0
                && cmds[j].mesh.0 == cmds[i].mesh.0
                && cmds[j].instance_offset == cmds[i].instance_offset + count
            {
                count += cmds[j].instance_count;
                j += 1;
            }
            merged.push((
                cmds[i].mesh.0,
                cmds[i].material.0,
                cmds[i].instance_offset,
                count,
            ));
            i = j;
        }

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], (0, 0, 0, 8));
        assert_eq!(merged[1], (1, 0, 8, 2));
    }

    #[test]
    fn batch_stats_default() {
        let stats = BatchStats3D::default();
        assert_eq!(stats.draw_calls, 0);
        assert_eq!(stats.total_triangles, 0);
    }

    #[test]
    fn transparency_sort_back_to_front() {
        let cam_pos = glam::Vec3::new(0.0, 0.0, 0.0);

        let staging = vec![
            InstanceData {
                model: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, -5.0, 1.0],
                ],
                color: [1.0; 4],
                params: [0.0; 4],
            },
            InstanceData {
                model: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, -20.0, 1.0],
                ],
                color: [1.0; 4],
                params: [0.0; 4],
            },
            InstanceData {
                model: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, -10.0, 1.0],
                ],
                color: [1.0; 4],
                params: [0.0; 4],
            },
        ];

        let mut indices: Vec<usize> = vec![0, 1, 2];
        indices.sort_by(|&a, &b| {
            let pos_a = instance_translation(&staging, a as u32);
            let pos_b = instance_translation(&staging, b as u32);
            let dist_a = cam_pos.distance_squared(pos_a);
            let dist_b = cam_pos.distance_squared(pos_b);
            dist_b
                .partial_cmp(&dist_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        assert_eq!(indices, vec![1, 2, 0]);
    }
}
