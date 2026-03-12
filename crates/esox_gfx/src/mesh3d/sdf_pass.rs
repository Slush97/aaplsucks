//! SDF (Signed Distance Field) effects pass for the 3D renderer.
//!
//! Provides infrastructure for user-supplied SDF raymarching shaders
//! (volumetrics, procedural terrain, fog, particles). The framework supplies
//! boilerplate (ray generation, marching loop, normal computation, lighting).
//! The user supplies `fn sdf_scene(p: vec3<f32>) -> f32` and optionally
//! `fn sdf_material(p: vec3<f32>, n: vec3<f32>) -> vec4<f32>`.
//!
//! SDF effects render as fullscreen triangle passes composited onto the
//! existing 3D scene (LoadOp::Load) using alpha blending or additive blending.

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

// ── Surface format ──

/// Surface format for the SDF render target (matches 3D offscreen).
const SDF_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// ── Blend mode ──

/// Blend mode for compositing an SDF effect with the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdfBlendMode {
    /// Normal alpha blending over the scene.
    AlphaBlend,
    /// Additive blending (for glows, fog).
    Additive,
}

// ── Effect descriptor ──

/// Describes an SDF effect to register.
pub struct SdfEffectDescriptor {
    /// User-supplied WGSL body containing `fn sdf_scene(p: vec3<f32>) -> f32`
    /// and optionally `fn sdf_material(p: vec3<f32>, n: vec3<f32>) -> vec4<f32>`.
    pub shader_body: String,
    /// Whether to composite with scene depth (read depth buffer for occlusion).
    pub depth_composite: bool,
    /// Blend mode for compositing with the scene.
    pub blend: SdfBlendMode,
    /// Maximum raymarching distance.
    pub max_distance: f32,
    /// Maximum raymarching steps.
    pub max_steps: u32,
}

// ── Effect handle ──

/// Opaque handle to a registered SDF effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SdfEffectHandle(pub(crate) u32);

// ── GPU params ──

/// GPU uniform data for the SDF pass.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SdfParams {
    /// Inverse view-projection matrix (column-major).
    pub inv_view_projection: [[f32; 4]; 4],
    /// Camera world-space position (w unused).
    pub camera_position: [f32; 4],
    /// Viewport: [width, height, 1/width, 1/height].
    pub viewport: [f32; 4],
    /// Time: [elapsed_seconds, delta_seconds, max_distance, max_steps].
    pub time: [f32; 4],
}

// ── Shader source ──

/// WGSL preamble injected before every SDF effect shader.
///
/// Provides bindings, the fullscreen vertex shader, ray generation,
/// SDF normal computation, and scene-depth readback.
const SDF_PREAMBLE: &str = r#"
struct SdfParams {
    inv_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,
    time: vec4<f32>,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var depth_sampler: sampler;
@group(0) @binding(2) var<uniform> sdf: SdfParams;

// Fullscreen triangle vertex shader
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Generate ray from screen UV
fn generate_ray(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let world_far = sdf.inv_view_projection * vec4<f32>(ndc, 1.0, 1.0);
    let world_near = sdf.inv_view_projection * vec4<f32>(ndc, 0.0, 1.0);
    let far = world_far.xyz / world_far.w;
    let near = world_near.xyz / world_near.w;
    return normalize(far - near);
}

// Compute normal from SDF gradient
fn sdf_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.001;
    let n = vec3<f32>(
        sdf_scene(p + vec3<f32>(e, 0.0, 0.0)) - sdf_scene(p - vec3<f32>(e, 0.0, 0.0)),
        sdf_scene(p + vec3<f32>(0.0, e, 0.0)) - sdf_scene(p - vec3<f32>(0.0, e, 0.0)),
        sdf_scene(p + vec3<f32>(0.0, 0.0, e)) - sdf_scene(p - vec3<f32>(0.0, 0.0, e)),
    );
    return normalize(n);
}

// Read scene depth and convert to linear distance
fn scene_depth_at(uv: vec2<f32>) -> f32 {
    let d = textureSample(depth_tex, depth_sampler, uv);
    // Reconstruct world position from depth
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let world = sdf.inv_view_projection * vec4<f32>(ndc, d, 1.0);
    let world_pos = world.xyz / world.w;
    return length(world_pos - sdf.camera_position.xyz);
}
"#;

/// Fragment shader template for SDF effects WITHOUT user material.
///
/// The user must define `fn sdf_scene(p: vec3<f32>) -> f32` in their shader body.
/// This template provides the default directional lighting.
const SDF_FRAGMENT_DEFAULT: &str = r#"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = generate_ray(in.uv);
    let ray_origin = sdf.camera_position.xyz;
    let max_dist = sdf.time.z;
    let max_steps = i32(sdf.time.w);

    var t = 0.0;
    var hit = false;
    for (var i = 0; i < max_steps; i = i + 1) {
        let p = ray_origin + ray_dir * t;
        let d = sdf_scene(p);
        if d < 0.001 {
            hit = true;
            break;
        }
        t += d;
        if t > max_dist {
            break;
        }
    }

    if !hit {
        discard;
    }

    // Depth compositing
    let scene_dist = scene_depth_at(in.uv);
    if t > scene_dist {
        discard;
    }

    let hit_pos = ray_origin + ray_dir * t;
    let normal = sdf_normal(hit_pos);

    // Default material: basic directional lighting
    let color = vec4<f32>(0.8, 0.8, 0.8, 1.0);
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(normal, light_dir), 0.0);
    let lit = color.rgb * (0.2 + 0.8 * ndotl);

    return vec4<f32>(lit, color.a);
}
"#;

/// Fragment shader template for SDF effects WITH user-supplied `sdf_material`.
///
/// Calls `sdf_material(hit_pos, normal)` to get the surface color instead of
/// using the default lighting.
const SDF_FRAGMENT_MATERIAL: &str = r#"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = generate_ray(in.uv);
    let ray_origin = sdf.camera_position.xyz;
    let max_dist = sdf.time.z;
    let max_steps = i32(sdf.time.w);

    var t = 0.0;
    var hit = false;
    for (var i = 0; i < max_steps; i = i + 1) {
        let p = ray_origin + ray_dir * t;
        let d = sdf_scene(p);
        if d < 0.001 {
            hit = true;
            break;
        }
        t += d;
        if t > max_dist {
            break;
        }
    }

    if !hit {
        discard;
    }

    // Depth compositing
    let scene_dist = scene_depth_at(in.uv);
    if t > scene_dist {
        discard;
    }

    let hit_pos = ray_origin + ray_dir * t;
    let normal = sdf_normal(hit_pos);

    // User-supplied material
    let color = sdf_material(hit_pos, normal);
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(normal, light_dir), 0.0);
    let lit = color.rgb * (0.2 + 0.8 * ndotl);

    return vec4<f32>(lit, color.a);
}
"#;

/// Fragment shader template WITHOUT depth compositing and WITHOUT user material.
const SDF_FRAGMENT_NO_DEPTH_DEFAULT: &str = r#"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = generate_ray(in.uv);
    let ray_origin = sdf.camera_position.xyz;
    let max_dist = sdf.time.z;
    let max_steps = i32(sdf.time.w);

    var t = 0.0;
    var hit = false;
    for (var i = 0; i < max_steps; i = i + 1) {
        let p = ray_origin + ray_dir * t;
        let d = sdf_scene(p);
        if d < 0.001 {
            hit = true;
            break;
        }
        t += d;
        if t > max_dist {
            break;
        }
    }

    if !hit {
        discard;
    }

    let hit_pos = ray_origin + ray_dir * t;
    let normal = sdf_normal(hit_pos);

    // Default material: basic directional lighting
    let color = vec4<f32>(0.8, 0.8, 0.8, 1.0);
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(normal, light_dir), 0.0);
    let lit = color.rgb * (0.2 + 0.8 * ndotl);

    return vec4<f32>(lit, color.a);
}
"#;

/// Fragment shader template WITHOUT depth compositing but WITH user material.
const SDF_FRAGMENT_NO_DEPTH_MATERIAL: &str = r#"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let ray_dir = generate_ray(in.uv);
    let ray_origin = sdf.camera_position.xyz;
    let max_dist = sdf.time.z;
    let max_steps = i32(sdf.time.w);

    var t = 0.0;
    var hit = false;
    for (var i = 0; i < max_steps; i = i + 1) {
        let p = ray_origin + ray_dir * t;
        let d = sdf_scene(p);
        if d < 0.001 {
            hit = true;
            break;
        }
        t += d;
        if t > max_dist {
            break;
        }
    }

    if !hit {
        discard;
    }

    let hit_pos = ray_origin + ray_dir * t;
    let normal = sdf_normal(hit_pos);

    // User-supplied material
    let color = sdf_material(hit_pos, normal);
    let light_dir = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(normal, light_dir), 0.0);
    let lit = color.rgb * (0.2 + 0.8 * ndotl);

    return vec4<f32>(lit, color.a);
}
"#;

// ── Internal effect storage ──

/// A registered SDF effect with its pipeline and GPU resources.
struct SdfEffect {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    enabled: bool,
    #[allow(dead_code)]
    depth_composite: bool,
    max_distance: f32,
    max_steps: u32,
}

// ── SdfPass ──

/// Manages registered SDF effects and their pipelines.
///
/// Each effect is a fullscreen raymarching pass composited onto the existing 3D
/// scene. Effects are rendered in registration order.
pub struct SdfPass {
    effects: Vec<SdfEffect>,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
}

impl SdfPass {
    /// Create the SDF pass bind group layout and pipeline layout.
    pub fn new(device: &wgpu::Device) -> Self {
        // ── Bind group layout ──
        //
        // binding 0: scene depth texture (texture_depth_2d)
        // binding 1: sampler (non-filtering, for depth reads)
        // binding 2: SDF params uniform buffer
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf_bind_group_layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            size_of::<SdfParams>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        Self {
            effects: Vec::new(),
            bind_group_layout,
            pipeline_layout,
        }
    }

    /// Register an SDF effect.
    ///
    /// Composes the full WGSL source from the preamble, user shader body, and
    /// the appropriate fragment template. Validates the shader with naga, then
    /// creates the render pipeline, bind group, and params buffer.
    ///
    /// The user's `shader_body` must define `fn sdf_scene(p: vec3<f32>) -> f32`.
    /// It may optionally define `fn sdf_material(p: vec3<f32>, n: vec3<f32>) -> vec4<f32>`.
    pub fn register_effect(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        depth_view: &wgpu::TextureView,
        desc: &SdfEffectDescriptor,
    ) -> Result<SdfEffectHandle, String> {
        // Detect whether the user provides a custom material function.
        let has_material = desc.shader_body.contains("fn sdf_material");

        // Choose the appropriate fragment template.
        let fragment_template = match (desc.depth_composite, has_material) {
            (true, false) => SDF_FRAGMENT_DEFAULT,
            (true, true) => SDF_FRAGMENT_MATERIAL,
            (false, false) => SDF_FRAGMENT_NO_DEPTH_DEFAULT,
            (false, true) => SDF_FRAGMENT_NO_DEPTH_MATERIAL,
        };

        // Compose full WGSL source.
        let full_source = format!("{SDF_PREAMBLE}\n{}\n{fragment_template}", desc.shader_body);

        // Validate with naga.
        let module = naga::front::wgsl::parse_str(&full_source)
            .map_err(|e| format!("SDF WGSL parse error: {e}"))?;
        let _info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| format!("SDF WGSL validation error: {e}"))?;

        // Create shader module.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf_effect_shader"),
            source: wgpu::ShaderSource::Wgsl(full_source.into()),
        });

        // Blend state.
        let blend_state = match desc.blend {
            SdfBlendMode::AlphaBlend => wgpu::BlendState::ALPHA_BLENDING,
            SdfBlendMode::Additive => wgpu::BlendState {
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
        };

        // Create render pipeline.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sdf_effect_pipeline"),
            layout: Some(&self.pipeline_layout),
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
                    format: SDF_TARGET_FORMAT,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Fullscreen triangle — no culling.
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None, // SDF pass has no depth write.
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Create params buffer with default values.
        let params = SdfParams {
            inv_view_projection: Mat4::IDENTITY.to_cols_array_2d(),
            camera_position: [0.0; 4],
            viewport: [1.0, 1.0, 1.0, 1.0],
            time: [0.0, 0.0, desc.max_distance, desc.max_steps as f32],
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sdf_params_buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create sampler (non-filtering, for depth reads).
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sdf_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sdf_effect_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let handle = SdfEffectHandle(self.effects.len() as u32);

        self.effects.push(SdfEffect {
            pipeline,
            bind_group,
            params_buffer,
            enabled: true,
            depth_composite: desc.depth_composite,
            max_distance: desc.max_distance,
            max_steps: desc.max_steps,
        });

        Ok(handle)
    }

    /// Enable or disable an SDF effect.
    pub fn set_enabled(&mut self, handle: SdfEffectHandle, enabled: bool) {
        if let Some(effect) = self.effects.get_mut(handle.0 as usize) {
            effect.enabled = enabled;
        }
    }

    /// Encode all enabled SDF effects into the command encoder.
    ///
    /// Each effect renders a fullscreen triangle onto `target` (LoadOp::Load)
    /// with the appropriate blend mode. The params buffer is updated with the
    /// current camera and viewport state before each draw.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        _depth_view: &wgpu::TextureView,
        inv_vp: Mat4,
        camera_pos: Vec3,
        viewport: [f32; 4],
        elapsed: f32,
        delta: f32,
    ) {
        for effect in &self.effects {
            if !effect.enabled {
                continue;
            }

            // Update params buffer.
            let params = SdfParams {
                inv_view_projection: inv_vp.to_cols_array_2d(),
                camera_position: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
                viewport,
                time: [
                    elapsed,
                    delta,
                    effect.max_distance,
                    effect.max_steps as f32,
                ],
            };
            queue.write_buffer(&effect.params_buffer, 0, bytemuck::bytes_of(&params));

            // Begin render pass (Load existing scene content).
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sdf_effect_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&effect.pipeline);
            pass.set_bind_group(0, &effect.bind_group, &[]);
            pass.draw(0..3, 0..1); // Fullscreen triangle.
        }
    }

    /// Rebuild bind groups when the depth texture changes (e.g., on resize).
    pub fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
    ) {
        // Create a shared sampler for the new bind groups.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sdf_depth_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        for effect in &mut self.effects {
            effect.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sdf_effect_bind_group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: effect.params_buffer.as_entire_binding(),
                    },
                ],
            });
        }
    }
}

// ── Shader composition helper (exposed for testing) ──

/// Compose a full SDF shader from the preamble, user body, and fragment template.
///
/// Exposed for unit testing so we can validate composed shaders without a GPU.
#[cfg(test)]
fn compose_shader(shader_body: &str, depth_composite: bool) -> String {
    let has_material = shader_body.contains("fn sdf_material");
    let fragment_template = match (depth_composite, has_material) {
        (true, false) => SDF_FRAGMENT_DEFAULT,
        (true, true) => SDF_FRAGMENT_MATERIAL,
        (false, false) => SDF_FRAGMENT_NO_DEPTH_DEFAULT,
        (false, true) => SDF_FRAGMENT_NO_DEPTH_MATERIAL,
    };
    format!("{SDF_PREAMBLE}\n{shader_body}\n{fragment_template}")
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdf_params_size() {
        // 4x4 mat (64) + vec4 (16) + vec4 (16) + vec4 (16) = 112 bytes.
        assert_eq!(size_of::<SdfParams>(), 112);
    }

    #[test]
    fn sdf_preamble_parses() {
        // Compose a minimal valid shader with the preamble and a trivial sdf_scene.
        let user_body = r#"
fn sdf_scene(p: vec3<f32>) -> f32 {
    return length(p) - 1.0;
}
"#;
        let full = compose_shader(user_body, true);
        let module =
            naga::front::wgsl::parse_str(&full).expect("composed SDF shader should parse");
        let _info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("composed SDF shader should validate");
    }

    #[test]
    fn sdf_preamble_with_material_parses() {
        let user_body = r#"
fn sdf_scene(p: vec3<f32>) -> f32 {
    return length(p) - 1.0;
}

fn sdf_material(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(n * 0.5 + 0.5, 1.0);
}
"#;
        let full = compose_shader(user_body, true);
        let module = naga::front::wgsl::parse_str(&full)
            .expect("composed SDF shader with material should parse");
        let _info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("composed SDF shader with material should validate");
    }

    #[test]
    fn sdf_no_depth_parses() {
        let user_body = r#"
fn sdf_scene(p: vec3<f32>) -> f32 {
    return length(p) - 1.0;
}
"#;
        let full = compose_shader(user_body, false);
        let module = naga::front::wgsl::parse_str(&full)
            .expect("composed SDF shader (no depth) should parse");
        let _info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("composed SDF shader (no depth) should validate");
    }

    #[test]
    fn sdf_effect_handle_equality() {
        let a = SdfEffectHandle(0);
        let b = SdfEffectHandle(0);
        let c = SdfEffectHandle(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn sdf_blend_mode_equality() {
        assert_eq!(SdfBlendMode::AlphaBlend, SdfBlendMode::AlphaBlend);
        assert_eq!(SdfBlendMode::Additive, SdfBlendMode::Additive);
        assert_ne!(SdfBlendMode::AlphaBlend, SdfBlendMode::Additive);
    }

    #[test]
    fn sdf_params_zeroable() {
        use bytemuck::Zeroable;
        let params = SdfParams::zeroed();
        assert_eq!(params.time, [0.0; 4]);
        assert_eq!(params.camera_position, [0.0; 4]);
    }

    #[test]
    fn compose_detects_material() {
        let body_no_mat = "fn sdf_scene(p: vec3<f32>) -> f32 { return 1.0; }";
        let body_with_mat = "fn sdf_scene(p: vec3<f32>) -> f32 { return 1.0; }\nfn sdf_material(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> { return vec4<f32>(1.0); }";

        let no_mat = compose_shader(body_no_mat, true);
        let with_mat = compose_shader(body_with_mat, true);

        // The material variant calls sdf_material in the fragment shader.
        assert!(!no_mat.contains("sdf_material(hit_pos"));
        assert!(with_mat.contains("sdf_material(hit_pos"));
    }
}
