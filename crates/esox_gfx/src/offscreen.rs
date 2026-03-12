//! Offscreen render target for post-processing effects.
//!
//! Provides the foundation for CRT, blur, bloom, frosted glass, and other
//! full-screen effects that require rendering the scene to a texture first,
//! then sampling that texture in a second pass.

use crate::primitive::ShaderId;

/// Pipeline ID for the post-process fullscreen pass.
pub const PIPELINE_POST_PROCESS: ShaderId = ShaderId(100);

/// GPU-side parameters for post-processing effects (32 bytes, uniform buffer).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PostProcessParams {
    /// Vignette darkening at edges (0.0–1.0).
    pub vignette: f32,
    /// Film grain intensity (0.0–1.0).
    pub grain: f32,
    /// CRT scanline intensity (0.0–1.0).
    pub scanlines: f32,
    /// CRT barrel distortion (0.0–1.0).
    pub curvature: f32,
    /// Chromatic aberration offset in pixels (0.0–5.0).
    pub chromatic_aberration: f32,
    /// Elapsed time in seconds (drives grain animation).
    pub time: f32,
    /// ACES tone mapping strength (0.0 = off, 1.0 = full).
    pub tone_map: f32,
    /// Bloom intensity (0.0–1.0). Controls bloom texture composite strength.
    pub bloom_intensity: f32,
}

impl Default for PostProcessParams {
    fn default() -> Self {
        Self {
            vignette: 0.0,
            grain: 0.0,
            scanlines: 0.0,
            curvature: 0.0,
            chromatic_aberration: 0.0,
            time: 0.0,
            tone_map: 0.0,
            bloom_intensity: 0.0,
        }
    }
}

/// An offscreen render target: texture + views + bind group for sampling.
pub struct OffscreenTarget {
    /// The offscreen texture.
    pub texture: wgpu::Texture,
    /// View used as a render attachment (write).
    pub render_view: wgpu::TextureView,
    /// View used for sampling in the post-process pass (read).
    pub sample_view: wgpu::TextureView,
    /// Bind group for the post-process shader to sample this texture.
    pub sample_bind_group: wgpu::BindGroup,
    /// Uniform buffer for post-process parameters.
    pub params_buffer: wgpu::Buffer,
    /// Current width in pixels.
    pub width: u32,
    /// Current height in pixels.
    pub height: u32,
}

impl OffscreenTarget {
    /// Create a new offscreen render target.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        layout: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        bloom_view: &wgpu::TextureView,
    ) -> Self {
        let (texture, render_view, sample_view) =
            Self::create_texture(device, width, height, format);

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_process_params_buffer"),
            size: size_of::<PostProcessParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sample_bind_group = Self::create_bind_group(
            device,
            layout,
            uniform_buf,
            &sample_view,
            sampler,
            &params_buffer,
            bloom_view,
        );

        Self {
            texture,
            render_view,
            sample_view,
            sample_bind_group,
            params_buffer,
            width,
            height,
        }
    }

    /// Update post-process parameters for the current frame.
    pub fn update_params(&self, queue: &wgpu::Queue, params: &PostProcessParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Recreate the offscreen target if the dimensions changed.
    ///
    /// Returns `true` if the target was actually recreated.
    #[allow(clippy::too_many_arguments)]
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        layout: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        bloom_view: &wgpu::TextureView,
    ) -> bool {
        if self.width == width && self.height == height {
            return false;
        }

        let (texture, render_view, sample_view) =
            Self::create_texture(device, width, height, format);

        self.sample_bind_group = Self::create_bind_group(
            device,
            layout,
            uniform_buf,
            &sample_view,
            sampler,
            &self.params_buffer,
            bloom_view,
        );

        self.texture = texture;
        self.render_view = render_view;
        self.sample_view = sample_view;
        self.width = width;
        self.height = height;
        true
    }

    /// Create a bind group with all five bindings.
    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        sample_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
        bloom_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post_process_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
            ],
        })
    }

    /// Rebuild the bind group with a new bloom texture view.
    ///
    /// Called when the bloom pass is created or resized after the offscreen
    /// target already exists.
    pub fn rebuild_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform_buf: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        bloom_view: &wgpu::TextureView,
    ) {
        self.sample_bind_group = Self::create_bind_group(
            device,
            layout,
            uniform_buf,
            &self.sample_view,
            sampler,
            &self.params_buffer,
            bloom_view,
        );
    }

    /// Create the texture and its two views.
    fn create_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let render_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sample_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, render_view, sample_view)
    }
}

/// Create the bind group layout for post-process shaders.
///
/// Bindings:
/// - 0: frame uniforms (same as scene pass)
/// - 1: scene texture (2D, not array)
/// - 2: sampler
/// - 3: post-process parameters uniform
/// - 4: bloom texture (2D)
pub fn post_process_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("post_process_bind_group_layout"),
        entries: &[
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
                    view_dimension: wgpu::TextureViewDimension::D2,
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
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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
        ],
    })
}

/// Shared WGSL preamble for all post-process shaders.
///
/// Contains struct definitions, resource bindings, and the vertex output type.
/// Both the built-in fragment shader and user-supplied shaders are concatenated
/// after this preamble so they can reference `uniforms`, `pp`, `scene_texture`,
/// `scene_sampler`, and `PostProcessOutput` (aliased as `VertexOutput`).
pub const POST_PROCESS_PREAMBLE: &str = r"
struct FrameUniforms {
    viewport: vec4<f32>,
    time: vec4<f32>,
}

struct PostProcessParams {
    vignette: f32,
    grain: f32,
    scanlines: f32,
    curvature: f32,
    chromatic_aberration: f32,
    time: f32,
    tone_map: f32,
    bloom_intensity: f32,
}

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var scene_sampler: sampler;
@group(0) @binding(3) var<uniform> pp: PostProcessParams;
@group(0) @binding(4) var bloom_texture: texture_2d<f32>;

struct PostProcessOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Alias so user shaders can use `VertexOutput` as documented.
alias VertexOutput = PostProcessOutput;
";

/// Fullscreen triangle vertex shader (no vertex buffer needed).
///
/// Uses `vertex_index` to generate a full-screen triangle covering clip space.
pub const POST_PROCESS_VERTEX: &str = r"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> PostProcessOutput {
    // Full-screen triangle: 3 vertices cover [-1,1] clip space.
    var out: PostProcessOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // UV: [0,1] with y flipped for texture sampling.
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
";

/// Full vertex source (preamble + vertex shader) for convenience.
///
/// Equivalent to `POST_PROCESS_PREAMBLE` + `POST_PROCESS_VERTEX`.
pub const POST_PROCESS_VERTEX_SOURCE: &str = concat!(
    r"
struct FrameUniforms {
    viewport: vec4<f32>,
    time: vec4<f32>,
}

struct PostProcessParams {
    vignette: f32,
    grain: f32,
    scanlines: f32,
    curvature: f32,
    chromatic_aberration: f32,
    time: f32,
    tone_map: f32,
    bloom_intensity: f32,
}

@group(0) @binding(0) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var scene_sampler: sampler;
@group(0) @binding(3) var<uniform> pp: PostProcessParams;
@group(0) @binding(4) var bloom_texture: texture_2d<f32>;

struct PostProcessOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

alias VertexOutput = PostProcessOutput;
",
    r"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> PostProcessOutput {
    var out: PostProcessOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
",
);

/// Post-process fragment shader implementing CRT-style effects.
///
/// Applies barrel distortion, chromatic aberration, scanlines, vignette,
/// and film grain. Each effect is gated on its parameter being > 0.0,
/// so identity passthrough is preserved when all parameters are zero.
pub const POST_PROCESS_FRAGMENT: &str = r"
// Pseudo-random hash for film grain (returns 0..1).
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

// Apply barrel distortion to UVs (CRT curvature).
fn barrel_distort(uv: vec2<f32>, amount: f32) -> vec2<f32> {
    let centered = uv - 0.5;
    let r2 = dot(centered, centered);
    let distorted = centered * (1.0 + amount * r2);
    return distorted + 0.5;
}

@fragment
fn fs_main(in: PostProcessOutput) -> @location(0) vec4<f32> {
    var uv = in.uv;
    let resolution = uniforms.viewport.xy;

    // ── Barrel distortion (curvature) ──
    if pp.curvature > 0.0 {
        uv = barrel_distort(uv, pp.curvature * 4.0);
        // Discard pixels outside [0,1] after distortion (CRT border).
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }

    // ── Chromatic aberration ──
    var color: vec3<f32>;
    if pp.chromatic_aberration > 0.0 {
        let ca_offset = pp.chromatic_aberration * uniforms.viewport.zw;
        let dir = uv - 0.5;
        let r = textureSample(scene_texture, scene_sampler, uv + dir * ca_offset.x).r;
        let g = textureSample(scene_texture, scene_sampler, uv).g;
        let b = textureSample(scene_texture, scene_sampler, uv - dir * ca_offset.x).b;
        color = vec3<f32>(r, g, b);
    } else {
        color = textureSample(scene_texture, scene_sampler, uv).rgb;
    }

    // ── Scanlines ──
    if pp.scanlines > 0.0 {
        let y_pixel = uv.y * resolution.y;
        let scanline = sin(y_pixel * 3.14159265) * 0.5 + 0.5;
        color = color * (1.0 - pp.scanlines * (1.0 - scanline));
    }

    // ── Vignette ──
    if pp.vignette > 0.0 {
        let centered = uv - 0.5;
        let dist = dot(centered, centered);
        let vig = 1.0 - dist * pp.vignette * 4.0;
        color = color * clamp(vig, 0.0, 1.0);
    }

    // ── Bloom composite ──
    if pp.bloom_intensity > 0.0 {
        let bloom_sample = textureSample(bloom_texture, scene_sampler, uv).rgb;
        color = color + bloom_sample * pp.bloom_intensity;
    }

    // ── Film grain ──
    if pp.grain > 0.0 {
        let noise = hash12(uv * resolution + vec2<f32>(pp.time * 1000.0, 0.0));
        color = color + (noise - 0.5) * pp.grain * 0.3;
    }

    // ── ACES tone mapping (HDR → display range) ──
    if pp.tone_map > 0.0 {
        color = aces_tonemap(color);
    }

    return vec4<f32>(color, 1.0);
}

// ACES filmic tone mapping curve.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}
";

/// Identity post-process fragment (passthrough, kept for testing).
pub const POST_PROCESS_IDENTITY_FRAGMENT: &str = r"
@fragment
fn fs_main(in: PostProcessOutput) -> @location(0) vec4<f32> {
    return textureSample(scene_texture, scene_sampler, in.uv);
}
";

/// Compose a complete WGSL shader module from user-supplied fragment body.
///
/// The user provides only the body of `@fragment fn fs_main(in: VertexOutput) ->
/// @location(0) vec4<f32>`. This function wraps it with the shared preamble
/// (struct definitions, resource bindings), the vertex shader, and the fragment
/// function signature.
///
/// # Example user shader (`crt.wgsl`)
///
/// ```wgsl
/// let uv = in.uv;
/// let color = textureSample(scene_texture, scene_sampler, uv);
/// return color;
/// ```
pub fn compose_user_shader(user_body: &str) -> String {
    format!(
        "{preamble}\n{vertex}\n\
         @fragment\n\
         fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{\n\
         {body}\n\
         }}\n",
        preamble = POST_PROCESS_PREAMBLE,
        vertex = POST_PROCESS_VERTEX,
        body = user_body,
    )
}

/// Pre-validate a user shader body by composing the full WGSL source and
/// parsing it with naga's WGSL front-end.
///
/// Returns `Ok(())` if the shader parses successfully, or an error description
/// on failure. This catches syntax errors before wgpu pipeline creation.
pub fn validate_user_shader(user_body: &str) -> Result<(), String> {
    let full_source = compose_user_shader(user_body);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_user_shader_valid_body() {
        let body = r#"
            let uv = in.uv;
            let color = textureSample(scene_texture, scene_sampler, uv);
            return color;
        "#;
        assert!(validate_user_shader(body).is_ok());
    }

    #[test]
    fn validate_user_shader_invalid_body() {
        let body = "this is not valid WGSL {{{";
        assert!(validate_user_shader(body).is_err());
    }

    #[test]
    fn compose_user_shader_contains_body() {
        let body = "return vec4<f32>(1.0);";
        let full = compose_user_shader(body);
        assert!(full.contains(body));
        assert!(full.contains("fs_main"));
        assert!(full.contains("vs_main"));
    }

    #[test]
    fn validate_user_shader_empty_body_fails_validation() {
        // An empty body parses but fails naga IR validation (missing return).
        let result = validate_user_shader("");
        assert!(
            result.is_err(),
            "empty body should fail validation (missing return)"
        );
    }

    #[test]
    fn validate_user_shader_syntax_error() {
        // Completely broken syntax that naga cannot parse.
        let body = "let x = ;; return !!!;";
        let result = validate_user_shader(body);
        assert!(result.is_err(), "syntax error should fail validation");
    }

    #[test]
    fn validate_user_shader_type_error() {
        // Body returns wrong type (i32 instead of vec4<f32>).
        let body = "return 42;";
        let result = validate_user_shader(body);
        assert!(result.is_err(), "wrong return type should fail validation");
    }

    #[test]
    fn post_process_params_size() {
        assert_eq!(
            std::mem::size_of::<PostProcessParams>(),
            32,
            "PostProcessParams must be exactly 32 bytes for GPU uniform"
        );
    }
}
