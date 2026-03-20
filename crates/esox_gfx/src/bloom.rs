//! Dual-Kawase bloom post-processing pass.
//!
//! Renders a multi-level downsample/upsample mip chain from the scene texture,
//! producing a soft glow around bright areas. The result texture is composited
//! into the final post-process pass via a bind group binding.

use crate::primitive::ShaderId;

/// Pipeline ID for the bloom downsample pass.
pub const PIPELINE_BLOOM_DOWNSAMPLE: ShaderId = ShaderId(101);

/// Pipeline ID for the bloom upsample pass.
pub const PIPELINE_BLOOM_UPSAMPLE: ShaderId = ShaderId(102);

/// Number of mip levels in the bloom chain.
const BLOOM_MIP_LEVELS: usize = 5;

/// GPU-side parameters for a single bloom blur pass (16 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BloomParams {
    /// Texel size of the source texture (1/width, 1/height).
    pub texel_size: [f32; 2],
    /// Luminance threshold for bloom extraction (only first downsample uses this).
    pub threshold: f32,
    /// Soft knee width around threshold (smooth transition, 0 = hard cutoff).
    pub soft_knee: f32,
}

/// A single mip level in the bloom chain.
struct BloomMip {
    /// The mip texture (kept alive so views remain valid).
    #[allow(dead_code)]
    texture: wgpu::Texture,
    /// View used as render attachment (write).
    render_view: wgpu::TextureView,
    /// View used for sampling (read).
    sample_view: wgpu::TextureView,
    /// Mip width in pixels.
    width: u32,
    /// Mip height in pixels.
    height: u32,
}

/// The bloom post-processing pass.
///
/// Owns a mip chain, bind group layout, per-level bind groups, and a params
/// buffer. Call [`encode`] between the scene pass and the post-process composite.
pub struct BloomPass {
    /// Mip chain (level 0 = half scene resolution, each subsequent = half again).
    mips: Vec<BloomMip>,
    /// Bind group layout shared by downsample and upsample shaders.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the scene texture → mip0 downsample (index 0),
    /// then mip[i] → mip[i+1] for subsequent downsamples.
    down_bind_groups: Vec<wgpu::BindGroup>,
    /// Bind group for mip[i+1] → mip[i] upsample passes.
    up_bind_groups: Vec<wgpu::BindGroup>,
    /// Linear sampler for bloom texture sampling.
    sampler: wgpu::Sampler,
    /// Per-pass uniform buffers for bloom parameters (one per downsample + one per upsample).
    /// Separate buffers avoid the issue where multiple `queue.write_buffer` calls to the
    /// same offset only preserve the last write.
    down_params_buffers: Vec<wgpu::Buffer>,
    up_params_buffers: Vec<wgpu::Buffer>,
    /// Scene texture dimensions used to detect resize.
    scene_width: u32,
    /// Scene texture dimensions used to detect resize.
    scene_height: u32,
    /// Surface texture format.
    format: wgpu::TextureFormat,
}

impl BloomPass {
    /// Create a new bloom pass sized for the given scene dimensions.
    pub fn new(
        device: &wgpu::Device,
        scene_width: u32,
        scene_height: u32,
        format: wgpu::TextureFormat,
        scene_sample_view: &wgpu::TextureView,
    ) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mips = Self::create_mip_chain(device, scene_width, scene_height, format);

        let down_params_buffers = Self::create_params_buffers(device, mips.len(), "bloom_down_params");
        let up_params_buffers = Self::create_params_buffers(device, mips.len().saturating_sub(1), "bloom_up_params");

        let down_bind_groups = Self::create_down_bind_groups(
            device,
            &bind_group_layout,
            scene_sample_view,
            &mips,
            &sampler,
            &down_params_buffers,
        );

        let up_bind_groups = Self::create_up_bind_groups(
            device,
            &bind_group_layout,
            &mips,
            &sampler,
            &up_params_buffers,
        );

        Self {
            mips,
            bind_group_layout,
            down_bind_groups,
            up_bind_groups,
            sampler,
            down_params_buffers,
            up_params_buffers,
            scene_width,
            scene_height,
            format,
        }
    }

    /// Recreate mip textures if scene dimensions changed. Returns `true` if recreated.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        scene_width: u32,
        scene_height: u32,
        scene_sample_view: &wgpu::TextureView,
    ) -> bool {
        if self.scene_width == scene_width && self.scene_height == scene_height {
            return false;
        }
        self.scene_width = scene_width;
        self.scene_height = scene_height;
        self.mips = Self::create_mip_chain(device, scene_width, scene_height, self.format);
        self.down_params_buffers = Self::create_params_buffers(device, self.mips.len(), "bloom_down_params");
        self.up_params_buffers = Self::create_params_buffers(device, self.mips.len().saturating_sub(1), "bloom_up_params");
        self.down_bind_groups = Self::create_down_bind_groups(
            device,
            &self.bind_group_layout,
            scene_sample_view,
            &self.mips,
            &self.sampler,
            &self.down_params_buffers,
        );
        self.up_bind_groups = Self::create_up_bind_groups(
            device,
            &self.bind_group_layout,
            &self.mips,
            &self.sampler,
            &self.up_params_buffers,
        );
        true
    }

    /// Rebuild bind group 0 when the offscreen target is recreated.
    pub fn update_scene_texture(
        &mut self,
        device: &wgpu::Device,
        scene_sample_view: &wgpu::TextureView,
    ) {
        self.down_bind_groups = Self::create_down_bind_groups(
            device,
            &self.bind_group_layout,
            scene_sample_view,
            &self.mips,
            &self.sampler,
            &self.down_params_buffers,
        );
    }

    /// Encode the bloom downsample and upsample passes into the command encoder.
    ///
    /// `threshold` is the HDR luminance cutoff — only pixels brighter than this
    /// contribute to bloom. `soft_knee` controls the width of the soft transition
    /// around the threshold (0 = hard cutoff, 0.5 = smooth).
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        downsample_pipeline: &wgpu::RenderPipeline,
        upsample_pipeline: &wgpu::RenderPipeline,
        threshold: f32,
        soft_knee: f32,
    ) {
        let mip_count = self.mips.len();
        if mip_count == 0 {
            return;
        }

        // ── Downsample chain: scene → mip0 → mip1 → ... → mip[N-1] ──
        // Each pass writes to its own params buffer to avoid write_buffer coalescing.
        for i in 0..mip_count {
            let (src_w, src_h) = if i == 0 {
                (self.scene_width, self.scene_height)
            } else {
                (self.mips[i - 1].width, self.mips[i - 1].height)
            };

            let params = BloomParams {
                texel_size: [1.0 / src_w.max(1) as f32, 1.0 / src_h.max(1) as f32],
                threshold: if i == 0 { threshold } else { 0.0 },
                soft_knee: if i == 0 { soft_knee } else { 0.0 },
            };
            queue.write_buffer(&self.down_params_buffers[i], 0, bytemuck::bytes_of(&params));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mips[i].render_view,
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

            pass.set_pipeline(downsample_pipeline);
            pass.set_bind_group(0, Some(&self.down_bind_groups[i]), &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Upsample chain: mip[N-1] → mip[N-2] → ... → mip0 ──
        for i in (0..mip_count - 1).rev() {
            let src_w = self.mips[i + 1].width;
            let src_h = self.mips[i + 1].height;

            let params = BloomParams {
                texel_size: [1.0 / src_w.max(1) as f32, 1.0 / src_h.max(1) as f32],
                threshold: 0.0,
                soft_knee: 0.0,
            };
            queue.write_buffer(&self.up_params_buffers[i], 0, bytemuck::bytes_of(&params));

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_upsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.mips[i].render_view,
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

            pass.set_pipeline(upsample_pipeline);
            pass.set_bind_group(0, Some(&self.up_bind_groups[i]), &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Returns the bloom result texture view (mip0) for compositing.
    pub fn result_view(&self) -> &wgpu::TextureView {
        &self.mips[0].sample_view
    }

    /// Returns the bind group layout used by bloom shaders.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create the bind group layout for bloom shaders.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_bind_group_layout"),
            entries: &[
                // binding 0: source texture
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
                // binding 1: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 2: bloom params uniform
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
            ],
        })
    }

    /// Create the mip chain textures.
    fn create_mip_chain(
        device: &wgpu::Device,
        scene_width: u32,
        scene_height: u32,
        format: wgpu::TextureFormat,
    ) -> Vec<BloomMip> {
        let mut mips = Vec::with_capacity(BLOOM_MIP_LEVELS);
        let mut w = (scene_width / 2).max(1);
        let mut h = (scene_height / 2).max(1);

        for i in 0..BLOOM_MIP_LEVELS {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("bloom_mip_{i}")),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let render_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let sample_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            mips.push(BloomMip {
                texture,
                render_view,
                sample_view,
                width: w,
                height: h,
            });

            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }

        mips
    }

    /// Create per-pass params buffers.
    fn create_params_buffers(device: &wgpu::Device, count: usize, label: &str) -> Vec<wgpu::Buffer> {
        (0..count)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{label}_{i}")),
                    size: size_of::<BloomParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect()
    }

    /// Create downsample bind groups: scene → mip0, mip0 → mip1, ...
    fn create_down_bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        scene_view: &wgpu::TextureView,
        mips: &[BloomMip],
        sampler: &wgpu::Sampler,
        params_buffers: &[wgpu::Buffer],
    ) -> Vec<wgpu::BindGroup> {
        let mut groups = Vec::with_capacity(mips.len());

        for i in 0..mips.len() {
            let src_view = if i == 0 {
                scene_view
            } else {
                &mips[i - 1].sample_view
            };

            groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("bloom_down_bg_{i}")),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffers[i].as_entire_binding(),
                    },
                ],
            }));
        }

        groups
    }

    /// Create upsample bind groups: mip[i+1] → mip[i].
    fn create_up_bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        mips: &[BloomMip],
        sampler: &wgpu::Sampler,
        params_buffers: &[wgpu::Buffer],
    ) -> Vec<wgpu::BindGroup> {
        let mut groups = Vec::with_capacity(mips.len().saturating_sub(1));

        for i in 0..mips.len().saturating_sub(1) {
            let src_view = &mips[i + 1].sample_view;

            groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("bloom_up_bg_{i}")),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffers[i].as_entire_binding(),
                    },
                ],
            }));
        }

        groups
    }
}

/// Fullscreen triangle vertex shader shared by bloom passes.
pub const BLOOM_VERTEX_SOURCE: &str = r"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
";

/// Bloom bind group preamble shared by downsample and upsample shaders.
const BLOOM_BIND_GROUP_PREAMBLE: &str = r"
struct BloomParams {
    texel_size: vec2<f32>,
    threshold: f32,
    soft_knee: f32,
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BloomParams;
";

/// Dual-Kawase downsample fragment shader.
///
/// 5-tap kernel: center sample weighted ×4, plus 4 diagonal samples.
/// On the first downsample (threshold > 0), applies a soft brightness prefilter
/// so only HDR-bright pixels contribute to bloom.
pub const BLOOM_DOWNSAMPLE_FRAGMENT: &str = r"
// Clamp HDR sample to prevent Inf/NaN from entering the bloom chain.
fn safe_hdr(s: vec4<f32>) -> vec4<f32> {
    return clamp(s, vec4<f32>(0.0), vec4<f32>(65000.0));
}

// Luminance helper.
fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Karis average weight: suppresses extremely bright pixels to prevent fireflies.
fn karis_weight(color: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luma(color));
}

// Soft brightness thresholding: smoothly ramps from 0 at (threshold - knee)
// to full brightness at (threshold + knee). Returns the color contribution.
fn prefilter(color: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let brightness = luma(color);
    // Soft knee curve: quadratic ramp in the transition region.
    let soft = brightness - threshold + knee;
    let soft_clamped = clamp(soft, 0.0, 2.0 * knee);
    var contribution = soft_clamped * soft_clamped / (4.0 * knee + 0.00001);
    // Above threshold, use full excess brightness.
    contribution = max(contribution, brightness - threshold);
    contribution = max(contribution, 0.0);
    return color * (contribution / max(brightness, 0.00001));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let ts = params.texel_size;

    var center = safe_hdr(textureSample(src_texture, src_sampler, uv));
    var tl = safe_hdr(textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x, -ts.y)));
    var tr = safe_hdr(textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x, -ts.y)));
    var bl = safe_hdr(textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x,  ts.y)));
    var br = safe_hdr(textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x,  ts.y)));

    // Apply brightness threshold on first downsample only (threshold > 0).
    if params.threshold > 0.0 {
        center = vec4<f32>(prefilter(center.rgb, params.threshold, params.soft_knee), center.a);
        tl = vec4<f32>(prefilter(tl.rgb, params.threshold, params.soft_knee), tl.a);
        tr = vec4<f32>(prefilter(tr.rgb, params.threshold, params.soft_knee), tr.a);
        bl = vec4<f32>(prefilter(bl.rgb, params.threshold, params.soft_knee), bl.a);
        br = vec4<f32>(prefilter(br.rgb, params.threshold, params.soft_knee), br.a);

        // Karis average: weight each sample by 1/(1+luma) to suppress fireflies.
        let wc = karis_weight(center.rgb);
        let wtl = karis_weight(tl.rgb);
        let wtr = karis_weight(tr.rgb);
        let wbl = karis_weight(bl.rgb);
        let wbr = karis_weight(br.rgb);
        let total = wc * 4.0 + wtl + wtr + wbl + wbr;
        return (center * wc * 4.0 + tl * wtl + tr * wtr + bl * wbl + br * wbr) / total;
    }

    return (center * 4.0 + tl + tr + bl + br) / 8.0;
}
";

/// Dual-Kawase upsample fragment shader.
///
/// 8-tap diamond+cross kernel for smooth upsampling. The pipeline uses
/// additive blend state (One + One) so the result accumulates onto the
/// destination mip.
pub const BLOOM_UPSAMPLE_FRAGMENT: &str = r"
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let ts = params.texel_size;

    // 8-tap diamond + cross pattern.
    var result = textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x,  0.0))  * 2.0;
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x,  0.0))  * 2.0;
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0, -ts.y))  * 2.0;
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,  ts.y))  * 2.0;
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x, -ts.y));
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x, -ts.y));
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>(-ts.x,  ts.y));
    result    += textureSample(src_texture, src_sampler, uv + vec2<f32>( ts.x,  ts.y));

    return result / 12.0;
}
";

/// Compose the full downsample shader source.
pub fn downsample_shader_source() -> String {
    format!("{BLOOM_VERTEX_SOURCE}\n{BLOOM_BIND_GROUP_PREAMBLE}\n{BLOOM_DOWNSAMPLE_FRAGMENT}")
}

/// Compose the full upsample shader source.
pub fn upsample_shader_source() -> String {
    format!("{BLOOM_VERTEX_SOURCE}\n{BLOOM_BIND_GROUP_PREAMBLE}\n{BLOOM_UPSAMPLE_FRAGMENT}")
}

/// Create a 1×1 black texture for use as a placeholder bloom texture.
///
/// When bloom is disabled, the post-process bind group still needs all slots
/// filled. This provides a zero-value texture so the composite is a no-op.
pub fn create_black_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom_black_placeholder"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Write zeroes — format-appropriate size per texel.
    let bpp = format.block_copy_size(None).unwrap_or(4);
    let zeros = vec![0u8; bpp as usize];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &zeros,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bpp),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloom_params_size() {
        assert_eq!(
            std::mem::size_of::<BloomParams>(),
            16,
            "BloomParams must be exactly 16 bytes for GPU uniform"
        );
    }

    #[test]
    fn downsample_shader_parses() {
        let source = downsample_shader_source();
        let module = naga::front::wgsl::parse_str(&source).expect("downsample shader should parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("downsample shader should validate");
    }

    #[test]
    fn upsample_shader_parses() {
        let source = upsample_shader_source();
        let module = naga::front::wgsl::parse_str(&source).expect("upsample shader should parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("upsample shader should validate");
    }
}
