use crate::color::Color;
use crate::pipeline::{GpuContext, PipelineRegistry, RenderResources};
use crate::primitive::{BlendMode, QuadInstance, ShapeType};
use crate::scene::Scene;
use crate::shape::primitive_to_instance;

/// Maximum number of quad instances per frame (prevents GPU OOM from degenerate scenes).
const MAX_INSTANCES: u32 = 2_000_000;

/// Rendering phase, ordered by draw priority.
///
/// Phases control pipeline selection (opaque vs blended) and enforce
/// batch boundaries so the GPU can skip blend hardware for opaque
/// backgrounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum RenderPhase {
    /// Cell backgrounds — opaque, no blend needed.
    OpaqueBackground = 0,
    /// Glyph quads — alpha-blended text.
    Text = 1,
    /// Underlines, cursor, selection, images — alpha-blended decorations.
    Decoration = 2,
    /// Splash, settings, tab bar, scrollbar, borders — top-layer overlays.
    Overlay = 3,
}

/// A contiguous range of instances belonging to the same render phase.
#[derive(Debug, Clone)]
pub struct PhaseRange {
    /// Which phase this range belongs to.
    pub phase: RenderPhase,
    /// Index of the first instance in the instance buffer.
    pub first_instance: u32,
    /// Number of instances in this range.
    pub instance_count: u32,
}

/// Quantized clip rect for batching (integer pixels for reliable Eq/Hash).
///
/// All-zeros represents "full viewport" (no scissor needed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClipKey {
    /// X pixel offset.
    pub x: u32,
    /// Y pixel offset.
    pub y: u32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
}

impl ClipKey {
    /// Sentinel value representing the full viewport (no scissor restriction).
    pub const FULL_VIEWPORT: Self = Self {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };

    /// Quantize a floating-point clip rect to integer pixels.
    ///
    /// Returns [`FULL_VIEWPORT`](Self::FULL_VIEWPORT) when the clip rect is
    /// all zeros (the "no clip" sentinel).
    pub fn from_clip_rect(clip: [f32; 4]) -> Self {
        if clip[2] <= 0.0 && clip[3] <= 0.0 {
            return Self::FULL_VIEWPORT;
        }
        let max = u32::MAX as f32;
        Self {
            x: clip[0].max(0.0).min(max).floor() as u32,
            y: clip[1].max(0.0).min(max).floor() as u32,
            w: clip[2].max(0.0).min(max).ceil() as u32,
            h: clip[3].max(0.0).min(max).ceil() as u32,
        }
    }

    /// Whether this key represents the full viewport (no scissor).
    pub fn is_full_viewport(self) -> bool {
        self == Self::FULL_VIEWPORT
    }
}

/// A contiguous range of instances sharing the same pipeline + scissor.
#[derive(Debug, Clone)]
pub struct DrawBatch {
    /// Which pipeline to bind (0 = built-in quad pipeline).
    pub pipeline_id: u32,
    /// Clip rect for this batch.
    pub clip_key: ClipKey,
    /// Index of the first instance in the instance buffer.
    pub first_instance: u32,
    /// Number of instances in this batch.
    pub instance_count: u32,
    /// Render phase this batch belongs to (used for pipeline remapping).
    pub phase: RenderPhase,
}

/// Per-frame uniform data uploaded to the GPU (32 bytes).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FrameUniforms {
    /// Viewport: [width, height, 1/width, 1/height].
    pub viewport: [f32; 4],
    /// Time: [time_secs, delta_time, frame_number, 0].
    pub time: [f32; 4],
}

/// Collects quad instances for a single frame.
pub struct Frame {
    instances: Vec<QuadInstance>,
    batches: Vec<DrawBatch>,
    /// Reusable buffer for resolved primitives (avoids per-frame allocation).
    resolved_buf: Vec<crate::scene::ResolvedPrimitive>,
    /// Whether multi-phase rendering is enabled for this frame.
    multi_phase: bool,
    /// The currently active render phase (set by `begin_phase()`).
    current_phase: RenderPhase,
    /// Closed phase ranges (each records phase + instance span).
    phase_ranges: Vec<PhaseRange>,
    /// Instance index where the current phase started.
    phase_start: u32,
    /// When set, `push()` and `extend_instances()` stamp this clip rect onto every instance.
    active_clip: Option<[f32; 4]>,
    /// Tile buckets for partial redraw. When `Some`, `push()` routes instances
    /// to overlapping tiles by their position rect.
    tile_buckets: Option<Vec<Vec<QuadInstance>>>,
    /// Overlay instances that bypass tile routing (modals, toasts, tooltips).
    overlay_instances: Vec<QuadInstance>,
    /// Whether subsequent pushes should bypass tile routing.
    overlay_mode: bool,
    /// Reference to the tile grid (column count) for routing.
    tile_cols: u16,
    tile_rows: u16,
}

impl Frame {
    /// Create an empty frame.
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            batches: Vec::new(),
            resolved_buf: Vec::new(),
            multi_phase: false,
            current_phase: RenderPhase::OpaqueBackground,
            phase_ranges: Vec::new(),
            phase_start: 0,
            active_clip: None,
            tile_buckets: None,
            overlay_instances: Vec::new(),
            overlay_mode: false,
            tile_cols: 0,
            tile_rows: 0,
        }
    }

    /// Build the instance buffer from the current scene state.
    pub fn build_from_scene(&mut self, scene: &Scene) {
        self.instances.clear();
        self.batches.clear();
        scene.collect_primitives_into(&mut self.resolved_buf);
        for rp in &self.resolved_buf {
            self.instances
                .push(primitive_to_instance(&rp.primitive, rp.opacity, rp.clip));
        }
    }

    /// Push a single quad instance.
    ///
    /// If an active clip is set, the instance's `clip_rect` is overwritten.
    /// When tile buckets are active, routes the instance to overlapping tiles
    /// (or the overlay list if overlay mode is on).
    pub fn push(&mut self, mut instance: QuadInstance) {
        if let Some(clip) = self.active_clip {
            instance.clip_rect = clip;
        }
        if self.overlay_mode {
            self.overlay_instances.push(instance);
        } else if let Some(ref mut buckets) = self.tile_buckets {
            // Route to tile buckets by instance rect.
            let x = instance.rect[0];
            let y = instance.rect[1];
            let w = instance.rect[2];
            let h = instance.rect[3];
            let tile_size = crate::damage::TILE_SIZE;
            let col_start = (x.max(0.0) as u32 / tile_size).min(self.tile_cols.saturating_sub(1) as u32) as u16;
            let col_end = (((x + w).ceil() as u32 + tile_size - 1) / tile_size).min(self.tile_cols as u32) as u16;
            let row_start = (y.max(0.0) as u32 / tile_size).min(self.tile_rows.saturating_sub(1) as u32) as u16;
            let row_end = (((y + h).ceil() as u32 + tile_size - 1) / tile_size).min(self.tile_rows as u32) as u16;
            for row in row_start..row_end {
                for col in col_start..col_end {
                    let idx = row as usize * self.tile_cols as usize + col as usize;
                    if idx < buckets.len() {
                        buckets[idx].push(instance);
                    }
                }
            }
        } else {
            self.instances.push(instance);
        }
    }

    /// Number of instances to draw.
    pub fn instance_count(&self) -> u32 {
        self.instances.len() as u32
    }

    /// Number of instances currently collected (as `usize`).
    pub fn instance_len(&self) -> usize {
        self.instances.len()
    }

    /// Truncate the instance buffer to `len`, discarding instances added after that point.
    /// Used by `measure()` to discard off-screen rendering. No-op if `len >= instance_len()`.
    pub fn truncate_instances(&mut self, len: usize) {
        if len < self.instances.len() {
            self.instances.truncate(len);
        }
    }

    /// Replace an existing instance at `index` (e.g. to backfill a placeholder).
    ///
    /// If an active clip is set, it is stamped onto the instance.
    /// Out-of-bounds indices are silently ignored.
    pub fn replace_instance(&mut self, index: usize, mut inst: QuadInstance) {
        if let Some(clip) = self.active_clip {
            inst.clip_rect = clip;
        }
        if index < self.instances.len() {
            self.instances[index] = inst;
        }
    }

    /// Translate an existing instance by (dx, dy). Out-of-bounds indices are silently ignored.
    pub fn translate_instance(&mut self, index: usize, dx: f32, dy: f32) {
        if let Some(inst) = self.instances.get_mut(index) {
            inst.rect[0] += dx;
            inst.rect[1] += dy;
        }
    }

    /// Scale the x-position of an instance relative to an origin.
    /// new_x = origin + (old_x - origin) * scale. Out-of-bounds indices are silently ignored.
    pub fn offset_instance_x(&mut self, index: usize, origin: f32, scale: f32) {
        if let Some(inst) = self.instances.get_mut(index) {
            inst.rect[0] = origin + (inst.rect[0] - origin) * scale;
        }
    }

    /// Append a slice of pre-built instances (e.g. from a cache).
    ///
    /// If an active clip is set, each instance's `clip_rect` is overwritten.
    pub fn extend_instances(&mut self, instances: &[QuadInstance]) {
        if let Some(clip) = self.active_clip {
            self.instances.extend(instances.iter().map(|inst| {
                let mut inst = *inst;
                inst.clip_rect = clip;
                inst
            }));
        } else {
            self.instances.extend_from_slice(instances);
        }
    }

    /// Clear all instances and reset phase state.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.batches.clear();
        self.multi_phase = false;
        self.current_phase = RenderPhase::OpaqueBackground;
        self.phase_ranges.clear();
        self.phase_start = 0;
        self.active_clip = None;
        self.tile_buckets = None;
        self.overlay_instances.clear();
        self.overlay_mode = false;
    }

    /// Begin tile-based partial redraw. Allocates tile buckets matching the
    /// grid dimensions. Call before pushing any instances for this frame.
    pub fn begin_partial(&mut self, grid: &crate::damage::TileGrid) {
        let count = grid.tile_count();
        let mut buckets = Vec::with_capacity(count);
        buckets.resize_with(count, Vec::new);
        self.tile_buckets = Some(buckets);
        self.tile_cols = grid.cols();
        self.tile_rows = grid.rows();
        self.overlay_instances.clear();
        self.overlay_mode = false;
    }

    /// Push an instance that bypasses tile routing (for overlays: modals, toasts, tooltips).
    pub fn push_overlay(&mut self, mut instance: QuadInstance) {
        if let Some(clip) = self.active_clip {
            instance.clip_rect = clip;
        }
        self.overlay_instances.push(instance);
    }

    /// Enable or disable overlay mode. When enabled, all `push()` calls
    /// route to the overlay list (bypassing tile buckets).
    pub fn set_overlay_mode(&mut self, on: bool) {
        self.overlay_mode = on;
    }

    /// Whether overlay mode is currently active.
    pub fn overlay_mode(&self) -> bool {
        self.overlay_mode
    }

    /// Finalize partial redraw: merge dirty/clean tiles and overlay into
    /// the main instance list. Must be called before `build_batches()`.
    pub fn finalize_partial(&mut self, grid: &mut crate::damage::TileGrid) {
        if let Some(ref mut buckets) = self.tile_buckets {
            let merged = grid.finalize(buckets, &self.overlay_instances);
            self.instances = merged;
            self.overlay_instances.clear();
        }
        self.tile_buckets = None;
        self.overlay_mode = false;
    }

    /// Set the active clip rect. All subsequent `push()` / `extend_instances()` calls
    /// will stamp this clip onto every instance. Pass `None` to disable.
    pub fn set_active_clip(&mut self, clip: Option<[f32; 4]>) {
        self.active_clip = clip;
    }

    /// Get the current active clip rect.
    pub fn active_clip(&self) -> Option<[f32; 4]> {
        self.active_clip
    }

    /// Get the instance data as a byte slice (for uploading to GPU).
    pub fn instance_data(&self) -> &[QuadInstance] {
        &self.instances
    }

    /// Get the computed draw batches.
    pub fn batches(&self) -> &[DrawBatch] {
        &self.batches
    }

    /// Number of draw batches (available after `build_batches()`).
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Enable multi-phase rendering for this frame.
    ///
    /// Must be called before pushing any instances. When enabled, phase
    /// boundaries force batch breaks and the `OpaqueBackground` phase uses
    /// the opaque (no-blend) pipeline.
    pub fn enable_multi_phase(&mut self) {
        self.multi_phase = true;
        self.current_phase = RenderPhase::OpaqueBackground;
        self.phase_ranges.clear();
        self.phase_start = 0;
    }

    /// Whether multi-phase rendering is active for this frame.
    pub fn is_multi_phase(&self) -> bool {
        self.multi_phase
    }

    /// Transition to a new render phase.
    ///
    /// Closes the current phase range (if any instances were pushed) and
    /// begins a new one. No-op if `phase` equals the current phase.
    pub fn begin_phase(&mut self, phase: RenderPhase) {
        if !self.multi_phase || phase == self.current_phase {
            return;
        }
        let count = self.instances.len() as u32;
        let range_count = count - self.phase_start;
        if range_count > 0 {
            self.phase_ranges.push(PhaseRange {
                phase: self.current_phase,
                first_instance: self.phase_start,
                instance_count: range_count,
            });
        }
        self.current_phase = phase;
        self.phase_start = count;
    }

    /// Finalize phase ranges (closes the last open range).
    ///
    /// Called automatically by `build_batches()` when multi-phase is active.
    pub fn finalize_phases(&mut self) {
        if !self.multi_phase {
            return;
        }
        let count = self.instances.len() as u32;
        let range_count = count - self.phase_start;
        if range_count > 0 {
            self.phase_ranges.push(PhaseRange {
                phase: self.current_phase,
                first_instance: self.phase_start,
                instance_count: range_count,
            });
            self.phase_start = count;
        }
    }

    /// Get the finalized phase ranges.
    pub fn phase_ranges(&self) -> &[PhaseRange] {
        &self.phase_ranges
    }

    /// Extract the pipeline ID for an instance at the given index.
    ///
    /// Returns the shader pipeline ID from `flags[3]` for `ShapeType::Shader`
    /// instances, the blend-mode variant pipeline for SDF 2D shapes, or the
    /// shape type's built-in pipeline ID for everything else.
    fn pipeline_id_of(&self, idx: usize) -> u32 {
        let inst = &self.instances[idx];
        match ShapeType::from_f32(inst.flags[0]) {
            Some(ShapeType::Shader) => inst.flags[3] as u32,
            Some(shape) => {
                let base = shape.pipeline_id();
                if base == crate::primitive::PIPELINE_SDF_2D.0 {
                    // Bits 1–2 of flags[3] encode the blend mode.
                    let bits = inst.flags[3] as u32;
                    let blend = BlendMode::from_bits(((bits >> 1) & 0x3) as u8);
                    blend.sdf_pipeline_id()
                } else {
                    base
                }
            }
            None => 0,
        }
    }

    /// Build draw batches from the current instance list.
    ///
    /// When multi-phase is active, iterates phase ranges and builds batches
    /// within each range, forcing batch breaks at phase boundaries. Each
    /// batch is tagged with its render phase for pipeline remapping.
    ///
    /// When multi-phase is off, performs a single linear scan (legacy path)
    /// with all batches tagged as `OpaqueBackground` (no remapping occurs).
    pub fn build_batches(&mut self) {
        self.batches.clear();
        let count = self.instances.len();
        if count == 0 {
            return;
        }

        if self.multi_phase {
            self.finalize_phases();
            for range in 0..self.phase_ranges.len() {
                let phase = self.phase_ranges[range].phase;
                let start = self.phase_ranges[range].first_instance as usize;
                let end = start + self.phase_ranges[range].instance_count as usize;
                if start >= end {
                    continue;
                }
                self.build_batches_for_range(start, end, phase);
            }
        } else {
            self.build_batches_for_range(0, count, RenderPhase::OpaqueBackground);
        }
    }

    /// Build batches for a contiguous range of instances with the given phase.
    fn build_batches_for_range(&mut self, start: usize, end: usize, phase: RenderPhase) {
        if start >= end {
            return;
        }

        let mut batch_start = start as u32;
        let mut cur_pipeline = self.pipeline_id_of(start);
        let mut cur_clip = ClipKey::from_clip_rect(self.instances[start].clip_rect);

        for i in (start + 1)..end {
            let pip = self.pipeline_id_of(i);
            let clip = ClipKey::from_clip_rect(self.instances[i].clip_rect);
            if pip != cur_pipeline || clip != cur_clip {
                self.batches.push(DrawBatch {
                    pipeline_id: cur_pipeline,
                    clip_key: cur_clip,
                    first_instance: batch_start,
                    instance_count: i as u32 - batch_start,
                    phase,
                });
                batch_start = i as u32;
                cur_pipeline = pip;
                cur_clip = clip;
            }
        }

        // Final batch in this range.
        self.batches.push(DrawBatch {
            pipeline_id: cur_pipeline,
            clip_key: cur_clip,
            first_instance: batch_start,
            instance_count: end as u32 - batch_start,
            phase,
        });
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// A pre-acquired surface texture and view, for sharing between render passes.
pub struct SurfaceFrame {
    /// The surface texture (must be presented after all passes complete).
    pub texture: wgpu::SurfaceTexture,
    /// View into the surface texture.
    pub view: wgpu::TextureView,
}

/// Controls whether the 2D render pass clears or loads the color target.
pub enum ColorLoadOp {
    /// Clear the target to the given color (normal 2D-only path).
    Clear(wgpu::Color),
    /// Load existing content (preserves 3D content rendered earlier).
    Load,
}

/// Optional offscreen rendering configuration for post-processing.
pub struct PostProcessPass<'a> {
    /// The offscreen target to render the scene into.
    pub offscreen: &'a crate::offscreen::OffscreenTarget,
    /// The post-process pipeline to apply.
    pub pipeline_id: crate::primitive::ShaderId,
    /// Optional bloom pass to run between scene and composite.
    pub bloom: Option<&'a crate::bloom::BloomPass>,
}

/// Encodes and submits a frame's render pass to the GPU.
pub struct FrameEncoder;

impl FrameEncoder {
    /// Encode the frame and submit it to the GPU.
    ///
    /// When `post_process` is `Some`, the scene is rendered to the offscreen
    /// target first, then a fullscreen triangle applies the post-process
    /// shader to present the result. When `None`, the scene renders directly
    /// to the surface (zero overhead).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit(
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        uniforms: &FrameUniforms,
        bg_color: &Color,
        registry: &PipelineRegistry,
        msaa_view: Option<&wgpu::TextureView>,
        depth_view: Option<&wgpu::TextureView>,
    ) -> Result<(), crate::error::Error> {
        Self::encode_and_submit_inner(
            gpu, resources, frame, uniforms, bg_color, registry, None, msaa_view, depth_view,
        )
    }

    /// Encode the frame with an optional post-processing pass.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_with_post_process(
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        uniforms: &FrameUniforms,
        bg_color: &Color,
        registry: &PipelineRegistry,
        post_process: Option<PostProcessPass<'_>>,
        msaa_view: Option<&wgpu::TextureView>,
        depth_view: Option<&wgpu::TextureView>,
    ) -> Result<(), crate::error::Error> {
        Self::encode_and_submit_inner(
            gpu,
            resources,
            frame,
            uniforms,
            bg_color,
            registry,
            post_process,
            msaa_view,
            depth_view,
        )
    }

    /// Encode and submit using a pre-acquired surface with a custom load op.
    ///
    /// When `color_load_op` is `ColorLoadOp::Load`, existing surface content
    /// (e.g. from a prior 3D pass) is preserved. MSAA is skipped for `Load`
    /// to avoid clobbering the 3D content during resolve.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_with_surface(
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        uniforms: &FrameUniforms,
        registry: &PipelineRegistry,
        surface: SurfaceFrame,
        color_load_op: ColorLoadOp,
        post_process: Option<PostProcessPass<'_>>,
        msaa_view: Option<&wgpu::TextureView>,
        depth_view: Option<&wgpu::TextureView>,
        screenshot: Option<&crate::screenshot::ScreenshotCapture>,
    ) -> Result<(), crate::error::Error> {
        Self::encode_and_submit_surface_inner(
            gpu,
            resources,
            frame,
            uniforms,
            registry,
            surface,
            color_load_op,
            post_process,
            msaa_view,
            depth_view,
            screenshot,
        )
    }

    /// Internal implementation supporting optional post-processing.
    #[allow(clippy::too_many_arguments)]
    fn encode_and_submit_inner(
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        uniforms: &FrameUniforms,
        bg_color: &Color,
        registry: &PipelineRegistry,
        post_process: Option<PostProcessPass<'_>>,
        msaa_view: Option<&wgpu::TextureView>,
        depth_view: Option<&wgpu::TextureView>,
    ) -> Result<(), crate::error::Error> {
        let instance_count = frame.instance_count();

        // Upload uniforms (32 bytes — too small to benefit from staging belt).
        gpu.queue
            .write_buffer(&resources.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

        // Cap instance count to prevent GPU OOM.
        let instance_count = if instance_count > MAX_INSTANCES {
            tracing::warn!(
                count = instance_count,
                max = MAX_INSTANCES,
                "instance count exceeds maximum, clamping"
            );
            MAX_INSTANCES
        } else {
            instance_count
        };

        // Flip to the other instance buffer (double-buffered ping-pong).
        resources.flip_instance_buffer();

        // Resize instance buffer if needed (grow).
        let cur_cap = resources.current_instance_capacity();
        if instance_count > cur_cap {
            let new_cap = instance_count.saturating_mul(2).clamp(256, MAX_INSTANCES);
            resources.resize_current_buffer(&gpu.device, new_cap);
            resources.underutilization_frames = 0;
        } else if instance_count < cur_cap / 4 && cur_cap > 256 {
            // Shrink hysteresis: only shrink after 60 consecutive underutilized frames.
            resources.underutilization_frames += 1;
            if resources.underutilization_frames >= 60 {
                let new_cap = (cur_cap / 2).max(256);
                resources.resize_current_buffer(&gpu.device, new_cap);
                resources.underutilization_frames = 0;
            }
        } else {
            resources.underutilization_frames = 0;
        }

        // Get current surface texture.
        let output = gpu.surface.get_current_texture()?;
        let surface_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Determine the render target: offscreen texture or surface directly.
        let scene_target = match &post_process {
            Some(pp) => &pp.offscreen.render_view,
            None => &surface_view,
        };

        // Upload instance data directly (wgpu handles internal staging).
        if instance_count > 0 {
            let data = bytemuck::cast_slice(frame.instance_data());
            gpu.queue
                .write_buffer(resources.current_instance_buffer(), 0, data);
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_encoder"),
            });

        // ── Pass 1: Scene rendering ──
        {
            // When MSAA is active, render into the multisampled texture and
            // let the hardware resolve into scene_target (surface or offscreen).
            let (view, resolve) = match msaa_view {
                Some(mv) => (mv, Some(scene_target)),
                None => (scene_target, None),
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quad_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg_color.r as f64,
                            g: bg_color.g as f64,
                            b: bg_color.b as f64,
                            a: bg_color.a as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: depth_view.map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Store,
                        }),
                    }
                }),
                ..Default::default()
            });

            if instance_count > 0 {
                // Build draw batches for scissor + pipeline switching.
                frame.build_batches();

                pass.set_bind_group(0, Some(&resources.bind_group), &[]);
                pass.set_vertex_buffer(0, resources.quad_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, resources.current_instance_buffer().slice(..));

                let vp_w = gpu.config.width;
                let vp_h = gpu.config.height;
                let mut current_pipeline_id: Option<u32> = None;

                for batch in frame.batches() {
                    // Remap opaque-phase SDF batches to the no-blend pipeline.
                    let pip_id = if batch.phase == RenderPhase::OpaqueBackground
                        && batch.pipeline_id == crate::primitive::PIPELINE_SDF_2D.0
                    {
                        crate::primitive::PIPELINE_SDF_2D_OPAQUE.0
                    } else {
                        batch.pipeline_id
                    };
                    if current_pipeline_id != Some(pip_id) {
                        let sid = crate::primitive::ShaderId(pip_id);
                        let pipeline = match registry.get(sid) {
                            Some(handle) => &handle.pipeline,
                            None => {
                                tracing::warn!(
                                    pipeline_id = pip_id,
                                    "skipping batch: pipeline not found"
                                );
                                continue;
                            }
                        };
                        pass.set_pipeline(pipeline);
                        current_pipeline_id = Some(pip_id);
                    }

                    // Set scissor rect.
                    if batch.clip_key.is_full_viewport() {
                        pass.set_scissor_rect(0, 0, vp_w, vp_h);
                    } else {
                        let x = batch.clip_key.x.min(vp_w);
                        let y = batch.clip_key.y.min(vp_h);
                        let w = batch.clip_key.w.min(vp_w.saturating_sub(x));
                        let h = batch.clip_key.h.min(vp_h.saturating_sub(y));
                        if w == 0 || h == 0 {
                            continue;
                        }
                        pass.set_scissor_rect(x, y, w, h);
                    }

                    let end = batch.first_instance + batch.instance_count;
                    pass.draw(0..4, batch.first_instance..end);
                }
            }
        }

        // ── Pass 1.5: Bloom (downsample + upsample chain) ──
        if let Some(pp) = &post_process
            && let Some(bloom) = pp.bloom
        {
            let down_id = crate::bloom::PIPELINE_BLOOM_DOWNSAMPLE;
            let up_id = crate::bloom::PIPELINE_BLOOM_UPSAMPLE;
            if let (Some(down), Some(up)) = (registry.get(down_id), registry.get(up_id)) {
                bloom.encode(&mut encoder, &gpu.queue, &down.pipeline, &up.pipeline, 0.0, 0.0);
            }
        }

        // ── Pass 2: Post-process (fullscreen triangle) ──
        if let Some(pp) = &post_process {
            let pp_pipeline = match registry.get(pp.pipeline_id) {
                Some(handle) => &handle.pipeline,
                None => {
                    tracing::warn!(
                        pipeline_id = pp.pipeline_id.0,
                        "post-process pipeline not found, skipping"
                    );
                    gpu.queue.submit(std::iter::once(encoder.finish()));
                    output.present();
                    return Ok(());
                }
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_process_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
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

            pass.set_pipeline(pp_pipeline);
            pass.set_bind_group(0, Some(&pp.offscreen.sample_bind_group), &[]);
            pass.draw(0..3, 0..1); // Fullscreen triangle: 3 vertices, 1 instance.
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }

    /// Internal implementation for pre-acquired surface rendering.
    #[allow(clippy::too_many_arguments)]
    fn encode_and_submit_surface_inner(
        gpu: &GpuContext,
        resources: &mut RenderResources,
        frame: &mut Frame,
        uniforms: &FrameUniforms,
        registry: &PipelineRegistry,
        surface: SurfaceFrame,
        color_load_op: ColorLoadOp,
        post_process: Option<PostProcessPass<'_>>,
        msaa_view: Option<&wgpu::TextureView>,
        depth_view: Option<&wgpu::TextureView>,
        screenshot: Option<&crate::screenshot::ScreenshotCapture>,
    ) -> Result<(), crate::error::Error> {
        let instance_count = frame.instance_count();

        // Upload uniforms.
        gpu.queue
            .write_buffer(&resources.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

        let instance_count = if instance_count > MAX_INSTANCES {
            tracing::warn!(
                count = instance_count,
                max = MAX_INSTANCES,
                "instance count exceeds maximum, clamping"
            );
            MAX_INSTANCES
        } else {
            instance_count
        };

        // Double-buffered ping-pong.
        resources.flip_instance_buffer();

        // Resize instance buffer if needed.
        let cur_cap = resources.current_instance_capacity();
        if instance_count > cur_cap {
            let new_cap = instance_count.saturating_mul(2).clamp(256, MAX_INSTANCES);
            resources.resize_current_buffer(&gpu.device, new_cap);
            resources.underutilization_frames = 0;
        } else if instance_count < cur_cap / 4 && cur_cap > 256 {
            resources.underutilization_frames += 1;
            if resources.underutilization_frames >= 60 {
                let new_cap = (cur_cap / 2).max(256);
                resources.resize_current_buffer(&gpu.device, new_cap);
                resources.underutilization_frames = 0;
            }
        } else {
            resources.underutilization_frames = 0;
        }

        let surface_view = &surface.view;

        // Determine render target.
        let scene_target = match &post_process {
            Some(pp) => &pp.offscreen.render_view,
            None => surface_view,
        };

        // Upload instance data.
        if instance_count > 0 {
            let data = bytemuck::cast_slice(frame.instance_data());
            gpu.queue
                .write_buffer(resources.current_instance_buffer(), 0, data);
        }

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_encoder_surface"),
            });

        // ── Pass 1: Scene rendering ──
        {
            // When ColorLoadOp::Load, skip MSAA to avoid clobbering 3D content
            // during resolve. Render 2D directly at sample_count=1.
            let use_msaa = matches!(color_load_op, ColorLoadOp::Clear(_)) && msaa_view.is_some();
            let (view, resolve) = if use_msaa {
                (msaa_view.unwrap(), Some(scene_target))
            } else {
                (scene_target, None)
            };

            let load_op = match color_load_op {
                ColorLoadOp::Clear(c) => wgpu::LoadOp::Clear(c),
                ColorLoadOp::Load => wgpu::LoadOp::Load,
            };

            // Depth attachment must match the color attachment's sample count.
            // When MSAA is skipped (Load path), drop the multisampled depth too.
            let depth_attachment = if use_msaa {
                depth_view.map(|view| {
                    wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0),
                            store: wgpu::StoreOp::Store,
                        }),
                    }
                })
            } else {
                None
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quad_render_pass_surface"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: load_op,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: depth_attachment,
                ..Default::default()
            });

            if instance_count > 0 {
                frame.build_batches();

                pass.set_bind_group(0, Some(&resources.bind_group), &[]);
                pass.set_vertex_buffer(0, resources.quad_vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, resources.current_instance_buffer().slice(..));

                // When MSAA is configured but skipped (3D pre-render path),
                // remap pipeline IDs to the non-MSAA variants.
                let no_msaa_remap = !use_msaa && msaa_view.is_some();

                let vp_w = gpu.config.width;
                let vp_h = gpu.config.height;
                let mut current_pipeline_id: Option<u32> = None;

                for batch in frame.batches() {
                    let mut pip_id = if batch.phase == RenderPhase::OpaqueBackground
                        && batch.pipeline_id == crate::primitive::PIPELINE_SDF_2D.0
                    {
                        crate::primitive::PIPELINE_SDF_2D_OPAQUE.0
                    } else {
                        batch.pipeline_id
                    };
                    if no_msaa_remap && pip_id < crate::primitive::NO_MSAA_PIPELINE_OFFSET {
                        pip_id += crate::primitive::NO_MSAA_PIPELINE_OFFSET;
                    }
                    if current_pipeline_id != Some(pip_id) {
                        let sid = crate::primitive::ShaderId(pip_id);
                        let pipeline = match registry.get(sid) {
                            Some(handle) => &handle.pipeline,
                            None => {
                                tracing::warn!(
                                    pipeline_id = pip_id,
                                    "skipping batch: pipeline not found"
                                );
                                continue;
                            }
                        };
                        pass.set_pipeline(pipeline);
                        current_pipeline_id = Some(pip_id);
                    }

                    if batch.clip_key.is_full_viewport() {
                        pass.set_scissor_rect(0, 0, vp_w, vp_h);
                    } else {
                        let x = batch.clip_key.x.min(vp_w);
                        let y = batch.clip_key.y.min(vp_h);
                        let w = batch.clip_key.w.min(vp_w.saturating_sub(x));
                        let h = batch.clip_key.h.min(vp_h.saturating_sub(y));
                        if w == 0 || h == 0 {
                            continue;
                        }
                        pass.set_scissor_rect(x, y, w, h);
                    }

                    let end = batch.first_instance + batch.instance_count;
                    pass.draw(0..4, batch.first_instance..end);
                }
            }
        }

        // ── Pass 1.5: Bloom ──
        if let Some(pp) = &post_process
            && let Some(bloom) = pp.bloom
        {
            let down_id = crate::bloom::PIPELINE_BLOOM_DOWNSAMPLE;
            let up_id = crate::bloom::PIPELINE_BLOOM_UPSAMPLE;
            if let (Some(down), Some(up)) = (registry.get(down_id), registry.get(up_id)) {
                bloom.encode(&mut encoder, &gpu.queue, &down.pipeline, &up.pipeline, 0.0, 0.0);
            }
        }

        // ── Pass 2: Post-process ──
        if let Some(pp) = &post_process {
            let pp_pipeline = match registry.get(pp.pipeline_id) {
                Some(handle) => &handle.pipeline,
                None => {
                    tracing::warn!(
                        pipeline_id = pp.pipeline_id.0,
                        "post-process pipeline not found, skipping"
                    );
                    if let Some(sc) = screenshot {
                        sc.encode_copy(&mut encoder, &surface.texture.texture);
                    }
                    gpu.queue.submit(std::iter::once(encoder.finish()));
                    surface.texture.present();
                    return Ok(());
                }
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post_process_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
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

            pass.set_pipeline(pp_pipeline);
            pass.set_bind_group(0, Some(&pp.offscreen.sample_bind_group), &[]);
            pass.draw(0..3, 0..1);
        }

        if let Some(sc) = screenshot {
            sc.encode_copy(&mut encoder, &surface.texture.texture);
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));
        surface.texture.present();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitive::{BorderRadius, Rect};
    use crate::scene::{Node, NodeContent};

    #[test]
    fn frame_uniforms_is_32_bytes() {
        assert_eq!(size_of::<FrameUniforms>(), 32);
    }

    #[test]
    fn build_from_scene_collects_instances() {
        let mut scene = Scene::new();
        let p = crate::primitive::Primitive::SolidRect {
            rect: Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 50.0,
            },
            color: Color::WHITE,
            border_radius: BorderRadius::ZERO,
        };
        scene
            .insert(Node {
                parent: None,
                children: Vec::new(),
                offset: (10.0, 20.0),
                clip: None,
                z_order: 0,
                content: NodeContent::Leaf(p),
                dirty: false,
                opacity: 0.5,
            })
            .unwrap();

        let mut frame = Frame::new();
        frame.build_from_scene(&scene);
        assert_eq!(frame.instance_count(), 1);

        let inst = &frame.instance_data()[0];
        assert_eq!(inst.rect[0], 10.0); // x offset applied
        assert_eq!(inst.rect[1], 20.0); // y offset applied
        assert_eq!(inst.flags[2], 0.5); // opacity
    }

    #[test]
    fn build_from_scene_clears_previous() {
        let scene = Scene::new();
        let mut frame = Frame::new();
        frame.push(QuadInstance {
            rect: [0.0; 4],
            uv: [0.0; 4],
            color: [1.0; 4],
            border_radius: [0.0; 4],
            sdf_params: [0.0; 4],
            flags: [0.0; 4],
            clip_rect: [0.0; 4],
            color2: [0.0; 4],
            extra: [0.0; 4],
        });
        assert_eq!(frame.instance_count(), 1);

        frame.build_from_scene(&scene);
        assert_eq!(frame.instance_count(), 0);
    }

    #[test]
    fn frame_handles_large_instance_counts() {
        let mut frame = Frame::new();
        let instance = QuadInstance {
            rect: [0.0, 0.0, 10.0, 10.0],
            uv: [0.0; 4],
            color: [1.0; 4],
            border_radius: [0.0; 4],
            sdf_params: [0.0; 4],
            flags: [0.0; 4],
            clip_rect: [0.0; 4],
            color2: [0.0; 4],
            extra: [0.0; 4],
        };
        for _ in 0..10_000 {
            frame.push(instance);
        }
        assert_eq!(frame.instance_count(), 10_000);
        assert_eq!(frame.instance_data().len(), 10_000);
        frame.clear();
        assert_eq!(frame.instance_count(), 0);
    }

    // ── ClipKey tests ──

    #[test]
    fn clip_key_full_viewport_from_zeros() {
        let key = ClipKey::from_clip_rect([0.0, 0.0, 0.0, 0.0]);
        assert!(key.is_full_viewport());
        assert_eq!(key, ClipKey::FULL_VIEWPORT);
    }

    #[test]
    fn clip_key_quantization() {
        let key = ClipKey::from_clip_rect([10.5, 20.7, 100.3, 50.9]);
        assert_eq!(key.x, 10);
        assert_eq!(key.y, 20);
        assert_eq!(key.w, 101);
        assert_eq!(key.h, 51);
        assert!(!key.is_full_viewport());
    }

    #[test]
    fn clip_key_negative_clamped() {
        let key = ClipKey::from_clip_rect([-5.0, -10.0, 100.0, 50.0]);
        assert_eq!(key.x, 0);
        assert_eq!(key.y, 0);
    }

    // ── build_batches tests ──

    fn make_instance(clip: [f32; 4], shape_type: f32, flags3: f32) -> QuadInstance {
        QuadInstance {
            rect: [0.0, 0.0, 10.0, 10.0],
            uv: [0.0; 4],
            color: [1.0; 4],
            border_radius: [0.0; 4],
            sdf_params: [0.0; 4],
            flags: [shape_type, 0.0, 1.0, flags3],
            clip_rect: clip,
            color2: [0.0; 4],
            extra: [0.0; 4],
        }
    }

    #[test]
    fn build_batches_empty() {
        let mut frame = Frame::new();
        frame.build_batches();
        assert_eq!(frame.batches().len(), 0);
    }

    #[test]
    fn build_batches_uniform_clip_single_batch() {
        let mut frame = Frame::new();
        let no_clip = [0.0; 4];
        for _ in 0..5 {
            frame.push(make_instance(no_clip, ShapeType::Rect.to_f32(), 0.0));
        }
        frame.build_batches();
        assert_eq!(frame.batches().len(), 1);
        assert_eq!(frame.batches()[0].instance_count, 5);
        assert_eq!(frame.batches()[0].pipeline_id, 0);
        assert!(frame.batches()[0].clip_key.is_full_viewport());
    }

    #[test]
    fn build_batches_alternating_clips() {
        let mut frame = Frame::new();
        let clip_a = [10.0, 20.0, 100.0, 50.0];
        let clip_b = [50.0, 60.0, 200.0, 100.0];
        frame.push(make_instance(clip_a, ShapeType::Rect.to_f32(), 0.0));
        frame.push(make_instance(clip_b, ShapeType::Rect.to_f32(), 0.0));
        frame.push(make_instance(clip_a, ShapeType::Rect.to_f32(), 0.0));
        frame.build_batches();
        assert_eq!(frame.batches().len(), 3);
        // Check z-order preserved: instances are in original order.
        assert_eq!(frame.batches()[0].first_instance, 0);
        assert_eq!(frame.batches()[1].first_instance, 1);
        assert_eq!(frame.batches()[2].first_instance, 2);
    }

    #[test]
    fn build_batches_mixed_pipelines() {
        let mut frame = Frame::new();
        let no_clip = [0.0; 4];
        // Two built-in pipeline instances.
        frame.push(make_instance(no_clip, ShapeType::Rect.to_f32(), 0.0));
        frame.push(make_instance(no_clip, ShapeType::Rect.to_f32(), 0.0));
        // One shader pipeline instance (shader id = 42).
        frame.push(make_instance(no_clip, ShapeType::Shader.to_f32(), 42.0));
        // Back to built-in.
        frame.push(make_instance(no_clip, ShapeType::Rect.to_f32(), 0.0));
        frame.build_batches();
        assert_eq!(frame.batches().len(), 3);
        assert_eq!(frame.batches()[0].pipeline_id, 0);
        assert_eq!(frame.batches()[0].instance_count, 2);
        assert_eq!(frame.batches()[1].pipeline_id, 42);
        assert_eq!(frame.batches()[1].instance_count, 1);
        assert_eq!(frame.batches()[2].pipeline_id, 0);
        assert_eq!(frame.batches()[2].instance_count, 1);
    }

    #[test]
    fn build_batches_consecutive_same_merge() {
        let mut frame = Frame::new();
        let clip = [10.0, 20.0, 100.0, 50.0];
        for _ in 0..100 {
            frame.push(make_instance(clip, ShapeType::Rect.to_f32(), 0.0));
        }
        frame.build_batches();
        assert_eq!(frame.batches().len(), 1);
        assert_eq!(frame.batches()[0].instance_count, 100);
    }

    #[test]
    fn blend_mode_breaks_batch() {
        use crate::primitive::{PIPELINE_SDF_2D, PIPELINE_SDF_2D_ADDITIVE, PIPELINE_SDF_2D_SCREEN};
        let mut frame = Frame::new();
        let no_clip = [0.0; 4];
        // Normal blend (flags[3] = 0).
        frame.push(make_instance(no_clip, ShapeType::Circle.to_f32(), 0.0));
        // Additive blend (blend=1 << 1 = 2).
        frame.push(make_instance(no_clip, ShapeType::Circle.to_f32(), 2.0));
        // Screen blend (blend=2 << 1 = 4).
        frame.push(make_instance(no_clip, ShapeType::Circle.to_f32(), 4.0));
        // Back to normal.
        frame.push(make_instance(no_clip, ShapeType::Circle.to_f32(), 0.0));
        frame.build_batches();
        assert_eq!(frame.batches().len(), 4);
        assert_eq!(frame.batches()[0].pipeline_id, PIPELINE_SDF_2D.0);
        assert_eq!(frame.batches()[1].pipeline_id, PIPELINE_SDF_2D_ADDITIVE.0);
        assert_eq!(frame.batches()[2].pipeline_id, PIPELINE_SDF_2D_SCREEN.0);
        assert_eq!(frame.batches()[3].pipeline_id, PIPELINE_SDF_2D.0);
    }
}
