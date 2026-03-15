//! Text rendering via `esox_font` glyph pipeline.
//!
//! Shapes text with rustybuzz, rasterizes with swash, caches in a GPU atlas,
//! and pushes textured quads to the frame. A shaped-run cache eliminates
//! redundant rustybuzz calls across and within frames.

use std::collections::HashMap;

use esox_font::{
    FontFace, FontId, FontStyle, GlyphCache, GlyphKey, GlyphRasterizer, ShapedGlyph,
    SystemFontDb, TextShaper,
};
use esox_gfx::{
    AtlasAllocator, AtlasId, AtlasTexture, Color, Frame, GpuContext, QuadInstance,
    RenderResources, ShapeType, ShelfAllocator, UvRect,
};

/// Text truncation mode for overflow handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationMode {
    /// Truncate at end: `text…`
    End,
    /// Truncate at start: `…text`
    Start,
    /// Truncate in middle: `te…xt`
    Middle,
}

/// Initial atlas dimensions in texels. Grows on demand when full.
const ATLAS_SIZE: u32 = 256;

// ── Shaped-run cache ────────────────────────────────────────────────────────

/// Maximum entries before eviction is forced.
const SHAPE_CACHE_CAPACITY: usize = 4096;

/// Entries idle for this many frames are evicted on sweep.
const SHAPE_CACHE_MAX_IDLE: u32 = 240;

/// FNV-1a hash over a byte slice.
fn fnv1a_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x00000100000001b3);
    }
    h
}

/// Cache key for a shaped text run.
///
/// Style (bold/italic) is *not* included because shaping output is
/// style-independent — style only affects glyph rasterization.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ShapeKey {
    text_hash: u64,
    size_tenths: u32,
}

impl ShapeKey {
    fn new(text: &str, size: f32) -> Self {
        let mut h = fnv1a_bytes(text.as_bytes());
        // Fold text length for extra collision resistance.
        for b in (text.len() as u64).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x00000100000001b3);
        }
        Self {
            text_hash: h,
            size_tenths: (size * 10.0) as u32,
        }
    }
}

struct ShapeCacheEntry {
    glyphs: Vec<ShapedGlyph>,
    total_advance: f32,
    /// Byte length of the source text (collision guard).
    text_len: usize,
    last_generation: u32,
}

/// LRU cache for shaped text runs. Eliminates redundant rustybuzz calls for
/// static or repeated text. Entries evicted after [`SHAPE_CACHE_MAX_IDLE`]
/// idle frames.
struct ShapeCache {
    entries: HashMap<ShapeKey, ShapeCacheEntry>,
    generation: u32,
}

impl ShapeCache {
    fn new() -> Self {
        Self {
            entries: HashMap::with_capacity(256),
            generation: 0,
        }
    }

    fn advance_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation % 60 == 0 || self.entries.len() > SHAPE_CACHE_CAPACITY {
            let gen = self.generation;
            self.entries
                .retain(|_, e| gen.wrapping_sub(e.last_generation) < SHAPE_CACHE_MAX_IDLE);
        }
    }

    /// Zero-clone fast path — returns total advance only.
    fn lookup_advance(&mut self, key: &ShapeKey, text_len: usize) -> Option<f32> {
        if let Some(e) = self.entries.get_mut(key) {
            if e.text_len == text_len {
                e.last_generation = self.generation;
                return Some(e.total_advance);
            }
        }
        None
    }

    /// Returns cloned glyph data. Used when caller needs positioned glyphs.
    fn lookup_glyphs(&mut self, key: &ShapeKey, text_len: usize) -> Option<Vec<ShapedGlyph>> {
        if let Some(e) = self.entries.get_mut(key) {
            if e.text_len == text_len {
                e.last_generation = self.generation;
                return Some(e.glyphs.clone());
            }
        }
        None
    }

    fn insert(
        &mut self,
        key: ShapeKey,
        glyphs: Vec<ShapedGlyph>,
        total_advance: f32,
        text_len: usize,
    ) {
        self.entries.insert(
            key,
            ShapeCacheEntry {
                glyphs,
                total_advance,
                text_len,
                last_generation: self.generation,
            },
        );
    }
}

// ── TextRenderer ────────────────────────────────────────────────────────────

/// Renders text by shaping, rasterizing, caching glyphs in an atlas, and
/// pushing textured quads to the frame.
pub struct TextRenderer {
    face: FontFace,
    shaper: TextShaper,
    rasterizer: GlyphRasterizer,
    cache: GlyphCache,
    allocator: ShelfAllocator,
    atlas: AtlasTexture,
    atlas_bound: bool,
    shape_cache: ShapeCache,
}

impl TextRenderer {
    /// Initialize the text renderer, loading a UI font.
    ///
    /// Returns an error if no suitable sans-serif font is found on the system
    /// or the font data cannot be parsed.
    pub fn new(gpu: &GpuContext) -> Result<Self, String> {
        let db = SystemFontDb::new();

        let font_data = db
            .query_family("Inter", FontStyle::Regular)
            .or_else(|| db.query_family("Noto Sans", FontStyle::Regular))
            .or_else(|| db.query_family("DejaVu Sans", FontStyle::Regular))
            .or_else(|| db.query_family("Liberation Sans", FontStyle::Regular))
            .ok_or_else(|| "no sans-serif font found on system".to_string())?;

        tracing::info!(family = %font_data.family, "loaded UI font");

        let face = FontFace::from_bytes(FontId(0), font_data.data)
            .map_err(|e| format!("failed to load font face: {e}"))?;

        let allocator = ShelfAllocator::new(AtlasId(0), ATLAS_SIZE, ATLAS_SIZE);
        let atlas = AtlasTexture::new(&gpu.device, ATLAS_SIZE, ATLAS_SIZE, "ui_glyph_atlas");

        Ok(Self {
            face,
            shaper: TextShaper::new(),
            rasterizer: GlyphRasterizer::new(),
            cache: GlyphCache::new(),
            allocator,
            atlas,
            atlas_bound: false,
            shape_cache: ShapeCache::new(),
        })
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Draw text at the given position and return the total advance width.
    pub fn draw_text(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        color: Color,
        frame: &mut Frame,
        gpu: &GpuContext,
        resources: &mut RenderResources,
    ) -> f32 {
        let glyphs = self.resolve_glyphs(text, size);
        self.render_glyphs(&glyphs, x, y, size, color, 0, frame, gpu, resources)
    }

    /// Draw text with a style byte (bit 0 = bold via faux emboldening).
    pub fn draw_text_styled(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        color: Color,
        style: u8,
        frame: &mut Frame,
        gpu: &GpuContext,
        resources: &mut RenderResources,
    ) -> f32 {
        let glyphs = self.resolve_glyphs(text, size);
        self.render_glyphs(&glyphs, x, y, size, color, style, frame, gpu, resources)
    }

    /// Draw text at the standard UI font size (14px).
    pub fn draw_ui_text(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        color: Color,
        frame: &mut Frame,
        gpu: &GpuContext,
        resources: &mut RenderResources,
    ) -> f32 {
        self.draw_text(text, x, y, 14.0, color, frame, gpu, resources)
    }

    /// Draw text at the header font size (11px).
    pub fn draw_header_text(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        color: Color,
        frame: &mut Frame,
        gpu: &GpuContext,
        resources: &mut RenderResources,
    ) -> f32 {
        self.draw_text(text, x, y, 11.0, color, frame, gpu, resources)
    }

    /// Measure text width without drawing. Zero-clone cache fast path.
    pub fn measure_text(&mut self, text: &str, size: f32) -> f32 {
        if text.is_empty() {
            return 0.0;
        }
        let key = ShapeKey::new(text, size);
        if let Some(advance) = self.shape_cache.lookup_advance(&key, text.len()) {
            return advance;
        }
        let run = self.shaper.shape(&self.face, text, size);
        let advance = run.glyphs.iter().map(|g| g.x_advance).sum();
        self.shape_cache.insert(key, run.glyphs, advance, text.len());
        advance
    }

    /// Line height (cell_height) for a given font size.
    pub fn line_height(&mut self, size: f32) -> f32 {
        self.face.metrics(size).cell_height
    }

    /// Measure wrapped text dimensions. Returns `(max_line_width, total_height)`
    /// including line spacing between lines. Uses `wrap_lines_measured()` internally.
    pub fn measure_text_wrapped(
        &mut self,
        text: &str,
        size: f32,
        max_width: f32,
        line_spacing: f32,
    ) -> (f32, f32) {
        if text.is_empty() {
            return (0.0, self.line_height(size));
        }
        let lines = self.wrap_lines_measured(text, size, max_width);
        let line_height = self.line_height(size);
        let max_w = lines.iter().map(|&(_, _, w)| w).fold(0.0f32, f32::max);
        let total_h = lines.len() as f32 * line_height
            + (lines.len().saturating_sub(1)) as f32 * line_spacing;
        (max_w, total_h)
    }

    /// Split text into lines fitting within `max_width`, with per-line widths.
    /// Returns `(start_byte, end_byte, line_width)` triples.
    pub fn wrap_lines_measured(
        &mut self,
        text: &str,
        size: f32,
        max_width: f32,
    ) -> Vec<(usize, usize, f32)> {
        if text.is_empty() {
            return vec![(0, 0, 0.0)];
        }

        let mut lines: Vec<(usize, usize, f32)> = Vec::new();
        let mut line_start: usize = 0;
        let mut line_width: f32 = 0.0;
        let space_width = self.measure_text(" ", size);

        let mut words: Vec<(usize, usize)> = Vec::new();
        let mut word_start: Option<usize> = None;
        for (i, c) in text.char_indices() {
            if c.is_whitespace() {
                if let Some(ws) = word_start.take() {
                    words.push((ws, i));
                }
            } else if word_start.is_none() {
                word_start = Some(i);
            }
        }
        if let Some(ws) = word_start {
            words.push((ws, text.len()));
        }

        if words.is_empty() {
            return vec![(0, text.len(), 0.0)];
        }

        for &(word_start_byte, word_end_byte) in &words {
            let word = &text[word_start_byte..word_end_byte];
            let word_width = self.measure_text(word, size);

            if word_width > max_width {
                if line_width > 0.0 {
                    lines.push((line_start, word_start_byte, line_width));
                }
                let mut char_start = word_start_byte;
                let mut accum = 0.0_f32;
                for (i, c) in word[..].char_indices() {
                    let byte_pos = word_start_byte + i;
                    let cw =
                        self.measure_text(&text[byte_pos..byte_pos + c.len_utf8()], size);
                    if accum + cw > max_width && accum > 0.0 {
                        lines.push((char_start, byte_pos, accum));
                        char_start = byte_pos;
                        accum = cw;
                    } else {
                        accum += cw;
                    }
                }
                line_start = char_start;
                line_width = accum;
                continue;
            }

            if line_width == 0.0 {
                line_width = word_width;
            } else if line_width + space_width + word_width <= max_width {
                line_width += space_width + word_width;
            } else {
                lines.push((line_start, word_start_byte, line_width));
                line_start = word_start_byte;
                line_width = word_width;
            }
        }

        lines.push((line_start, text.len(), line_width));
        lines
    }

    /// Split text into lines fitting within `max_width`. Greedy word-wrap at
    /// whitespace. Falls back to character-level breaking for words wider than
    /// `max_width`. Returns byte-offset pairs `(start, end)`.
    pub fn wrap_lines(&mut self, text: &str, size: f32, max_width: f32) -> Vec<(usize, usize)> {
        if text.is_empty() {
            return vec![(0, 0)];
        }

        let mut lines: Vec<(usize, usize)> = Vec::new();
        let mut line_start: usize = 0;
        let mut line_width: f32 = 0.0;
        let space_width = self.measure_text(" ", size);

        let mut words: Vec<(usize, usize)> = Vec::new();
        let mut word_start: Option<usize> = None;
        for (i, c) in text.char_indices() {
            if c.is_whitespace() {
                if let Some(ws) = word_start.take() {
                    words.push((ws, i));
                }
            } else if word_start.is_none() {
                word_start = Some(i);
            }
        }
        if let Some(ws) = word_start {
            words.push((ws, text.len()));
        }

        if words.is_empty() {
            return vec![(0, text.len())];
        }

        for &(word_start_byte, word_end_byte) in &words {
            let word = &text[word_start_byte..word_end_byte];
            let word_width = self.measure_text(word, size);

            if word_width > max_width {
                if line_width > 0.0 {
                    lines.push((line_start, word_start_byte));
                }
                let mut char_start = word_start_byte;
                let mut accum = 0.0_f32;
                for (i, c) in word[..].char_indices() {
                    let byte_pos = word_start_byte + i;
                    let cw =
                        self.measure_text(&text[byte_pos..byte_pos + c.len_utf8()], size);
                    if accum + cw > max_width && accum > 0.0 {
                        lines.push((char_start, byte_pos));
                        char_start = byte_pos;
                        accum = cw;
                    } else {
                        accum += cw;
                    }
                }
                line_start = char_start;
                line_width = accum;
                continue;
            }

            if line_width == 0.0 {
                line_width = word_width;
            } else if line_width + space_width + word_width <= max_width {
                line_width += space_width + word_width;
            } else {
                lines.push((line_start, word_start_byte));
                line_start = word_start_byte;
                line_width = word_width;
            }
        }

        lines.push((line_start, text.len()));
        lines
    }

    /// Draw text truncated with "\u{2026}" if it exceeds `max_width`.
    ///
    /// Uses a glyph walk to find the truncation point in O(glyphs) instead of
    /// O(chars) shaping calls. Returns the advance width of what was drawn.
    pub fn draw_text_truncated(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        max_width: f32,
        color: Color,
        frame: &mut Frame,
        gpu: &GpuContext,
        resources: &mut RenderResources,
    ) -> f32 {
        let full_width = self.measure_text(text, size);
        if full_width <= max_width {
            return self.draw_text(text, x, y, size, color, frame, gpu, resources);
        }

        let ellipsis = "\u{2026}";
        let ellipsis_width = self.measure_text(ellipsis, size);
        let target = max_width - ellipsis_width;

        if target <= 0.0 {
            return self.draw_text(ellipsis, x, y, size, color, frame, gpu, resources);
        }

        // Walk shaped glyphs to find truncation point.
        let glyphs = self.resolve_glyphs(text, size);
        let mut accum = 0.0;
        let mut trunc_count = 0;
        for glyph in &glyphs {
            if accum + glyph.x_advance > target {
                break;
            }
            accum += glyph.x_advance;
            trunc_count += 1;
        }

        // Render prefix glyphs directly — no re-shaping of the truncated string.
        let advance = self.render_glyphs(
            &glyphs[..trunc_count],
            x, y, size, color, 0, frame, gpu, resources,
        );
        self.draw_text(ellipsis, x + advance, y, size, color, frame, gpu, resources);
        advance + ellipsis_width
    }

    /// Draw text truncated with a specified truncation mode.
    ///
    /// - `End`: `text…` (same as `draw_text_truncated`)
    /// - `Start`: `…text` (walk glyphs from end backward)
    /// - `Middle`: `te…xt` (split budget between prefix and suffix)
    pub fn draw_text_truncated_mode(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        max_width: f32,
        color: Color,
        mode: TruncationMode,
        frame: &mut Frame,
        gpu: &GpuContext,
        resources: &mut RenderResources,
    ) -> f32 {
        let full_width = self.measure_text(text, size);
        if full_width <= max_width {
            return self.draw_text(text, x, y, size, color, frame, gpu, resources);
        }

        let ellipsis = "\u{2026}";
        let ellipsis_width = self.measure_text(ellipsis, size);
        let target = max_width - ellipsis_width;

        if target <= 0.0 {
            return self.draw_text(ellipsis, x, y, size, color, frame, gpu, resources);
        }

        let glyphs = self.resolve_glyphs(text, size);

        match mode {
            TruncationMode::End => {
                // Walk forward, same as draw_text_truncated.
                let mut accum = 0.0;
                let mut trunc_count = 0;
                for glyph in &glyphs {
                    if accum + glyph.x_advance > target {
                        break;
                    }
                    accum += glyph.x_advance;
                    trunc_count += 1;
                }
                let advance = self.render_glyphs(
                    &glyphs[..trunc_count],
                    x, y, size, color, 0, frame, gpu, resources,
                );
                self.draw_text(ellipsis, x + advance, y, size, color, frame, gpu, resources);
                advance + ellipsis_width
            }
            TruncationMode::Start => {
                // Walk backward from end.
                let mut accum = 0.0;
                let mut suffix_start = glyphs.len();
                for i in (0..glyphs.len()).rev() {
                    if accum + glyphs[i].x_advance > target {
                        break;
                    }
                    accum += glyphs[i].x_advance;
                    suffix_start = i;
                }
                let ew = self.draw_text(ellipsis, x, y, size, color, frame, gpu, resources);
                self.render_glyphs(
                    &glyphs[suffix_start..],
                    x + ew, y, size, color, 0, frame, gpu, resources,
                );
                ew + accum
            }
            TruncationMode::Middle => {
                // Split budget: half for prefix, half for suffix.
                let half = target / 2.0;

                let mut prefix_accum = 0.0;
                let mut prefix_count = 0;
                for glyph in &glyphs {
                    if prefix_accum + glyph.x_advance > half {
                        break;
                    }
                    prefix_accum += glyph.x_advance;
                    prefix_count += 1;
                }

                let mut suffix_accum = 0.0;
                let mut suffix_start = glyphs.len();
                for i in (0..glyphs.len()).rev() {
                    if suffix_accum + glyphs[i].x_advance > half {
                        break;
                    }
                    suffix_accum += glyphs[i].x_advance;
                    suffix_start = i;
                }

                let prefix_advance = self.render_glyphs(
                    &glyphs[..prefix_count],
                    x, y, size, color, 0, frame, gpu, resources,
                );
                let ew = self.draw_text(
                    ellipsis,
                    x + prefix_advance, y, size, color, frame, gpu, resources,
                );
                self.render_glyphs(
                    &glyphs[suffix_start..],
                    x + prefix_advance + ew, y, size, color, 0, frame, gpu, resources,
                );
                prefix_advance + ew + suffix_accum
            }
        }
    }

    /// Map a pixel x-offset to the nearest byte offset in `text`.
    ///
    /// Uses the cached shaped run to walk glyph advances in O(glyphs) without
    /// per-character shaping. Intended for cursor placement in text inputs.
    pub fn x_to_byte_offset(&mut self, text: &str, size: f32, target_x: f32) -> usize {
        if text.is_empty() || target_x <= 0.0 {
            return 0;
        }

        let glyphs = self.resolve_glyphs(text, size);
        let mut accum = 0.0_f32;
        let mut best = 0usize;
        let mut best_dist = target_x;

        for (i, glyph) in glyphs.iter().enumerate() {
            accum += glyph.x_advance;
            // Byte position after this glyph's cluster.
            let byte_pos = if i + 1 < glyphs.len() {
                glyphs[i + 1].cluster as usize
            } else {
                text.len()
            };
            let dist = (accum - target_x).abs();
            if dist < best_dist {
                best = byte_pos;
                best_dist = dist;
            }
        }

        best
    }

    /// Advance the allocator generation (call once per frame for LRU tracking).
    pub fn advance_generation(&mut self) {
        self.allocator.advance_generation();
        self.shape_cache.advance_generation();
    }

    // ── Internal ────────────────────────────────────────────────────────

    /// Get shaped glyphs from cache (cloned) or shape fresh.
    fn resolve_glyphs(&mut self, text: &str, size: f32) -> Vec<ShapedGlyph> {
        if text.is_empty() {
            return Vec::new();
        }
        let key = ShapeKey::new(text, size);
        if let Some(glyphs) = self.shape_cache.lookup_glyphs(&key, text.len()) {
            return glyphs;
        }
        let run = self.shaper.shape(&self.face, text, size);
        let advance = run.glyphs.iter().map(|g| g.x_advance).sum();
        let ret = run.glyphs.clone();
        self.shape_cache.insert(key, run.glyphs, advance, text.len());
        ret
    }

    /// Core glyph renderer. Renders pre-shaped glyphs, handling rasterization,
    /// atlas allocation, eviction, and GPU upload. Returns total advance width.
    fn render_glyphs(
        &mut self,
        glyphs: &[ShapedGlyph],
        x: f32,
        y: f32,
        size: f32,
        color: Color,
        style: u8,
        frame: &mut Frame,
        gpu: &GpuContext,
        resources: &mut RenderResources,
    ) -> f32 {
        let metrics = self.face.metrics(size);
        let (atlas_w, atlas_h) = self.allocator.size();
        let size_tenths = (size * 10.0) as u32;
        let bold = style & 1 != 0;

        let mut pen_x = x;
        for glyph in glyphs {
            let key = GlyphKey {
                font_id: self.face.id(),
                glyph_id: glyph.glyph_id,
                size_tenths,
                style,
            };

            let cached = match self.get_or_insert_with_eviction(key, size, bold) {
                Some(c) => c,
                None => {
                    pen_x += glyph.x_advance;
                    continue;
                }
            };

            if cached.region.w > 0 && cached.region.h > 0 {
                let uv = cached.region.to_uv_rect(atlas_w, atlas_h);
                let gx = (pen_x + glyph.x_offset + cached.bearing_x).round();
                let gy = (y + metrics.ascent - cached.bearing_y + glyph.y_offset).round();
                let gw = cached.region.w as f32;
                let gh = cached.region.h as f32;

                frame.push(make_textured_quad(gx, gy, gw, gh, uv, color));
            }

            pen_x += glyph.x_advance;
        }

        self.flush_uploads(gpu, resources);

        pen_x - x
    }

    /// Upload pending glyph data to the atlas and rebind if needed.
    fn flush_uploads(&mut self, gpu: &GpuContext, resources: &mut RenderResources) {
        let uploads = self.cache.drain_uploads();
        if !uploads.is_empty() {
            for (region, data) in &uploads {
                self.atlas.upload_region(
                    &gpu.queue,
                    region.x,
                    region.y,
                    region.w,
                    region.h,
                    data,
                );
            }
        }

        if !self.atlas_bound {
            resources.bind_textures(
                &gpu.device,
                self.atlas.view(),
                self.atlas.view(),
            );
            self.atlas_bound = true;
        }
    }

    /// Cache-or-rasterize with atlas-full eviction recovery.
    fn get_or_insert_with_eviction(
        &mut self,
        key: GlyphKey,
        size_px: f32,
        bold: bool,
    ) -> Option<esox_font::CachedGlyph> {
        match self.cache.get_or_insert(
            key,
            &self.face,
            &mut self.rasterizer,
            &mut self.allocator,
            size_px,
            bold,
        ) {
            Ok(cached) => Some(cached),
            Err(esox_font::Error::Atlas(esox_gfx::Error::AtlasFull)) => {
                if self.allocator.fragmentation() > 0.3 {
                    self.cache.clear();
                    self.allocator.clear();
                } else {
                    let count = self.cache.len() / 4;
                    let evicted = self.allocator.evict_lru(0, count.max(16));
                    self.cache.invalidate(&evicted);
                }
                self.cache
                    .get_or_insert(
                        key,
                        &self.face,
                        &mut self.rasterizer,
                        &mut self.allocator,
                        size_px,
                        bold,
                    )
                    .ok()
            }
            Err(e) => {
                tracing::warn!("glyph rasterization failed: {e}");
                None
            }
        }
    }
}

/// Build a textured quad for a glyph.
fn make_textured_quad(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    uv: UvRect,
    color: Color,
) -> QuadInstance {
    QuadInstance {
        rect: [x, y, w, h],
        uv: [uv.u0, uv.v0, uv.u1, uv.v1],
        color: [color.r, color.g, color.b, color.a],
        border_radius: [0.0; 4],
        sdf_params: [0.0; 4],
        flags: [ShapeType::Textured.to_f32(), 0.0, 1.0, 0.0],
        clip_rect: [0.0; 4],
        color2: [0.0; 4],
        extra: [0.0; 4],
    }
}
