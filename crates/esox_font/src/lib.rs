//! `esox_font` — Font loading, shaping, and glyph rasterization.
//!
//! Uses `rustybuzz` for shaping and `swash` for rasterization, with glyph
//! caching backed by `esox_gfx::AtlasAllocator`.
//!
//! ## Font fallback
//!
//! The font pipeline uses `fc-match` to resolve system fonts at startup, then
//! builds a [`FontFallbackChain`] that maps each codepoint to the best
//! available font face. CJK, emoji, and symbol ranges are handled by querying
//! fontconfig for coverage rather than bundling fallback data.
//!
//! ## Pipeline
//!
//! 1. [`SystemFontDb`] / `fc-match` → font file paths
//! 2. [`FontFace`] → loaded `ttf-parser` face
//! 3. [`TextShaper`] → `rustybuzz` shaping → [`ShapedRun`]
//! 4. [`GlyphRasterizer`] → `swash` rasterization → [`RasterizedGlyph`]
//! 5. [`GlyphCache`] → atlas-backed LRU cache with generation eviction

pub mod cache;
pub mod face;
pub mod fallback;
pub mod lookup;
pub mod rasterizer;
pub mod shaper;

// Re-exports for convenience.
pub use cache::{CachedGlyph, GlyphCache};
pub use face::FontFace;
pub use fallback::FontFallbackChain;
pub use lookup::{FontStyle, SystemFontDb};
pub use rasterizer::GlyphRasterizer;
pub use shaper::TextShaper;

/// Errors produced by the font subsystem.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Failed to load a font file.
    #[error("failed to load font: {0}")]
    Load(String),

    /// Failed to rasterize a glyph.
    #[error("rasterization failed for glyph {glyph_id}: {reason}")]
    Rasterize {
        /// The glyph that failed.
        glyph_id: u32,
        /// Why it failed.
        reason: String,
    },

    /// Atlas allocation error (forwarded from esox_gfx).
    #[error("atlas error: {0}")]
    Atlas(#[from] esox_gfx::Error),
}

/// Unique handle for a loaded font face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontId(pub u32);

/// Font metrics for layout (all in font units or pixels at a given size).
#[derive(Debug, Clone, Copy)]
pub struct FontMetrics {
    /// Cell width in pixels.
    pub cell_width: f32,
    /// Cell height in pixels (ascent + descent + leading).
    pub cell_height: f32,
    /// Ascent from baseline.
    pub ascent: f32,
    /// Descent from baseline (typically negative).
    pub descent: f32,
    /// Distance from baseline to underline position (positive = below).
    pub underline_offset: f32,
    /// Thickness for underline/strikethrough strokes.
    pub stroke_size: f32,
    /// Distance from baseline to strikethrough position.
    pub strikeout_offset: f32,
}

/// A run of shaped text (one font, one style).
pub struct ShapedRun {
    /// The glyphs in this run.
    pub glyphs: Vec<ShapedGlyph>,
}

/// A single shaped glyph with positioning info.
#[derive(Debug, Clone, Copy)]
pub struct ShapedGlyph {
    /// Glyph ID in the font.
    pub glyph_id: u32,
    /// X offset from the pen position.
    pub x_offset: f32,
    /// Y offset from the pen position.
    pub y_offset: f32,
    /// How far to advance the pen after this glyph.
    pub x_advance: f32,
    /// Cluster index (maps back to the source text).
    pub cluster: u32,
}

/// A rasterized glyph bitmap ready for atlas upload.
pub struct RasterizedGlyph {
    /// Glyph ID.
    pub glyph_id: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Horizontal bearing (offset from origin to left edge).
    pub bearing_x: f32,
    /// Vertical bearing (offset from baseline to top edge).
    pub bearing_y: f32,
    /// RGBA8 pixel data. For monochrome glyphs this is white + alpha mask;
    /// for color glyphs (COLR/CBDT/sbix) this is full RGBA from the font.
    pub data: Vec<u8>,
    /// Whether this glyph was rasterized from a color source (COLR/CBDT/sbix).
    pub is_color: bool,
}

/// Key for looking up a cached glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// Which font face.
    pub font_id: FontId,
    /// Glyph ID in the font.
    pub glyph_id: u32,
    /// Font size in tenths of a pixel (for integer hashing).
    pub size_tenths: u32,
    /// Style bits: bit 0 = bold, bit 1 = italic, bit 2 = color request.
    pub style: u8,
}
