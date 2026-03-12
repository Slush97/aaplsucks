//! Glyph rasterization using swash.

use crate::face::FontFace;
use crate::{Error, RasterizedGlyph};
use swash::scale::{Render, ScaleContext, Source, StrikeWith};
use swash::zeno::{Angle, Format, Transform};

/// Glyph rasterizer wrapping swash's `ScaleContext`.
pub struct GlyphRasterizer {
    context: ScaleContext,
}

impl GlyphRasterizer {
    /// Create a new rasterizer.
    pub fn new() -> Self {
        Self {
            context: ScaleContext::new(),
        }
    }

    /// Rasterize a single glyph at the given pixel size.
    ///
    /// Returns an R8 alpha mask (one byte per pixel).
    /// Handles zero-size glyphs (like space) by returning empty data.
    ///
    /// `style` bits: bit 0 = bold (faux embolden), bit 1 = italic (12° skew).
    pub fn rasterize(
        &mut self,
        face: &FontFace,
        glyph_id: u32,
        size_px: f32,
        style: u8,
    ) -> Result<RasterizedGlyph, Error> {
        let font_ref = face.as_swash_ref();
        let is_bold = style & 1 != 0;

        let mut scaler = self
            .context
            .builder(font_ref)
            .size(size_px)
            .hint(true)
            .build();

        let mut render = Render::new(&[Source::Outline]);
        render.format(Format::Alpha);

        // Note: we do NOT use swash's `render.embolden()` — it produces nearly
        // transparent output for narrow glyphs like 'i'. Instead, faux bold is
        // applied as a post-rasterization alpha dilation (see below).
        if style & 2 != 0 {
            render.transform(Some(Transform::skew(
                Angle::from_degrees(12.0),
                Angle::ZERO,
            )));
        }

        let image = render.render(&mut scaler, glyph_id as u16);

        match image {
            Some(img) => {
                let placement = img.placement;
                let width = placement.width;
                let height = placement.height;

                if is_bold && width > 0 && height > 0 {
                    // Faux bold: dilate the alpha mask by 1px to the right.
                    // This widens stems without distorting shapes, matching the
                    // approach used by Alacritty and other GPU terminal emulators.
                    let new_width = width + 1;
                    let mut dilated = vec![0u8; (new_width * height) as usize];
                    for row in 0..height {
                        for col in 0..new_width {
                            let orig = if col < width {
                                img.data[(row * width + col) as usize]
                            } else {
                                0
                            };
                            let left = if col > 0 && col - 1 < width {
                                img.data[(row * width + col - 1) as usize]
                            } else {
                                0
                            };
                            dilated[(row * new_width + col) as usize] = orig.max(left);
                        }
                    }

                    Ok(RasterizedGlyph {
                        glyph_id,
                        width: new_width,
                        height,
                        bearing_x: placement.left as f32,
                        bearing_y: placement.top as f32,
                        data: dilated,
                        is_color: false,
                    })
                } else {
                    // Alpha mask — single channel (R8), no RGBA expansion.
                    Ok(RasterizedGlyph {
                        glyph_id,
                        width,
                        height,
                        bearing_x: placement.left as f32,
                        bearing_y: placement.top as f32,
                        data: img.data,
                        is_color: false,
                    })
                }
            }
            None => {
                // Zero-size glyph (space, .notdef, etc.).
                Ok(RasterizedGlyph {
                    glyph_id,
                    width: 0,
                    height: 0,
                    bearing_x: 0.0,
                    bearing_y: 0.0,
                    data: Vec::new(),
                    is_color: false,
                })
            }
        }
    }

    /// Rasterize a glyph attempting color sources first (COLR/CPAL, CBDT/sbix).
    ///
    /// Source priority: `ColorOutline(0)` → `ColorBitmap(BestFit)` → monochrome
    /// fallback via [`rasterize()`](Self::rasterize). Returns RGBA8 data with
    /// `is_color = true` when a color source was used. No faux bold/italic is
    /// applied to color glyphs (emoji should not be emboldened/skewed).
    pub fn rasterize_color(
        &mut self,
        face: &FontFace,
        glyph_id: u32,
        size_px: f32,
        style: u8,
    ) -> Result<RasterizedGlyph, Error> {
        let font_ref = face.as_swash_ref();

        let mut scaler = self
            .context
            .builder(font_ref)
            .size(size_px)
            .hint(false)
            .build();

        // Try color outline (COLR/CPAL tables).
        let mut render = Render::new(&[
            Source::ColorOutline(0),
            Source::ColorBitmap(StrikeWith::BestFit),
        ]);
        render.format(Format::CustomSubpixel([0.0, 0.0, 0.0]));

        if let Some(img) = render.render(&mut scaler, glyph_id as u16) {
            let placement = img.placement;
            let width = placement.width;
            let height = placement.height;

            if width > 0 && height > 0 {
                // swash CustomSubpixel gives us raw RGBA8 data.
                return Ok(RasterizedGlyph {
                    glyph_id,
                    width,
                    height,
                    bearing_x: placement.left as f32,
                    bearing_y: placement.top as f32,
                    data: img.data,
                    is_color: true,
                });
            }
        }

        // No color source available — fall back to monochrome.
        self.rasterize(face, glyph_id, size_px, style)
    }
}

impl Default for GlyphRasterizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FontId;

    fn test_face() -> FontFace {
        let data = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-data/JetBrainsMono-Regular.ttf"
        ))
        .expect("test font not found");
        FontFace::from_bytes(FontId(0), data).unwrap()
    }

    #[test]
    fn rasterize_letter_a() {
        let face = test_face();
        let mut rasterizer = GlyphRasterizer::new();
        let font_ref = face.as_swash_ref();
        let glyph_id = u32::from(font_ref.charmap().map('A'));

        let glyph = rasterizer.rasterize(&face, glyph_id, 16.0, 0).unwrap();
        assert!(glyph.width > 0, "width={}", glyph.width);
        assert!(glyph.height > 0, "height={}", glyph.height);
        assert!(!glyph.data.is_empty());
        // R8: 1 byte per pixel (alpha-only).
        assert_eq!(glyph.data.len(), (glyph.width * glyph.height) as usize);
    }

    #[test]
    fn rasterize_dimensions_reasonable() {
        let face = test_face();
        let mut rasterizer = GlyphRasterizer::new();
        let font_ref = face.as_swash_ref();
        let glyph_id = u32::from(font_ref.charmap().map('A'));

        let glyph = rasterizer.rasterize(&face, glyph_id, 16.0, 0).unwrap();
        // A 16px glyph should be roughly 8-20 pixels wide/tall.
        assert!(glyph.width < 30, "width={}", glyph.width);
        assert!(glyph.height < 30, "height={}", glyph.height);
    }

    #[test]
    fn rasterize_space_is_zero_size() {
        let face = test_face();
        let mut rasterizer = GlyphRasterizer::new();
        let font_ref = face.as_swash_ref();
        let glyph_id = u32::from(font_ref.charmap().map(' '));

        let glyph = rasterizer.rasterize(&face, glyph_id, 16.0, 0).unwrap();
        assert_eq!(glyph.width, 0);
        assert_eq!(glyph.height, 0);
        assert!(glyph.data.is_empty());
    }

    #[test]
    fn rasterize_italic_produces_output() {
        let face = test_face();
        let mut rasterizer = GlyphRasterizer::new();
        let font_ref = face.as_swash_ref();
        let glyph_id = u32::from(font_ref.charmap().map('A'));

        let glyph = rasterizer.rasterize(&face, glyph_id, 16.0, 2).unwrap();
        assert!(glyph.width > 0);
        assert!(glyph.height > 0);
        assert!(!glyph.data.is_empty());
    }

    #[test]
    fn rasterize_bold_lowercase_i() {
        let face = test_face();
        let mut rasterizer = GlyphRasterizer::new();
        let font_ref = face.as_swash_ref();
        let glyph_id = u32::from(font_ref.charmap().map('i'));

        let regular = rasterizer.rasterize(&face, glyph_id, 16.0, 0).unwrap();
        let bold = rasterizer.rasterize(&face, glyph_id, 16.0, 1).unwrap();

        // Both must produce non-zero bitmaps.
        assert!(regular.width > 0 && regular.height > 0);
        assert!(bold.width > 0 && bold.height > 0);

        // Bold should be wider (1px dilation) and same height.
        assert_eq!(bold.width, regular.width + 1);
        assert_eq!(bold.height, regular.height);

        // Data size must match R8 dimensions (1 byte per pixel).
        assert_eq!(bold.data.len(), (bold.width * bold.height) as usize);

        // Bold must have full opacity (not faint from broken embolden).
        // R8 format: each byte is the alpha value directly.
        let bold_max_alpha = bold.data.iter().copied().max().unwrap_or(0);
        assert!(
            bold_max_alpha > 200,
            "bold i max alpha too low: {bold_max_alpha}"
        );

        // Bold should have more total ink than regular.
        let bold_total: u64 = bold.data.iter().map(|&a| a as u64).sum();
        let regular_total: u64 = regular.data.iter().map(|&a| a as u64).sum();
        assert!(bold_total > regular_total, "bold should have more ink");
    }

    #[test]
    fn rasterize_color_fallback_on_monochrome_font() {
        let face = test_face();
        let mut rasterizer = GlyphRasterizer::new();
        let font_ref = face.as_swash_ref();
        let glyph_id = u32::from(font_ref.charmap().map('A'));

        let color = rasterizer
            .rasterize_color(&face, glyph_id, 16.0, 0)
            .unwrap();
        // JetBrains Mono has no color tables — should fall back to monochrome.
        assert!(!color.is_color);
        assert!(color.width > 0);
        assert!(color.height > 0);
        // Monochrome fallback uses R8 (1 byte per pixel).
        assert_eq!(color.data.len(), (color.width * color.height) as usize);

        // Output should match the regular rasterize() path.
        let mono = rasterizer.rasterize(&face, glyph_id, 16.0, 0).unwrap();
        assert_eq!(color.width, mono.width);
        assert_eq!(color.height, mono.height);
    }
}
