//! Terminal-optimized text shaping.

use crate::face::FontFace;
use crate::{ShapedGlyph, ShapedRun};

/// Text shaper with buffer recycling for terminal rendering.
///
/// Provides a fast monospace ASCII path and a full rustybuzz path.
pub struct TextShaper {
    buffer: Option<rustybuzz::UnicodeBuffer>,
}

impl TextShaper {
    /// Create a new text shaper.
    pub fn new() -> Self {
        Self {
            buffer: Some(rustybuzz::UnicodeBuffer::new()),
        }
    }

    /// Fast path for monospace ASCII text.
    ///
    /// Returns `Some` if the text is pure ASCII and the font is monospace.
    /// Each character gets a uniform `cell_width` advance via charmap lookup.
    /// Returns `None` if conditions are not met.
    pub fn shape_monospace_ascii(
        &self,
        face: &FontFace,
        text: &str,
        cell_width: f32,
    ) -> Option<ShapedRun> {
        if !text.is_ascii() || !face.is_monospace() {
            return None;
        }

        let font_ref = face.as_swash_ref();
        let charmap = font_ref.charmap();

        let glyphs: Vec<ShapedGlyph> = text
            .chars()
            .enumerate()
            .map(|(i, c)| {
                let glyph_id = u32::from(charmap.map(c));
                ShapedGlyph {
                    glyph_id,
                    x_offset: 0.0,
                    y_offset: 0.0,
                    x_advance: cell_width,
                    cluster: i as u32,
                }
            })
            .collect();

        Some(ShapedRun { glyphs })
    }

    /// Full rustybuzz shaping path.
    ///
    /// Uses rustybuzz for complex text layout. Positions are scaled from
    /// font units to pixels using `size_px / units_per_em`.
    pub fn shape(&mut self, face: &FontFace, text: &str, size_px: f32) -> ShapedRun {
        let buzz_face = match face.as_rustybuzz_face() {
            Some(f) => f,
            None => {
                tracing::warn!("rustybuzz failed to parse font; returning empty run");
                return ShapedRun { glyphs: Vec::new() };
            }
        };
        let upem = buzz_face.units_per_em() as f32;
        let scale = if upem > 0.0 { size_px / upem } else { 1.0 };

        let mut buffer = self.buffer.take().unwrap_or_default();
        buffer.push_str(text);

        let glyph_buffer = rustybuzz::shape(&buzz_face, &[], buffer);

        let infos = glyph_buffer.glyph_infos();
        let positions = glyph_buffer.glyph_positions();

        let glyphs: Vec<ShapedGlyph> = infos
            .iter()
            .zip(positions.iter())
            .map(|(info, pos)| ShapedGlyph {
                glyph_id: info.glyph_id,
                x_offset: pos.x_offset as f32 * scale,
                y_offset: pos.y_offset as f32 * scale,
                x_advance: pos.x_advance as f32 * scale,
                cluster: info.cluster,
            })
            .collect();

        // Recycle the buffer.
        self.buffer = Some(glyph_buffer.clear());

        ShapedRun { glyphs }
    }

    /// Shape text with OpenType ligature features enabled (liga, calt).
    ///
    /// Same as [`shape`] but passes `liga` and `calt` feature tags to rustybuzz,
    /// enabling programming ligatures in fonts like JetBrains Mono, Fira Code, etc.
    pub fn shape_with_ligatures(&mut self, face: &FontFace, text: &str, size_px: f32) -> ShapedRun {
        let buzz_face = match face.as_rustybuzz_face() {
            Some(f) => f,
            None => {
                tracing::warn!("rustybuzz failed to parse font; returning empty run");
                return ShapedRun { glyphs: Vec::new() };
            }
        };
        let upem = buzz_face.units_per_em() as f32;
        let scale = if upem > 0.0 { size_px / upem } else { 1.0 };

        let features = [
            rustybuzz::Feature::new(rustybuzz::ttf_parser::Tag::from_bytes(b"liga"), 1, ..),
            rustybuzz::Feature::new(rustybuzz::ttf_parser::Tag::from_bytes(b"calt"), 1, ..),
        ];

        let mut buffer = self.buffer.take().unwrap_or_default();
        buffer.push_str(text);

        let glyph_buffer = rustybuzz::shape(&buzz_face, &features, buffer);

        let infos = glyph_buffer.glyph_infos();
        let positions = glyph_buffer.glyph_positions();

        let glyphs: Vec<ShapedGlyph> = infos
            .iter()
            .zip(positions.iter())
            .map(|(info, pos)| ShapedGlyph {
                glyph_id: info.glyph_id,
                x_offset: pos.x_offset as f32 * scale,
                y_offset: pos.y_offset as f32 * scale,
                x_advance: pos.x_advance as f32 * scale,
                cluster: info.cluster,
            })
            .collect();

        self.buffer = Some(glyph_buffer.clear());

        ShapedRun { glyphs }
    }
}

impl Default for TextShaper {
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
    fn shape_hello_produces_five_glyphs() {
        let face = test_face();
        let mut shaper = TextShaper::new();
        let run = shaper.shape(&face, "hello", 16.0);
        assert_eq!(run.glyphs.len(), 5);
    }

    #[test]
    fn shape_empty_produces_zero_glyphs() {
        let face = test_face();
        let mut shaper = TextShaper::new();
        let run = shaper.shape(&face, "", 16.0);
        assert_eq!(run.glyphs.len(), 0);
    }

    #[test]
    fn monospace_ascii_fast_path_returns_some() {
        let face = test_face();
        let shaper = TextShaper::new();
        let run = shaper.shape_monospace_ascii(&face, "hello", 10.0);
        assert!(run.is_some());
        let run = run.unwrap();
        assert_eq!(run.glyphs.len(), 5);
        for g in &run.glyphs {
            assert_eq!(g.x_advance, 10.0);
        }
    }

    #[test]
    fn non_ascii_returns_none_for_fast_path() {
        let face = test_face();
        let shaper = TextShaper::new();
        let run = shaper.shape_monospace_ascii(&face, "héllo", 10.0);
        assert!(run.is_none());
    }

    #[test]
    fn buffer_recycling_works() {
        let face = test_face();
        let mut shaper = TextShaper::new();
        let run1 = shaper.shape(&face, "abc", 16.0);
        let run2 = shaper.shape(&face, "def", 16.0);
        assert_eq!(run1.glyphs.len(), 3);
        assert_eq!(run2.glyphs.len(), 3);
    }

    #[test]
    fn shape_with_ligatures_produces_glyphs() {
        let face = test_face();
        let mut shaper = TextShaper::new();
        let run = shaper.shape_with_ligatures(&face, "=>", 16.0);
        // JetBrains Mono has a ligature for "=>" — should produce fewer glyphs.
        assert!(!run.glyphs.is_empty());
        assert!(
            run.glyphs.len() <= 2,
            "expected ligature to reduce glyph count, got {}",
            run.glyphs.len()
        );
    }

    #[test]
    fn shape_without_ligatures_no_merge() {
        let face = test_face();
        let mut shaper = TextShaper::new();
        let run = shaper.shape(&face, "=>", 16.0);
        // Without ligature features, should produce exactly 2 glyphs.
        assert_eq!(run.glyphs.len(), 2);
    }
}
