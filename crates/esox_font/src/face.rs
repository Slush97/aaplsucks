//! Font face loading and validation.

use crate::{Error, FontId, FontMetrics};

/// Maximum font file size (256 MiB).
const MAX_FONT_SIZE: usize = 256 * 1024 * 1024;

/// Minimum font file size (12 bytes for the offset table header).
const MIN_FONT_SIZE: usize = 12;

/// A loaded and validated font face with its raw data.
///
/// Stores the original font bytes and pre-computed metadata for efficient
/// access to swash/rustybuzz APIs.
pub struct FontFace {
    id: FontId,
    data: Vec<u8>,
    cache_key: swash::CacheKey,
    offset: u32,
    is_monospace: bool,
}

impl std::fmt::Debug for FontFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontFace")
            .field("id", &self.id)
            .field("data_len", &self.data.len())
            .field("is_monospace", &self.is_monospace)
            .finish()
    }
}

impl FontFace {
    /// Load a font face from raw bytes.
    ///
    /// Validates the data with both swash and rustybuzz, extracts cache key,
    /// offset, and monospace flag.
    pub fn from_bytes(id: FontId, data: Vec<u8>) -> Result<Self, Error> {
        if data.len() < MIN_FONT_SIZE {
            return Err(Error::Load(format!(
                "font data too small ({} bytes, minimum {})",
                data.len(),
                MIN_FONT_SIZE
            )));
        }

        if data.len() > MAX_FONT_SIZE {
            return Err(Error::Load(format!(
                "font data too large ({} bytes, maximum {})",
                data.len(),
                MAX_FONT_SIZE
            )));
        }

        // Validate with swash.
        let font_data = swash::FontDataRef::new(&data)
            .ok_or_else(|| Error::Load("swash: invalid font data".into()))?;
        let font_ref = font_data
            .get(0)
            .ok_or_else(|| Error::Load("swash: no font at index 0".into()))?;

        let cache_key = font_ref.key;
        let offset = font_ref.offset;

        // Validate with rustybuzz.
        rustybuzz::Face::from_slice(&data, 0)
            .ok_or_else(|| Error::Load("rustybuzz: failed to parse font".into()))?;

        // Extract monospace flag from swash metrics.
        let metrics = font_ref.metrics(&[]);
        let is_monospace = metrics.is_monospace;

        Ok(Self {
            id,
            data,
            cache_key,
            offset,
            is_monospace,
        })
    }

    /// Create a short-lived swash `FontRef` for this face.
    pub fn as_swash_ref(&self) -> swash::FontRef<'_> {
        swash::FontRef {
            data: &self.data,
            offset: self.offset,
            key: self.cache_key,
        }
    }

    /// Create a short-lived rustybuzz `Face` for this face.
    ///
    /// Returns `None` if rustybuzz cannot parse the font data (should not happen
    /// for faces created via `from_bytes`, but avoids panicking).
    pub fn as_rustybuzz_face(&self) -> Option<rustybuzz::Face<'_>> {
        rustybuzz::Face::from_slice(&self.data, 0)
    }

    /// Compute font metrics at the given pixel size.
    pub fn metrics(&self, size_px: f32) -> FontMetrics {
        let font_ref = self.as_swash_ref();
        let raw = font_ref.metrics(&[]);
        let scaled = raw.scale(size_px);

        let ascent = scaled.ascent.round();
        let descent = scaled.descent;
        let leading = scaled.leading;
        let average_width = if scaled.average_width > 0.0 {
            scaled.average_width
        } else {
            // Fallback: use max_width or half the em size.
            if scaled.max_width > 0.0 {
                scaled.max_width
            } else {
                size_px * 0.5
            }
        };

        // Swash's `descent` is the distance below baseline. It can be positive
        // (swash convention) or negative (OpenType raw). Use the absolute value
        // to ensure cell_height = ascent + |descent| + leading.
        let abs_descent = descent.abs();

        FontMetrics {
            cell_width: average_width.ceil(),
            cell_height: (ascent + abs_descent + leading).ceil(),
            ascent,
            descent,
            underline_offset: scaled.underline_offset,
            stroke_size: scaled.stroke_size.max(1.0),
            strikeout_offset: scaled.strikeout_offset,
        }
    }

    /// Check if this font has a glyph for the given character.
    pub fn has_glyph(&self, c: char) -> bool {
        let font_ref = self.as_swash_ref();
        let charmap = font_ref.charmap();
        charmap.map(c) != 0
    }

    /// Whether this font is monospace.
    pub fn is_monospace(&self) -> bool {
        self.is_monospace
    }

    /// The unique identifier for this face.
    pub fn id(&self) -> FontId {
        self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_font_data() -> Vec<u8> {
        std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-data/JetBrainsMono-Regular.ttf"
        ))
        .expect("test font not found — run from repo root")
    }

    #[test]
    fn empty_data_returns_error() {
        let result = FontFace::from_bytes(FontId(0), vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn garbage_data_returns_error() {
        let result = FontFace::from_bytes(FontId(0), vec![0xDE; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn valid_font_loads() {
        let data = test_font_data();
        let face = FontFace::from_bytes(FontId(0), data).unwrap();
        assert_eq!(face.id(), FontId(0));
    }

    #[test]
    fn metrics_are_positive() {
        let data = test_font_data();
        let face = FontFace::from_bytes(FontId(0), data).unwrap();
        let m = face.metrics(16.0);
        assert!(m.cell_width > 0.0, "cell_width={}", m.cell_width);
        assert!(m.cell_height > 0.0, "cell_height={}", m.cell_height);
        assert!(m.ascent > 0.0, "ascent={}", m.ascent);
        assert!(m.stroke_size >= 1.0, "stroke_size={}", m.stroke_size);
    }

    #[test]
    fn has_glyph_ascii() {
        let data = test_font_data();
        let face = FontFace::from_bytes(FontId(0), data).unwrap();
        assert!(face.has_glyph('A'));
        assert!(face.has_glyph('z'));
        assert!(face.has_glyph('0'));
    }

    #[test]
    fn has_glyph_rare_codepoint() {
        let data = test_font_data();
        let face = FontFace::from_bytes(FontId(0), data).unwrap();
        // U+10FFFD is a private-use character unlikely to be in JetBrains Mono.
        assert!(!face.has_glyph('\u{10FFFD}'));
    }

    #[test]
    fn is_monospace_flag() {
        let data = test_font_data();
        let face = FontFace::from_bytes(FontId(0), data).unwrap();
        assert!(face.is_monospace());
    }

    #[test]
    fn cell_height_accounts_for_descent_sign() {
        let data = test_font_data();
        let face = FontFace::from_bytes(FontId(0), data).unwrap();
        let m = face.metrics(16.0);
        // cell_height must be at least ascent (descent can be negative in some APIs).
        assert!(
            m.cell_height >= m.ascent,
            "cell_height ({}) < ascent ({})",
            m.cell_height,
            m.ascent
        );
        // cell_height should be ascent + |descent| + leading, so always > ascent alone.
        assert!(
            m.cell_height > m.ascent,
            "cell_height ({}) should exceed ascent ({}) due to descent",
            m.cell_height,
            m.ascent
        );
    }

    #[test]
    fn size_cap_enforcement() {
        let huge = vec![0u8; MAX_FONT_SIZE + 1];
        let result = FontFace::from_bytes(FontId(0), huge);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too large"));
    }
}
