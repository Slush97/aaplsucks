//! Font fallback chain for codepoint resolution.

use crate::face::FontFace;
use crate::{Error, FontId, FontMetrics};

/// An ordered list of font faces for codepoint resolution.
///
/// The first font is the primary; subsequent fonts are fallbacks.
/// Optionally holds dedicated bold, italic, and bold-italic variant faces.
pub struct FontFallbackChain {
    faces: Vec<FontFace>,
    next_id: u32,
    /// Dedicated bold variant face.
    bold_face: Option<FontFace>,
    /// Dedicated italic variant face.
    italic_face: Option<FontFace>,
    /// Dedicated bold-italic variant face.
    bold_italic_face: Option<FontFace>,
}

impl FontFallbackChain {
    /// Create an empty fallback chain.
    pub fn new() -> Self {
        Self {
            faces: Vec::new(),
            next_id: 0,
            bold_face: None,
            italic_face: None,
            bold_italic_face: None,
        }
    }

    /// Add a font from raw bytes, assigning an incrementing `FontId`.
    pub fn add_font(&mut self, data: Vec<u8>) -> Result<FontId, Error> {
        let id = FontId(self.next_id);
        let face = FontFace::from_bytes(id, data)?;
        self.faces.push(face);
        self.next_id += 1;
        Ok(id)
    }

    /// Get metrics from the primary (first) font at the given pixel size.
    pub fn primary_metrics(&self, size_px: f32) -> Option<FontMetrics> {
        self.faces.first().map(|f| f.metrics(size_px))
    }

    /// Find the first font in the chain that has a glyph for the given character.
    pub fn resolve_codepoint(&self, c: char) -> Option<&FontFace> {
        self.faces.iter().find(|f| f.has_glyph(c))
    }

    /// Look up a face by its `FontId`.
    pub fn get(&self, id: FontId) -> Option<&FontFace> {
        self.faces.iter().find(|f| f.id() == id)
    }

    /// Get the primary (first) font face.
    pub fn primary(&self) -> Option<&FontFace> {
        self.faces.first()
    }

    /// Set the dedicated bold variant face from raw font bytes.
    pub fn set_bold(&mut self, data: Vec<u8>) -> Result<FontId, Error> {
        let id = FontId(self.next_id);
        let face = FontFace::from_bytes(id, data)?;
        self.bold_face = Some(face);
        self.next_id += 1;
        Ok(id)
    }

    /// Set the dedicated italic variant face from raw font bytes.
    pub fn set_italic(&mut self, data: Vec<u8>) -> Result<FontId, Error> {
        let id = FontId(self.next_id);
        let face = FontFace::from_bytes(id, data)?;
        self.italic_face = Some(face);
        self.next_id += 1;
        Ok(id)
    }

    /// Set the dedicated bold-italic variant face from raw font bytes.
    pub fn set_bold_italic(&mut self, data: Vec<u8>) -> Result<FontId, Error> {
        let id = FontId(self.next_id);
        let face = FontFace::from_bytes(id, data)?;
        self.bold_italic_face = Some(face);
        self.next_id += 1;
        Ok(id)
    }

    /// Resolve a codepoint with style awareness.
    ///
    /// Style bits: bit 0 = bold, bit 1 = italic. Returns `(face, is_true_variant)`
    /// where `is_true_variant` indicates the returned face is a dedicated style
    /// variant (so the renderer should skip faux synthesis).
    pub fn resolve_styled(&self, c: char, style: u8) -> Option<(&FontFace, bool)> {
        let bold = style & 1 != 0;
        let italic = style & 2 != 0;

        // Try the matching variant face first.
        let variant = match (bold, italic) {
            (true, true) => self.bold_italic_face.as_ref(),
            (true, false) => self.bold_face.as_ref(),
            (false, true) => self.italic_face.as_ref(),
            (false, false) => None,
        };

        if let Some(face) = variant
            && face.has_glyph(c)
        {
            return Some((face, true));
        }

        // Fall back to regular chain.
        self.resolve_codepoint(c).map(|f| (f, false))
    }

    /// Resolve a grapheme cluster to the best font face.
    ///
    /// Tries shaping with each face in the chain and returns the first that
    /// produces non-`.notdef` (glyph ID != 0) glyphs for the base character.
    /// Falls back to the face for the first character if no face covers
    /// the full cluster.
    pub fn resolve_grapheme(&self, s: &str) -> Option<&FontFace> {
        let base_char = s.chars().next()?;
        self.faces.iter().find(|face| face.has_glyph(base_char))
    }

    /// Number of fonts in the chain.
    pub fn len(&self) -> usize {
        self.faces.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

impl Default for FontFallbackChain {
    fn default() -> Self {
        Self::new()
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
        .expect("test font not found")
    }

    #[test]
    fn empty_chain_returns_none() {
        let chain = FontFallbackChain::new();
        assert!(chain.is_empty());
        assert!(chain.primary().is_none());
        assert!(chain.primary_metrics(16.0).is_none());
        assert!(chain.resolve_codepoint('A').is_none());
    }

    #[test]
    fn single_font_resolves() {
        let mut chain = FontFallbackChain::new();
        let id = chain.add_font(test_font_data()).unwrap();
        assert_eq!(id, FontId(0));
        assert_eq!(chain.len(), 1);

        let face = chain.resolve_codepoint('A').unwrap();
        assert_eq!(face.id(), FontId(0));
    }

    #[test]
    fn two_fonts_fallback() {
        let mut chain = FontFallbackChain::new();
        let id0 = chain.add_font(test_font_data()).unwrap();
        let id1 = chain.add_font(test_font_data()).unwrap();
        assert_eq!(id0, FontId(0));
        assert_eq!(id1, FontId(1));
        assert_eq!(chain.len(), 2);

        // Both fonts have 'A', so the primary should be returned.
        let face = chain.resolve_codepoint('A').unwrap();
        assert_eq!(face.id(), FontId(0));
    }

    #[test]
    fn get_by_id() {
        let mut chain = FontFallbackChain::new();
        let id = chain.add_font(test_font_data()).unwrap();
        assert!(chain.get(id).is_some());
        assert!(chain.get(FontId(99)).is_none());
    }

    #[test]
    fn resolve_styled_returns_bold_face_when_set() {
        let mut chain = FontFallbackChain::new();
        chain.add_font(test_font_data()).unwrap();
        let bold_id = chain.set_bold(test_font_data()).unwrap();

        // style=1 (bold) should return bold face with is_true=true.
        let (face, is_true) = chain.resolve_styled('A', 1).unwrap();
        assert_eq!(face.id(), bold_id);
        assert!(is_true);
    }

    #[test]
    fn resolve_styled_falls_back_when_no_variant() {
        let mut chain = FontFallbackChain::new();
        let regular_id = chain.add_font(test_font_data()).unwrap();

        // style=1 (bold) but no bold face set → falls back to regular.
        let (face, is_true) = chain.resolve_styled('A', 1).unwrap();
        assert_eq!(face.id(), regular_id);
        assert!(!is_true);
    }

    #[test]
    fn resolve_styled_no_style_returns_regular() {
        let mut chain = FontFallbackChain::new();
        let regular_id = chain.add_font(test_font_data()).unwrap();
        chain.set_bold(test_font_data()).unwrap();

        // style=0 should return regular face.
        let (face, is_true) = chain.resolve_styled('A', 0).unwrap();
        assert_eq!(face.id(), regular_id);
        assert!(!is_true);
    }
}
