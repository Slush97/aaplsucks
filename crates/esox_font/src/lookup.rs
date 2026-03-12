//! System font lookup via `fontdb`.
//!
//! Scans system font directories and resolves family names to font file data.

use crate::Error;

/// Common monospace fallback families, tried in order.
const MONOSPACE_FALLBACKS: &[&str] = &[
    "DejaVu Sans Mono",
    "Noto Sans Mono",
    "Liberation Mono",
    "Fira Code",
    "Source Code Pro",
    "Inconsolata",
    "Droid Sans Mono",
    "Cascadia Code",
    "Consolas",
    "Menlo",
    "Courier New",
];

/// Result of a system font query — the raw font file bytes.
pub struct FontMatch {
    /// The raw font file data.
    pub data: Vec<u8>,
    /// The family name that matched.
    pub family: String,
}

/// A system font database that resolves family names to font data.
pub struct SystemFontDb {
    db: fontdb::Database,
}

impl SystemFontDb {
    /// Create a new database loaded with system fonts.
    pub fn new() -> Self {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        let count = db.len();
        tracing::info!(fonts = count, "loaded system font database");
        Self { db }
    }

    /// Look up a font by family name and style.
    ///
    /// Returns the font file data if found.
    pub fn query_family(&self, family: &str, style: FontStyle) -> Option<FontMatch> {
        let weight = match style {
            FontStyle::Regular | FontStyle::Italic => fontdb::Weight::NORMAL,
            FontStyle::Bold | FontStyle::BoldItalic => fontdb::Weight::BOLD,
        };
        let font_style = match style {
            FontStyle::Regular | FontStyle::Bold => fontdb::Style::Normal,
            FontStyle::Italic | FontStyle::BoldItalic => fontdb::Style::Italic,
        };

        let query = fontdb::Query {
            families: &[fontdb::Family::Name(family)],
            weight,
            style: font_style,
            ..fontdb::Query::default()
        };

        let id = self.db.query(&query)?;
        self.load_face_data(id, family)
    }

    /// Resolve the generic "monospace" family to a concrete system font.
    pub fn query_monospace(&self, style: FontStyle) -> Option<FontMatch> {
        // Try fontdb's generic monospace first.
        let weight = match style {
            FontStyle::Regular | FontStyle::Italic => fontdb::Weight::NORMAL,
            FontStyle::Bold | FontStyle::BoldItalic => fontdb::Weight::BOLD,
        };
        let font_style = match style {
            FontStyle::Regular | FontStyle::Bold => fontdb::Style::Normal,
            FontStyle::Italic | FontStyle::BoldItalic => fontdb::Style::Italic,
        };

        let query = fontdb::Query {
            families: &[fontdb::Family::Monospace],
            weight,
            style: font_style,
            ..fontdb::Query::default()
        };

        if let Some(id) = self.db.query(&query)
            && let Some(m) = self.load_face_data(id, "monospace")
        {
            return Some(m);
        }

        // Fall through to explicit fallback list.
        for &name in MONOSPACE_FALLBACKS {
            if let Some(m) = self.query_family(name, style) {
                tracing::info!(family = name, "resolved monospace via fallback list");
                return Some(m);
            }
        }

        None
    }

    /// Resolve a family name, handling "monospace" as a generic alias.
    pub fn resolve(&self, family: &str, style: FontStyle) -> Option<FontMatch> {
        if family.eq_ignore_ascii_case("monospace") {
            return self.query_monospace(style);
        }

        // Try exact family first.
        if let Some(m) = self.query_family(family, style) {
            return Some(m);
        }

        // If the requested family wasn't found, try monospace fallback.
        tracing::warn!(
            family = family,
            "font family not found, falling back to system monospace"
        );
        self.query_monospace(style)
    }

    /// Load font file data for a given face ID.
    fn load_face_data(&self, id: fontdb::ID, query_family: &str) -> Option<FontMatch> {
        let (src, face_index) = self.db.face_source(id)?;
        if face_index != 0 {
            // We only support single-face files for now (index 0).
            // TTC/OTC files would need face index threading.
            tracing::debug!(face_index, "skipping non-zero face index in collection");
        }

        let data = match src {
            fontdb::Source::Binary(arc) => arc.as_ref().as_ref().to_vec(),
            fontdb::Source::File(path) => match std::fs::read(&path) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(path = %path.display(), "failed to read font file: {e}");
                    return None;
                }
            },
            fontdb::Source::SharedFile(_path, data) => data.as_ref().as_ref().to_vec(),
        };

        // Resolve actual family name from the database.
        let actual_family = self
            .db
            .face(id)
            .and_then(|info| info.families.first().map(|(name, _)| name.clone()))
            .unwrap_or_else(|| query_family.to_string());

        Some(FontMatch {
            data,
            family: actual_family,
        })
    }

    /// Find symbol, emoji, CJK, and Nerd Font fallbacks on the system.
    ///
    /// Searches for well-known fallback families first, then scans the
    /// database for any font with "Nerd Font" in its family name so users
    /// don't have to configure Nerd Font paths explicitly.
    pub fn find_fallback_fonts(&self) -> Vec<FontMatch> {
        let well_known = [
            "Symbols Nerd Font Mono",
            "Symbols Nerd Font",
            "Noto Color Emoji",
            "Noto Sans Symbols",
            "Noto Sans Symbols2",
            "Noto Sans CJK",
        ];

        let mut results = Vec::new();
        let mut seen_families = std::collections::HashSet::new();

        // Well-known fallbacks first (highest priority).
        for &family in &well_known {
            if let Some(m) = self.query_family(family, FontStyle::Regular) {
                tracing::debug!(family = m.family, "found well-known fallback font");
                seen_families.insert(m.family.clone());
                results.push(m);
            }
        }

        // Scan for any Nerd Font variant installed on the system.
        for face_info in self.db.faces() {
            for (family_name, _) in &face_info.families {
                if family_name.contains("Nerd Font") && !seen_families.contains(family_name) {
                    // Prefer Mono variants for terminal use.
                    if let Some(m) = self.query_family(family_name, FontStyle::Regular) {
                        tracing::debug!(family = m.family, "found Nerd Font fallback");
                        seen_families.insert(m.family.clone());
                        results.push(m);
                    }
                    break;
                }
            }
        }

        results
    }

    /// Find a font that contains the given codepoint.
    ///
    /// Scans all loaded system fonts for one whose cmap includes `c`.
    /// Returns the font data if found. This enables on-demand fallback
    /// for codepoints not covered by the static fallback list.
    pub fn query_codepoint(&self, c: char) -> Option<FontMatch> {
        for face_info in self.db.faces() {
            // Only consider Regular weight, Normal style to avoid duplicates.
            if face_info.weight != fontdb::Weight::NORMAL
                || face_info.style != fontdb::Style::Normal
            {
                continue;
            }
            // Check if this font's cmap contains the codepoint by loading it.
            let (src, _face_index) = self.db.face_source(face_info.id)?;
            let data = match src {
                fontdb::Source::Binary(arc) => arc.as_ref().as_ref().to_vec(),
                fontdb::Source::File(path) => std::fs::read(&path).ok()?,
                fontdb::Source::SharedFile(_path, data) => data.as_ref().as_ref().to_vec(),
            };
            // Use swash to check charmap without fully parsing.
            if let Some(font_data) = swash::FontDataRef::new(&data)
                && let Some(font_ref) = font_data.get(0)
                && font_ref.charmap().map(c) != 0
            {
                let family = face_info
                    .families
                    .first()
                    .map(|(n, _)| n.clone())
                    .unwrap_or_default();
                tracing::debug!(family = %family, codepoint = ?c, "dynamic fallback found");
                return Some(FontMatch { data, family });
            }
        }
        None
    }
}

impl Default for SystemFontDb {
    fn default() -> Self {
        Self::new()
    }
}

/// Font style variant for queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontStyle {
    /// Normal weight, upright.
    Regular,
    /// Bold weight, upright.
    Bold,
    /// Normal weight, italic.
    Italic,
    /// Bold weight, italic.
    BoldItalic,
}

/// Resolve a font family name to raw font data.
///
/// This is a convenience function that creates a temporary `SystemFontDb`,
/// resolves the family, and returns the data. For repeated lookups, prefer
/// creating a `SystemFontDb` once and reusing it.
pub fn resolve_family(family: &str) -> Result<Vec<u8>, Error> {
    let db = SystemFontDb::new();
    db.resolve(family, FontStyle::Regular)
        .map(|m| m.data)
        .ok_or_else(|| Error::Load(format!("font family '{family}' not found on system")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_font_db_loads() {
        let db = SystemFontDb::new();
        // Should have found at least some fonts on any system with fonts installed.
        // This test might fail in a bare container, so just check it doesn't panic.
        let _ = db;
    }

    #[test]
    fn monospace_resolves() {
        let db = SystemFontDb::new();
        // On most systems there's some monospace font. If not, this is expected to return None.
        if let Some(m) = db.query_monospace(FontStyle::Regular) {
            assert!(!m.data.is_empty());
            tracing::info!(family = m.family, "resolved monospace");
        }
    }

    #[test]
    fn nonexistent_family_falls_back() {
        let db = SystemFontDb::new();
        // A made-up family should fall back to monospace.
        let result = db.resolve("ZZZNonexistentFontFamily999", FontStyle::Regular);
        // If monospace exists on the system, we get a fallback.
        if let Some(m) = result {
            assert!(!m.data.is_empty());
        }
    }
}
