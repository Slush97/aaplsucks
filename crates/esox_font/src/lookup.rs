//! System font lookup via `fc-match` (fontconfig CLI).
//!
//! Resolves family names to font file paths without loading every system font
//! into memory at startup. Falls back to probing well-known font directories
//! if `fc-match` is unavailable.

use std::path::{Path, PathBuf};
use std::process::Command;

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

/// Directories to probe when fc-match is unavailable.
const FONT_DIRS: &[&str] = &[
    "/usr/share/fonts",
    "/usr/local/share/fonts",
];

/// Result of a system font query — the raw font file bytes.
pub struct FontMatch {
    /// The raw font file data.
    pub data: Vec<u8>,
    /// The family name that matched.
    pub family: String,
}

/// A system font resolver that uses `fc-match` to find fonts.
///
/// Unlike the previous fontdb-based approach, this does not scan every font at
/// startup — it shells out per query, which is near-instant (~2-5ms each).
pub struct SystemFontDb {
    /// Whether fc-match is available on this system.
    fc_available: bool,
}

impl SystemFontDb {
    /// Create a new font resolver, probing for fc-match availability.
    pub fn new() -> Self {
        let fc_available = Command::new("fc-match")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if fc_available {
            tracing::info!("font resolver: using fc-match");
        } else {
            tracing::warn!("fc-match not found, falling back to directory probing");
        }

        Self { fc_available }
    }

    /// Look up a font by family name and style.
    pub fn query_family(&self, family: &str, style: FontStyle) -> Option<FontMatch> {
        let style_str = match style {
            FontStyle::Regular => "Regular",
            FontStyle::Bold => "Bold",
            FontStyle::Italic => "Italic",
            FontStyle::BoldItalic => "Bold Italic",
        };

        if self.fc_available {
            self.query_fc_match(family, style_str)
        } else {
            self.query_probe(family)
        }
    }

    /// Resolve the generic "monospace" family to a concrete system font.
    pub fn query_monospace(&self, style: FontStyle) -> Option<FontMatch> {
        // Try "monospace" generic first (fc-match handles this natively).
        if let Some(m) = self.query_family("monospace", style) {
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

    /// Find symbol, emoji, CJK, and Nerd Font fallbacks on the system.
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

        for &family in &well_known {
            if let Some(m) = self.query_family(family, FontStyle::Regular) {
                tracing::debug!(family = m.family, "found well-known fallback font");
                seen_families.insert(m.family.clone());
                results.push(m);
            }
        }

        // Scan for Nerd Font variants via fc-list.
        if self.fc_available {
            if let Ok(output) = Command::new("fc-list")
                .args(["--format", "%{family}\n"])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    // fc-list may return comma-separated family names.
                    for family_name in line.split(',') {
                        let family_name = family_name.trim();
                        if family_name.contains("Nerd Font")
                            && !seen_families.contains(family_name)
                        {
                            if let Some(m) =
                                self.query_family(family_name, FontStyle::Regular)
                            {
                                tracing::debug!(
                                    family = m.family,
                                    "found Nerd Font fallback"
                                );
                                seen_families.insert(m.family.clone());
                                results.push(m);
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Find a font that contains the given codepoint.
    ///
    /// Uses `fc-match` with charset matching to find a font covering `c`.
    pub fn query_codepoint(&self, c: char) -> Option<FontMatch> {
        if !self.fc_available {
            return None;
        }

        let codepoint = c as u32;
        // fc-match can match by charset.
        let pattern = format!("charset={codepoint:04x}");
        let output = Command::new("fc-match")
            .args([&pattern, "--format", "%{file}|%{family}\n"])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().next()?;
        let (path_str, family) = line.split_once('|')?;

        let path = Path::new(path_str.trim());
        if !path.exists() {
            return None;
        }

        let data = std::fs::read(path).ok()?;
        tracing::debug!(family = %family.trim(), codepoint = ?c, "dynamic fallback found");
        Some(FontMatch {
            data,
            family: family.trim().to_string(),
        })
    }

    /// Query fc-match for a font file path.
    fn query_fc_match(&self, family: &str, style: &str) -> Option<FontMatch> {
        let pattern = format!("{family}:style={style}");
        let output = Command::new("fc-match")
            .args([&pattern, "--format", "%{file}|%{family}\n"])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout.lines().next()?;
        let (path_str, family_out) = line.split_once('|')?;

        let path = Path::new(path_str.trim());
        if !path.exists() {
            tracing::warn!(path = %path.display(), "fc-match returned non-existent path");
            return None;
        }

        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(path = %path.display(), "failed to read font file: {e}");
                return None;
            }
        };

        Some(FontMatch {
            data,
            family: family_out.trim().to_string(),
        })
    }

    /// Probe well-known font directories for a matching file.
    fn query_probe(&self, family: &str) -> Option<FontMatch> {
        // Build candidate filenames from the family name.
        let normalized = family.replace(' ', "");
        let candidates = [
            format!("{normalized}.ttf"),
            format!("{normalized}-Regular.ttf"),
            format!("{normalized}.otf"),
            format!("{normalized}-Regular.otf"),
        ];

        for dir in FONT_DIRS {
            let base = Path::new(dir);
            if !base.exists() {
                continue;
            }
            for candidate in &candidates {
                if let Some(m) = self.probe_recursive(base, candidate, family) {
                    return Some(m);
                }
            }
        }
        None
    }

    /// Recursively search a directory for a font file matching `filename`.
    fn probe_recursive(&self, dir: &Path, filename: &str, family: &str) -> Option<FontMatch> {
        let entries = std::fs::read_dir(dir).ok()?;
        let mut subdirs: Vec<PathBuf> = Vec::new();

        for entry in entries.flatten() {
            let ft = entry.file_type().ok()?;
            if ft.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.eq_ignore_ascii_case(filename) {
                        let data = std::fs::read(entry.path()).ok()?;
                        tracing::info!(
                            path = %entry.path().display(),
                            "found font via directory probe"
                        );
                        return Some(FontMatch {
                            data,
                            family: family.to_string(),
                        });
                    }
                }
            } else if ft.is_dir() {
                subdirs.push(entry.path());
            }
        }

        for sub in subdirs {
            if let Some(m) = self.probe_recursive(&sub, filename, family) {
                return Some(m);
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
        // Should construct without panicking.
        let _ = db;
    }

    #[test]
    fn monospace_resolves() {
        let db = SystemFontDb::new();
        if let Some(m) = db.query_monospace(FontStyle::Regular) {
            assert!(!m.data.is_empty());
            tracing::info!(family = m.family, "resolved monospace");
        }
    }

    #[test]
    fn nonexistent_family_falls_back() {
        let db = SystemFontDb::new();
        let result = db.resolve("ZZZNonexistentFontFamily999", FontStyle::Regular);
        if let Some(m) = result {
            assert!(!m.data.is_empty());
        }
    }
}
