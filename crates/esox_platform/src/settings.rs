//! Settings persistence via TOML files in XDG config directories.
//!
//! Requires the `settings` feature (enables `serde` + `toml`).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::xdg::AppDirs;

/// Saved window geometry for session restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowState {
    pub width: u32,
    pub height: u32,
    pub x: Option<i32>,
    pub y: Option<i32>,
}

impl WindowState {
    /// Capture the current window geometry from a winit window.
    pub fn from_window(window: &winit::window::Window) -> Self {
        let size = window.inner_size();
        let pos = window.outer_position().ok();
        Self {
            width: size.width,
            height: size.height,
            x: pos.map(|p| p.x),
            y: pos.map(|p| p.y),
        }
    }

    /// Apply saved state to a `WindowConfig`, overriding size and position.
    pub fn apply_to(&self, config: &mut crate::config::WindowConfig) {
        config.width = Some(self.width);
        config.height = Some(self.height);
        if let (Some(x), Some(y)) = (self.x, self.y) {
            config.position = Some((x, y));
        }
    }

    /// Load window state from `<config_dir>/window.toml`.
    pub fn load(dirs: &AppDirs) -> Option<Self> {
        let path = dirs.config_dir().join("window.toml");
        let content = std::fs::read_to_string(&path).ok()?;
        toml::from_str(&content).ok()
    }

    /// Save window state to `<config_dir>/window.toml`.
    pub fn save(&self, dirs: &AppDirs) -> std::io::Result<()> {
        let dir = dirs.config_dir();
        std::fs::create_dir_all(&dir)?;
        let content = toml::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;
        std::fs::write(dir.join("window.toml"), content)
    }
}

/// Load an app-specific settings file from `<config_dir>/<filename>`.
pub fn load_settings<T: for<'de> Deserialize<'de>>(dirs: &AppDirs, filename: &str) -> Option<T> {
    let path = dirs.config_dir().join(filename);
    let content = std::fs::read_to_string(&path).ok()?;
    toml::from_str(&content).ok()
}

/// Save an app-specific settings file to `<config_dir>/<filename>`.
pub fn save_settings<T: Serialize>(dirs: &AppDirs, filename: &str, value: &T) -> std::io::Result<()> {
    let dir = dirs.config_dir();
    std::fs::create_dir_all(&dir)?;
    let content = toml::to_string_pretty(value).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e)
    })?;
    std::fs::write(dir.join(filename), content)
}

/// Resolve the settings config path for an app. Returns `None` if no app_name is set.
pub fn settings_path(config: &crate::config::PlatformConfig) -> Option<PathBuf> {
    config.app_name.as_ref().map(|name| AppDirs::new(name).config_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_state_roundtrip() {
        let state = WindowState {
            width: 1280,
            height: 720,
            x: Some(100),
            y: Some(200),
        };
        let toml_str = toml::to_string_pretty(&state).unwrap();
        let loaded: WindowState = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded.width, 1280);
        assert_eq!(loaded.height, 720);
        assert_eq!(loaded.x, Some(100));
        assert_eq!(loaded.y, Some(200));
    }

    #[test]
    fn window_state_apply_to_config() {
        let state = WindowState {
            width: 1920,
            height: 1080,
            x: Some(50),
            y: Some(75),
        };
        let mut config = crate::config::WindowConfig::default();
        state.apply_to(&mut config);
        assert_eq!(config.width, Some(1920));
        assert_eq!(config.height, Some(1080));
        assert_eq!(config.position, Some((50, 75)));
    }

    #[test]
    fn generic_settings_roundtrip() {
        #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
        struct MySettings {
            font_size: f32,
            theme: String,
        }
        let settings = MySettings {
            font_size: 14.0,
            theme: "dark".into(),
        };
        let toml_str = toml::to_string_pretty(&settings).unwrap();
        let loaded: MySettings = toml::from_str(&toml_str).unwrap();
        assert_eq!(loaded, settings);
    }
}
