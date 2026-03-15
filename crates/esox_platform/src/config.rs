//! Platform configuration types.
//!
//! These replace the terminal-specific `esox_config` with a minimal set of
//! settings needed by the platform layer.

/// Frame timing configuration.
#[derive(Debug, Clone)]
pub struct FrameConfig {
    /// Maximum render FPS. `None` = match monitor refresh rate.
    pub max_fps: Option<u32>,
    /// Fixed game-logic update rate in Hz (default: 60).
    pub tick_rate: f32,
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            max_fps: None,
            tick_rate: 60.0,
        }
    }
}

/// Top-level platform configuration.
#[derive(Debug, Clone)]
pub struct PlatformConfig {
    /// Window properties.
    pub window: WindowConfig,
    /// Background opacity (0.0 = fully transparent, 1.0 = opaque).
    pub opacity: f32,
    /// Background clear color as a hex string (e.g. "#2b2b2b").
    pub background: String,
    /// Whether to request an HDR (Rgba16Float) surface format.
    pub hdr: bool,
    /// Multisample anti-aliasing sample count (1 = off, 4 = 4x MSAA).
    pub msaa: u32,
    /// Security settings.
    pub security: SecurityConfig,
    /// Accessibility settings.
    pub accessibility: AccessibilityConfig,
    /// Frame timing settings.
    pub frame: FrameConfig,
    /// Application name for XDG directories and settings persistence.
    ///
    /// When set, enables auto-save/restore of window state via `settings` feature.
    pub app_name: Option<String>,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            window: WindowConfig::default(),
            opacity: 1.0,
            background: "#2b2b2b".into(),
            hdr: false,
            msaa: 4,
            security: SecurityConfig::default(),
            accessibility: AccessibilityConfig::default(),
            frame: FrameConfig::default(),
            app_name: None,
        }
    }
}

/// Window configuration.
#[derive(Debug, Clone)]
pub struct WindowConfig {
    /// Window title.
    pub title: String,
    /// Whether to show window manager decorations.
    pub decorations: bool,
    /// Initial window width in logical pixels (None = OS default).
    pub width: Option<u32>,
    /// Initial window height in logical pixels (None = OS default).
    pub height: Option<u32>,
    /// Initial window position (x, y) in logical pixels (None = OS-placed).
    pub position: Option<(i32, i32)>,
    /// Window icon as raw RGBA pixel data.
    pub icon_rgba: Option<IconData>,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "eso".into(),
            decorations: true,
            width: None,
            height: None,
            position: None,
            icon_rgba: None,
        }
    }
}

/// Raw RGBA icon data for the window.
#[derive(Debug, Clone)]
pub struct IconData {
    /// RGBA pixel data.
    pub rgba: Vec<u8>,
    /// Icon width in pixels.
    pub width: u32,
    /// Icon height in pixels.
    pub height: u32,
}

/// Accessibility settings detected from the system or user-configured.
#[derive(Debug, Clone)]
pub struct AccessibilityConfig {
    /// Whether high-contrast mode is active.
    pub high_contrast: bool,
    /// Whether accessibility features are enabled.
    pub enabled: bool,
}

impl AccessibilityConfig {
    /// Detect accessibility preferences from the system.
    ///
    /// Reads GNOME/GTK settings via `gsettings`. Returns defaults if
    /// gsettings is unavailable or the keys don't exist.
    pub fn from_system() -> Self {
        let high_contrast = Self::read_gsettings_high_contrast();
        let enabled = high_contrast || Self::read_gsettings_a11y_enabled();

        if high_contrast {
            tracing::info!("system high-contrast mode detected");
        }
        if enabled {
            tracing::info!("accessibility features enabled");
        }

        Self {
            high_contrast,
            enabled,
        }
    }

    fn read_gsettings_high_contrast() -> bool {
        // Check GTK theme name for "HighContrast".
        if let Ok(output) = std::process::Command::new("gsettings")
            .args(["get", "org.gnome.desktop.interface", "gtk-theme"])
            .output()
        {
            let theme = String::from_utf8_lossy(&output.stdout);
            if theme.contains("HighContrast") || theme.contains("high-contrast") {
                return true;
            }
        }

        // Also check the high-contrast key directly.
        if let Ok(output) = std::process::Command::new("gsettings")
            .args(["get", "org.gnome.desktop.a11y.interface", "high-contrast"])
            .output()
        {
            let val = String::from_utf8_lossy(&output.stdout);
            if val.trim() == "true" {
                return true;
            }
        }

        false
    }

    fn read_gsettings_a11y_enabled() -> bool {
        if let Ok(output) = std::process::Command::new("gsettings")
            .args(["get", "org.gnome.desktop.interface", "toolkit-accessibility"])
            .output()
        {
            let val = String::from_utf8_lossy(&output.stdout);
            return val.trim() == "true";
        }
        false
    }
}

impl Default for AccessibilityConfig {
    fn default() -> Self {
        Self {
            high_contrast: false,
            enabled: false,
        }
    }
}

/// Security settings relevant to the platform layer.
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Maximum paste size in bytes (0 = unlimited).
    pub max_paste_bytes: usize,
    /// Enable seccomp-BPF + Landlock sandbox on Linux.
    pub sandbox: bool,
    /// Enable seccomp enforcement mode (deny instead of audit).
    pub sandbox_enforce: bool,
    /// Enable strict Landlock filesystem enforcement.
    pub landlock_enforce: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            max_paste_bytes: 0,
            sandbox: false,
            landlock_enforce: true,
            sandbox_enforce: false,
        }
    }
}
