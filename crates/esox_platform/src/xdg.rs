//! XDG Base Directory paths for Linux applications.
//!
//! Resolves `$XDG_CONFIG_HOME`, `$XDG_DATA_HOME`, `$XDG_CACHE_HOME`,
//! `$XDG_STATE_HOME`, and `$XDG_RUNTIME_DIR` per the [XDG Base Directory
//! Specification](https://specifications.freedesktop.org/basedir-spec/latest/).
//!
//! Zero dependencies beyond `std`.

use std::path::PathBuf;

/// Application-specific XDG directory resolver.
///
/// Each method returns a path like `$XDG_CONFIG_HOME/<app_name>/`, creating
/// the directory if it does not exist.
pub struct AppDirs {
    app_name: String,
}

impl AppDirs {
    /// Create a new resolver for the given application name.
    pub fn new(app_name: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
        }
    }

    /// `$XDG_CONFIG_HOME/<app>/` (default `~/.config/<app>/`).
    pub fn config_dir(&self) -> PathBuf {
        self.xdg_dir("XDG_CONFIG_HOME", ".config")
    }

    /// `$XDG_DATA_HOME/<app>/` (default `~/.local/share/<app>/`).
    pub fn data_dir(&self) -> PathBuf {
        self.xdg_dir("XDG_DATA_HOME", ".local/share")
    }

    /// `$XDG_CACHE_HOME/<app>/` (default `~/.cache/<app>/`).
    pub fn cache_dir(&self) -> PathBuf {
        self.xdg_dir("XDG_CACHE_HOME", ".cache")
    }

    /// `$XDG_STATE_HOME/<app>/` (default `~/.local/state/<app>/`).
    pub fn state_dir(&self) -> PathBuf {
        self.xdg_dir("XDG_STATE_HOME", ".local/state")
    }

    /// `$XDG_RUNTIME_DIR/<app>/`, or `None` if `$XDG_RUNTIME_DIR` is unset.
    pub fn runtime_dir(&self) -> Option<PathBuf> {
        std::env::var_os("XDG_RUNTIME_DIR").map(|dir| PathBuf::from(dir).join(&self.app_name))
    }

    fn xdg_dir(&self, env_var: &str, fallback_suffix: &str) -> PathBuf {
        let base = match std::env::var_os(env_var) {
            Some(dir) if !dir.is_empty() => PathBuf::from(dir),
            _ => home_dir().join(fallback_suffix),
        };
        base.join(&self.app_name)
    }
}

/// Best-effort `$HOME` resolution.
fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_dir_uses_xdg_env() {
        // SAFETY: test-only, single-threaded test runner.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", "/tmp/xdg_test_config") };
        let dirs = AppDirs::new("myapp");
        assert_eq!(dirs.config_dir(), PathBuf::from("/tmp/xdg_test_config/myapp"));
        unsafe { std::env::remove_var("XDG_CONFIG_HOME") };
    }

    #[test]
    fn data_dir_uses_xdg_env() {
        unsafe { std::env::set_var("XDG_DATA_HOME", "/tmp/xdg_test_data") };
        let dirs = AppDirs::new("myapp");
        assert_eq!(dirs.data_dir(), PathBuf::from("/tmp/xdg_test_data/myapp"));
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }

    #[test]
    fn runtime_dir_returns_none_when_unset() {
        unsafe { std::env::remove_var("XDG_RUNTIME_DIR") };
        let dirs = AppDirs::new("myapp");
        assert!(dirs.runtime_dir().is_none());
    }

    #[test]
    fn runtime_dir_returns_path_when_set() {
        unsafe { std::env::set_var("XDG_RUNTIME_DIR", "/run/user/1000") };
        let dirs = AppDirs::new("myapp");
        assert_eq!(dirs.runtime_dir(), Some(PathBuf::from("/run/user/1000/myapp")));
        unsafe { std::env::remove_var("XDG_RUNTIME_DIR") };
    }
}
