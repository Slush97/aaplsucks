//! `dwnldr-gui` — Native GPU-rendered desktop frontend for dwnldr.

mod app;
mod dispatch;
mod layout;
mod render;
mod state;
mod tools;

use std::path::PathBuf;

use tracing_subscriber::EnvFilter;

#[derive(serde::Serialize, serde::Deserialize)]
struct WindowState {
    width: u32,
    height: u32,
    x: i32,
    y: i32,
}

pub(crate) fn state_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| ".".into())
        .join("dwnldr")
        .join("window.toml")
}

fn load_window_state() -> Option<WindowState> {
    toml::from_str(&std::fs::read_to_string(state_path()).ok()?).ok()
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("dwnldr_gui=info".parse().unwrap()),
        )
        .init();

    let saved = load_window_state();

    let icon_img = image::load_from_memory(include_bytes!("../assets/icon-32.png"))
        .expect("embedded icon")
        .to_rgba8();
    let (iw, ih) = icon_img.dimensions();

    let width = saved.as_ref().map(|s| s.width).or(Some(1024));
    let height = saved.as_ref().map(|s| s.height).or(Some(700));

    let config = esox_platform::config::PlatformConfig {
        window: esox_platform::config::WindowConfig {
            title: "dwnldr".into(),
            decorations: true,
            width,
            height,
            position: saved
                .as_ref()
                .filter(|s| s.x != 0 || s.y != 0)
                .map(|s| (s.x, s.y)),
            icon_rgba: Some(esox_platform::config::IconData {
                rgba: icon_img.into_raw(),
                width: iw,
                height: ih,
            }),
        },
        background: "#0f0f0f".into(),
        ..Default::default()
    };

    let initial_w = width.unwrap_or(1024) as f32;
    let initial_h = height.unwrap_or(700) as f32;
    let app = app::App::new(initial_w, initial_h);

    if let Err(e) = esox_platform::run(config, Box::new(app)) {
        tracing::error!("fatal: {e}");
        std::process::exit(1);
    }
}
