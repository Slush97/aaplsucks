//! `esox_platform` — Windowing, input dispatch, and platform integration.
//!
//! ## Event loop
//!
//! Call [`run()`] with a [`PlatformConfig`](config::PlatformConfig) and a boxed
//! [`AppDelegate`]. The platform creates a winit window, initialises wgpu via
//! [`esox_gfx::GpuContext`], and enters the event loop. Input events are
//! dispatched to `AppDelegate` methods; rendering happens in
//! [`AppDelegate::on_redraw`].
//!
//! ## AppDelegate pattern
//!
//! Implement [`AppDelegate`] to receive lifecycle callbacks:
//!
//! - **`on_init`** — called once after GPU is ready; load textures here
//! - **`on_redraw`** — called each frame; build your [`esox_gfx::Frame`] here
//! - **`on_key`** / **`on_mouse`** — input dispatch
//! - **`needs_continuous_redraw`** — return `true` while animating; `false` to
//!   idle and save power (the platform sleeps via `WaitUntil`)
//!
//! ## Optional features
//!
//! - `a11y` — AT-SPI2 accessibility bridge (Linux, requires `zbus` + `tokio`)
//! - `sandbox` — seccomp/Landlock sandboxing (Linux)

pub mod config;
pub mod perf;
pub mod sandbox;
pub mod xdg;

// Re-export esox_input so downstream crates can access input types
// through esox_platform without a direct dependency.
pub use esox_input;

#[cfg(feature = "settings")]
pub mod settings;

#[cfg(feature = "portals")]
pub mod portal;

#[cfg(feature = "a11y")]
pub mod atspi;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::window::{Window, WindowAttributes, WindowId};

/// Errors produced by the platform subsystem.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Window creation failed.
    #[error("failed to create window: {0}")]
    WindowCreation(String),

    /// GPU initialization failed.
    #[error("gpu error: {0}")]
    Gpu(#[from] esox_gfx::Error),

    /// Event loop error.
    #[error("event loop error: {0}")]
    EventLoop(String),

    /// Clipboard error.
    #[error("clipboard error: {0}")]
    Clipboard(String),
}

/// High-level application events dispatched to the terminal core.
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// The window was resized.
    Resized { width: u32, height: u32 },
    /// A keyboard input was received.
    KeyInput { key: String, modifiers: u8 },
    /// A mouse input was received.
    MouseInput { x: f64, y: f64, button: u8 },
    /// The window close was requested.
    CloseRequested,
    /// The window gained or lost focus.
    FocusChanged(bool),
    /// A redraw was requested.
    RedrawRequested,
}

/// User-defined events sent from background threads to wake the event loop.
///
/// Used by the PTY watcher thread and cursor blink timer to trigger redraws
/// without polling.
#[derive(Debug, Clone)]
pub enum AppUserEvent {
    /// A PTY file descriptor has data ready for reading.
    PtyReady,
    /// A timer tick (e.g. cursor blink) requests a redraw.
    TimerTick,
    /// A watched shader file was modified on disk.
    ShaderFileChanged,
    /// A render pipeline finished compiling on the background thread.
    PipelineReady,
    /// The XDG portal bridge is connected and ready.
    #[cfg(feature = "portals")]
    PortalReady,
}

/// Handle for background threads to wake the event loop.
///
/// Wraps the backend-specific event loop proxy so that `AppDelegate`
/// implementors can request redraws without knowing about winit.
#[derive(Clone)]
pub struct Waker {
    proxy: EventLoopProxy<AppUserEvent>,
}

impl Waker {
    /// Wake the event loop, causing a redraw.
    pub fn wake(&self) {
        let _ = self.proxy.send_event(AppUserEvent::TimerTick);
    }
}

impl std::fmt::Debug for Waker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Waker").finish()
    }
}

// ── Winit → esox_input conversions (crate-internal) ──

/// Convert a winit `ModifiersState` to `esox_input::Modifiers`.
pub(crate) fn convert_modifiers(m: winit::keyboard::ModifiersState) -> esox_input::Modifiers {
    esox_input::Modifiers::from_flags(
        m.shift_key(),
        m.control_key(),
        m.alt_key(),
        m.super_key(),
    )
}

/// Convert a winit `KeyEvent` to `esox_input::KeyEvent`.
pub(crate) fn convert_key_event(e: &winit::event::KeyEvent) -> esox_input::KeyEvent {
    esox_input::KeyEvent {
        key: convert_key(&e.logical_key),
        physical_key: convert_key_code(e.physical_key),
        pressed: e.state.is_pressed(),
        repeat: e.repeat,
        text: e.text.as_ref().map(|s| esox_input::SmolStr::new(s.as_str())),
    }
}

/// Convert a winit logical `Key` to `esox_input::Key`.
fn convert_key(key: &winit::keyboard::Key) -> esox_input::Key {
    use winit::keyboard::Key as WKey;
    match key {
        WKey::Named(n) => match convert_named_key(*n) {
            Some(nk) => esox_input::Key::Named(nk),
            None => esox_input::Key::Unidentified,
        },
        WKey::Character(s) => esox_input::Key::Character(esox_input::SmolStr::new(s.as_str())),
        _ => esox_input::Key::Unidentified,
    }
}

/// Convert a winit `NamedKey` to `esox_input::NamedKey`.
fn convert_named_key(key: winit::keyboard::NamedKey) -> Option<esox_input::NamedKey> {
    use winit::keyboard::NamedKey as W;
    use esox_input::NamedKey as N;
    Some(match key {
        W::Enter => N::Enter,
        W::Tab => N::Tab,
        W::Space => N::Space,
        W::Backspace => N::Backspace,
        W::Delete => N::Delete,
        W::Escape => N::Escape,
        W::ArrowUp => N::ArrowUp,
        W::ArrowDown => N::ArrowDown,
        W::ArrowLeft => N::ArrowLeft,
        W::ArrowRight => N::ArrowRight,
        W::Home => N::Home,
        W::End => N::End,
        W::PageUp => N::PageUp,
        W::PageDown => N::PageDown,
        W::F1 => N::F1,
        W::F2 => N::F2,
        W::F3 => N::F3,
        W::F4 => N::F4,
        W::F5 => N::F5,
        W::F6 => N::F6,
        W::F7 => N::F7,
        W::F8 => N::F8,
        W::F9 => N::F9,
        W::F10 => N::F10,
        W::F11 => N::F11,
        W::F12 => N::F12,
        _ => return None,
    })
}

/// Convert a winit `PhysicalKey` to `esox_input::KeyCode`.
fn convert_key_code(key: winit::keyboard::PhysicalKey) -> esox_input::KeyCode {
    use winit::keyboard::{KeyCode as WK, PhysicalKey};
    use esox_input::KeyCode as K;
    match key {
        PhysicalKey::Code(c) => match c {
            WK::KeyA => K::KeyA,
            WK::KeyB => K::KeyB,
            WK::KeyC => K::KeyC,
            WK::KeyD => K::KeyD,
            WK::KeyE => K::KeyE,
            WK::KeyF => K::KeyF,
            WK::KeyG => K::KeyG,
            WK::KeyH => K::KeyH,
            WK::KeyI => K::KeyI,
            WK::KeyJ => K::KeyJ,
            WK::KeyK => K::KeyK,
            WK::KeyL => K::KeyL,
            WK::KeyM => K::KeyM,
            WK::KeyN => K::KeyN,
            WK::KeyO => K::KeyO,
            WK::KeyP => K::KeyP,
            WK::KeyQ => K::KeyQ,
            WK::KeyR => K::KeyR,
            WK::KeyS => K::KeyS,
            WK::KeyT => K::KeyT,
            WK::KeyU => K::KeyU,
            WK::KeyV => K::KeyV,
            WK::KeyW => K::KeyW,
            WK::KeyX => K::KeyX,
            WK::KeyY => K::KeyY,
            WK::KeyZ => K::KeyZ,
            WK::Digit0 => K::Digit0,
            WK::Digit1 => K::Digit1,
            WK::Digit2 => K::Digit2,
            WK::Digit3 => K::Digit3,
            WK::Digit4 => K::Digit4,
            WK::Digit5 => K::Digit5,
            WK::Digit6 => K::Digit6,
            WK::Digit7 => K::Digit7,
            WK::Digit8 => K::Digit8,
            WK::Digit9 => K::Digit9,
            WK::F1 => K::F1,
            WK::F2 => K::F2,
            WK::F3 => K::F3,
            WK::F4 => K::F4,
            WK::F5 => K::F5,
            WK::F6 => K::F6,
            WK::F7 => K::F7,
            WK::F8 => K::F8,
            WK::F9 => K::F9,
            WK::F10 => K::F10,
            WK::F11 => K::F11,
            WK::F12 => K::F12,
            WK::ArrowUp => K::ArrowUp,
            WK::ArrowDown => K::ArrowDown,
            WK::ArrowLeft => K::ArrowLeft,
            WK::ArrowRight => K::ArrowRight,
            WK::Home => K::Home,
            WK::End => K::End,
            WK::PageUp => K::PageUp,
            WK::PageDown => K::PageDown,
            WK::Space => K::Space,
            WK::Enter => K::Enter,
            WK::Tab => K::Tab,
            WK::Backspace => K::Backspace,
            WK::Delete => K::Delete,
            WK::Escape => K::Escape,
            WK::ShiftLeft => K::ShiftLeft,
            WK::ShiftRight => K::ShiftRight,
            WK::ControlLeft => K::ControlLeft,
            WK::ControlRight => K::ControlRight,
            WK::AltLeft => K::AltLeft,
            WK::AltRight => K::AltRight,
            WK::SuperLeft => K::SuperLeft,
            WK::SuperRight => K::SuperRight,
            WK::Minus => K::Minus,
            WK::Equal => K::Equal,
            WK::BracketLeft => K::BracketLeft,
            WK::BracketRight => K::BracketRight,
            WK::Backslash => K::Backslash,
            WK::Semicolon => K::Semicolon,
            WK::Quote => K::Quote,
            WK::Backquote => K::Backquote,
            WK::Comma => K::Comma,
            WK::Period => K::Period,
            WK::Slash => K::Slash,
            _ => K::Escape, // fallback for unmapped codes
        },
        PhysicalKey::Unidentified(_) => K::Escape,
    }
}

/// Convert an `esox_input::CursorIcon` to `winit::window::CursorIcon`.
pub(crate) fn convert_cursor_icon(icon: esox_input::CursorIcon) -> winit::window::CursorIcon {
    match icon {
        esox_input::CursorIcon::Default => winit::window::CursorIcon::Default,
        esox_input::CursorIcon::Text => winit::window::CursorIcon::Text,
        esox_input::CursorIcon::Pointer => winit::window::CursorIcon::Pointer,
        esox_input::CursorIcon::Grab => winit::window::CursorIcon::Grab,
        esox_input::CursorIcon::Grabbing => winit::window::CursorIcon::Grabbing,
        esox_input::CursorIcon::ColResize => winit::window::CursorIcon::ColResize,
        esox_input::CursorIcon::RowResize => winit::window::CursorIcon::RowResize,
        esox_input::CursorIcon::NotAllowed => winit::window::CursorIcon::NotAllowed,
    }
}

/// Trait for injecting application behavior into the platform event loop.
///
/// The binary crate implements this to wire terminal logic without
/// `esox_platform` knowing about `esox_font`, `esox_grid`, or `esox_term`.
pub trait AppDelegate {
    /// Called once after GPU and pipeline initialization, before [`on_init`].
    ///
    /// Use this to register custom shader pipelines via
    /// [`PipelineRegistry::register_shader_pipeline`].
    fn register_pipelines(
        &mut self,
        _gpu: &esox_gfx::GpuContext,
        _registry: &mut esox_gfx::PipelineRegistry,
    ) {
    }

    /// Called once after GPU initialization is complete.
    fn on_init(&mut self, gpu: &esox_gfx::GpuContext, resources: &mut esox_gfx::RenderResources);

    /// Called each frame to render content.
    ///
    /// `perf` provides live performance statistics (FPS, RSS, CPU%) that can
    /// be rendered as an overlay.
    fn on_redraw(
        &mut self,
        gpu: &esox_gfx::GpuContext,
        resources: &mut esox_gfx::RenderResources,
        frame: &mut esox_gfx::Frame,
        perf: &crate::perf::PerfMonitor,
    );

    /// Called when a keyboard event is received.
    fn on_key(
        &mut self,
        event: &esox_input::KeyEvent,
        modifiers: esox_input::Modifiers,
    );

    /// Called when the window is resized.
    fn on_resize(&mut self, width: u32, height: u32, gpu: &esox_gfx::GpuContext);

    /// Called when a mouse event occurs.
    fn on_mouse(&mut self, event: MouseInputEvent);

    /// Called when the DPI scale factor changes.
    fn on_scale_changed(&mut self, scale_factor: f64, gpu: &esox_gfx::GpuContext);

    /// Return a new window title if one has been set, consuming the pending value.
    fn take_title(&mut self) -> Option<String> {
        None
    }

    /// Return a pending window title set by settings, consuming the value.
    fn take_settings_title(&mut self) -> Option<String> {
        None
    }

    /// Return a pending window decorations toggle, consuming the value.
    fn take_decorations(&mut self) -> Option<bool> {
        None
    }

    /// Return a pending clear color change, consuming the value.
    fn take_clear_color(&mut self) -> Option<[f32; 4]> {
        None
    }

    /// Paste text from clipboard into the terminal.
    fn on_paste(&mut self, text: &str);

    /// Called when the IME commits text (NOT a paste — no bracketed paste wrapping).
    fn on_ime_commit(&mut self, text: &str);

    /// Called when the IME preedit (composition) text changes.
    fn on_ime_preedit(&mut self, _text: String, _cursor: Option<(usize, usize)>) {}

    /// Called when the IME is enabled or disabled.
    fn on_ime_enabled(&mut self, _enabled: bool) {}

    /// Copy selected text to clipboard.
    fn on_copy(&mut self) -> Option<String>;

    /// Called when the window gains or loses focus.
    fn on_focus_changed(&mut self, _focused: bool) {}

    /// Called when a file is dropped onto the window.
    fn on_file_dropped(&mut self, _path: std::path::PathBuf) {}

    /// Called when a file is hovered over the window (Some) or leaves (None).
    fn on_file_hover(&mut self, _path: Option<std::path::PathBuf>, _x: f64, _y: f64) {}

    /// Whether a bell is pending (to trigger urgency hint).
    fn has_bell(&mut self) -> bool {
        false
    }

    /// Called after the bell has been handled.
    fn on_bell(&mut self) {}

    /// Check if a shader reload is pending and return the new source.
    ///
    /// Called by the platform layer during redraw after a `ShaderFileChanged`
    /// event. Returns `Some(source)` to trigger pipeline recreation, or
    /// `None` to skip.
    fn pending_shader_reload(&mut self) -> Option<String> {
        None
    }

    /// Called once before the event loop starts, providing a waker for
    /// background threads to wake the event loop.
    fn set_waker(&mut self, _waker: Waker) {}

    /// Called when the portal bridge is ready. Receive the handle to issue portal requests.
    #[cfg(feature = "portals")]
    fn set_portal_handle(&mut self, _handle: crate::portal::PortalHandle) {}

    /// Return a pending MSAA sample count change, consuming the value.
    ///
    /// When the sample count changes, the platform layer recreates all scene
    /// pipelines and the MSAA render target.
    fn take_msaa_change(&mut self) -> Option<u32> {
        None
    }

    /// Return a pending HDR mode change, consuming the value.
    ///
    /// When the HDR mode changes, the platform layer reconfigures the surface
    /// format and recreates all pipelines and resources.
    fn take_hdr_change(&mut self) -> Option<bool> {
        None
    }

    /// Whether the application should exit (e.g., child shell has exited).
    fn should_exit(&self) -> bool {
        false
    }

    /// Called just before the window is destroyed (close or exit).
    fn on_close(&mut self) {}

    /// Whether the current frame has damage that requires GPU submission.
    ///
    /// When `false`, the platform skips rendering and GPU submission for this
    /// frame, saving power.  Defaults to `true` (always redraw).
    fn needs_redraw(&self) -> bool {
        true
    }

    /// Called before the 2D render pass, with the pre-acquired surface view.
    ///
    /// Override this to render 3D content (via `Renderer3D::encode()`) to the
    /// surface before the 2D UI is composited on top. Return any command
    /// buffers that should be submitted before the 2D pass.
    ///
    /// Default is no-op (no pre-render pass).
    fn on_pre_render(
        &mut self,
        _gpu: &esox_gfx::GpuContext,
        _surface_view: &wgpu::TextureView,
    ) -> Vec<wgpu::CommandBuffer> {
        vec![]
    }

    /// Whether continuous redraw is needed (animations, active PTY output, etc.).
    ///
    /// When `false`, the platform only redraws in response to input events.
    /// Defaults to `false` for power savings.
    fn needs_continuous_redraw(&self) -> bool {
        false
    }

    /// Whether post-processing is enabled.
    ///
    /// When `true`, the scene is rendered to an offscreen texture first, then
    /// a fullscreen post-process pass presents the result.
    fn post_process_enabled(&self) -> bool {
        false
    }

    /// Return post-process effect parameters.
    ///
    /// Defaults to all zeros (no effects). Override to supply config values.
    fn post_process_params(&self) -> esox_gfx::PostProcessParams {
        esox_gfx::PostProcessParams::default()
    }

    /// Return user-supplied post-process WGSL fragment shader body, if any.
    ///
    /// The returned string is the body of `@fragment fn fs_main(in: VertexOutput)`.
    /// The platform wraps it with the standard preamble (bindings, uniforms, etc.).
    fn post_process_shader_source(&self) -> Option<String> {
        None
    }

    /// Return the desired mouse cursor icon for the current pointer position.
    ///
    /// Called on every mouse move so the platform can update the OS cursor.
    /// Defaults to `Text` (IBeam) for the terminal grid.
    fn cursor_icon(&self, _x: f64, _y: f64) -> esox_input::CursorIcon {
        esox_input::CursorIcon::Text
    }

    /// Whether the cursor should be grabbed (confined to the window) and hidden.
    ///
    /// When `true`, the platform hides the cursor and locks it to the window
    /// center, providing relative mouse motion for camera control. Defaults to `false`.
    fn cursor_grabbed(&self) -> bool {
        false
    }
}

/// Mouse input event dispatched from platform to the delegate.
#[derive(Debug, Clone, Copy)]
pub enum MouseInputEvent {
    /// Mouse moved to pixel coordinates.
    Moved { x: f64, y: f64 },
    /// Mouse button pressed.
    Press {
        /// Pixel X coordinate.
        x: f64,
        /// Pixel Y coordinate.
        y: f64,
        /// Button (0=left, 1=middle, 2=right).
        button: u8,
    },
    /// Mouse button released.
    Release {
        /// Pixel X coordinate.
        x: f64,
        /// Pixel Y coordinate.
        y: f64,
        /// Button (0=left, 1=middle, 2=right).
        button: u8,
    },
    /// Mouse wheel scroll.
    Scroll {
        /// Pixel X coordinate.
        x: f64,
        /// Pixel Y coordinate.
        y: f64,
        /// Scroll delta (positive = up/left).
        delta_y: f32,
    },
    /// Raw mouse motion delta (from `DeviceEvent`; used when cursor is grabbed).
    RawMotion { dx: f64, dy: f64 },
    /// Cursor left the window surface.
    Left,
}

/// Detect whether a key event represents Ctrl+Shift+C (copy shortcut).
///
/// Uses the physical key to be layout-independent. Accepts two modifier
/// sources to work around Wayland timing issues where `ModifiersChanged`
/// may arrive late.
pub(crate) fn is_copy_shortcut(
    physical_key: winit::keyboard::PhysicalKey,
    logical_key: &winit::keyboard::Key,
    modifiers: winit::keyboard::ModifiersState,
    text_with_all_modifiers: Option<&str>,
) -> bool {
    use winit::keyboard::{Key as WKey, KeyCode, PhysicalKey};

    let ctrl_shift_from_mods = modifiers.control_key() && modifiers.shift_key();
    let ctrl_shift_from_event = {
        let is_c = matches!(physical_key, PhysicalKey::Code(KeyCode::KeyC));
        let shift_in_logical = matches!(
            logical_key,
            WKey::Character(s) if s.as_str().chars().next().is_some_and(|c| c.is_ascii_uppercase())
        );
        let ctrl_in_text = text_with_all_modifiers
            .is_some_and(|t| t.as_bytes().first().is_some_and(|&b| b < 0x20));
        is_c && shift_in_logical && ctrl_in_text
    };

    (ctrl_shift_from_mods || ctrl_shift_from_event)
        && matches!(physical_key, PhysicalKey::Code(KeyCode::KeyC))
}

/// Detect whether a key event represents Ctrl+Shift+V (paste shortcut).
///
/// Mirror of [`is_copy_shortcut`] for the V key.
pub(crate) fn is_paste_shortcut(
    physical_key: winit::keyboard::PhysicalKey,
    logical_key: &winit::keyboard::Key,
    modifiers: winit::keyboard::ModifiersState,
    text_with_all_modifiers: Option<&str>,
) -> bool {
    use winit::keyboard::{Key as WKey, KeyCode, PhysicalKey};

    let ctrl_shift_from_mods = modifiers.control_key() && modifiers.shift_key();
    let ctrl_shift_from_event = {
        let is_v = matches!(physical_key, PhysicalKey::Code(KeyCode::KeyV));
        let shift_in_logical = matches!(
            logical_key,
            WKey::Character(s) if s.as_str().chars().next().is_some_and(|c| c.is_ascii_uppercase())
        );
        let ctrl_in_text = text_with_all_modifiers
            .is_some_and(|t| t.as_bytes().first().is_some_and(|&b| b < 0x20));
        is_v && shift_in_logical && ctrl_in_text
    };

    (ctrl_shift_from_mods || ctrl_shift_from_event)
        && matches!(physical_key, PhysicalKey::Code(KeyCode::KeyV))
}

/// Map a winit mouse button to a numeric code (0=left, 1=middle, 2=right, 3=other).
pub fn classify_mouse_button(button: winit::event::MouseButton) -> u8 {
    match button {
        winit::event::MouseButton::Left => 0,
        winit::event::MouseButton::Middle => 1,
        winit::event::MouseButton::Right => 2,
        _ => 3,
    }
}

/// The main application struct that drives the event loop.
pub struct App {
    config: crate::config::PlatformConfig,
    delegate: Box<dyn AppDelegate>,
    window: Option<Arc<Window>>,
    gpu: Option<esox_gfx::GpuContext>,
    pipeline_registry: Option<esox_gfx::PipelineRegistry>,
    render_resources: Option<esox_gfx::RenderResources>,
    frame: esox_gfx::Frame,
    frame_number: u32,
    start_time: std::time::Instant,
    last_frame_elapsed: f32,
    clear_color: esox_gfx::Color,
    current_modifiers: winit::keyboard::ModifiersState,
    /// Last known cursor position in physical pixels.
    cursor_position: (f64, f64),
    /// Offscreen render target for post-processing (created lazily).
    offscreen: Option<esox_gfx::OffscreenTarget>,
    /// Post-process bind group layout (created once with offscreen).
    pp_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Linear sampler for post-process texture sampling.
    pp_sampler: Option<wgpu::Sampler>,
    /// Bloom post-processing pass (created when bloom > 0).
    bloom_pass: Option<esox_gfx::BloomPass>,
    /// 1×1 black placeholder texture view for when bloom is disabled.
    black_bloom_view: Option<wgpu::TextureView>,
    /// Whether a shader file change event is pending (set by user_event, consumed in redraw).
    shader_reload_pending: bool,
    /// Multisampled render target view (Some when MSAA is active, None for sample_count=1).
    msaa_view: Option<wgpu::TextureView>,
    /// Depth/stencil render target view for early-z rejection and future stencil masking.
    depth_view: Option<wgpu::TextureView>,
    /// Monitor refresh rate in Hz (queried on window creation, default 60).
    monitor_refresh_hz: u32,
    /// Timestamp of the last redraw (for frame rate throttling).
    last_redraw: std::time::Instant,
    /// Whether a redraw has been requested but not yet serviced.
    redraw_pending: bool,
    /// Whether the cursor is currently grabbed (locked + hidden).
    cursor_grabbed: bool,
    /// Count of consecutive render failures (for device-lost recovery).
    consecutive_render_failures: u32,
    /// Whether a screenshot should be captured on the next frame.
    screenshot_pending: bool,
    /// Receiver for pipelines compiled on a background thread.
    pipeline_rx: Option<esox_gfx::PipelineReceiver>,
    /// Event loop proxy for waking the main thread from background compilation.
    event_proxy: Option<EventLoopProxy<AppUserEvent>>,
    /// Live performance monitor (frame times, RSS, CPU%).
    perf: crate::perf::PerfMonitor,
}

impl App {
    /// Create a new application with the given config and delegate.
    pub fn new(config: crate::config::PlatformConfig, delegate: Box<dyn AppDelegate>) -> Self {
        Self {
            config,
            delegate,
            window: None,
            gpu: None,
            pipeline_registry: None,
            render_resources: None,
            frame: esox_gfx::Frame::new(),
            frame_number: 0,
            start_time: std::time::Instant::now(),
            last_frame_elapsed: 0.0,
            clear_color: esox_gfx::Color::BLACK,
            current_modifiers: winit::keyboard::ModifiersState::empty(),
            cursor_position: (0.0, 0.0),
            offscreen: None,
            pp_bind_group_layout: None,
            pp_sampler: None,
            bloom_pass: None,
            black_bloom_view: None,
            shader_reload_pending: false,
            msaa_view: None,
            depth_view: None,
            monitor_refresh_hz: 60,
            last_redraw: std::time::Instant::now(),
            redraw_pending: false,
            cursor_grabbed: false,
            consecutive_render_failures: 0,
            screenshot_pending: false,
            pipeline_rx: None,
            event_proxy: None,
            perf: crate::perf::PerfMonitor::new(300),
        }
    }

    /// Write perf report to `perf_report.txt` in the current directory.
    fn write_perf_report(&self) {
        let path = std::path::PathBuf::from("perf_report.txt");
        if let Err(e) = self.perf.write_report(&path) {
            tracing::error!("failed to write perf report: {e}");
        }
    }
}

/// Create a multisampled texture and return its view.
fn create_msaa_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Create a depth/stencil texture and return its view.
fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    sample_count: u32,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24PlusStencil8,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

impl ApplicationHandler<AppUserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut attrs = WindowAttributes::default()
            .with_title(&self.config.window.title)
            .with_decorations(self.config.window.decorations);
        if let (Some(w), Some(h)) = (self.config.window.width, self.config.window.height) {
            attrs = attrs.with_inner_size(winit::dpi::LogicalSize::new(w, h));
        }
        if let Some((x, y)) = self.config.window.position {
            attrs = attrs.with_position(winit::dpi::LogicalPosition::new(x, y));
        }
        if let Some(ref icon) = self.config.window.icon_rgba {
            if let Ok(i) = winit::window::Icon::from_rgba(icon.rgba.clone(), icon.width, icon.height) {
                attrs = attrs.with_window_icon(Some(i));
            }
        }
        // Tell the compositor this window uses transparency so it honors alpha.
        if self.config.opacity < 1.0 {
            attrs = attrs.with_transparent(true);
        }
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(esox_gfx::GpuContext::new(window.clone(), self.config.hdr)) {
            Ok(mut gpu) => {
                // Prefer PreMultiplied alpha so the compositor honors background opacity.
                if self.config.opacity < 1.0 {
                    let caps = gpu.surface.get_capabilities(&gpu.adapter);
                    if caps
                        .alpha_modes
                        .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
                    {
                        gpu.config.alpha_mode = wgpu::CompositeAlphaMode::PreMultiplied;
                        gpu.surface.configure(&gpu.device, &gpu.config);
                    }
                }

                // Set MSAA sample count before pipeline creation.
                gpu.sample_count = self.config.msaa;

                let mut registry = esox_gfx::PipelineRegistry::new();
                // Create bind group layout synchronously (cheap descriptor).
                let _scene_layout = registry.create_scene_bind_group_layout(&gpu);

                match esox_gfx::RenderResources::new(&gpu, &registry) {
                    Ok(mut resources) => {
                        // Create post-process layout and sampler (cheap, sync).
                        let pp_layout = esox_gfx::post_process_bind_group_layout(&gpu.device);
                        let user_shader = self.delegate.post_process_shader_source();
                        if let Some(ref src) = user_shader
                            && let Err(e) = esox_gfx::validate_user_shader(src)
                        {
                            tracing::warn!("user post-process shader failed pre-validation: {e}");
                        }
                        let pp_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                            label: Some("post_process_sampler"),
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            ..Default::default()
                        });
                        // Create placeholder black texture for bloom binding.
                        let (_black_tex, black_view) = esox_gfx::bloom::create_black_texture(
                            &gpu.device,
                            &gpu.queue,
                            gpu.config.format,
                        );
                        self.black_bloom_view = Some(black_view);
                        let bloom_view_ref = self.black_bloom_view.as_ref().unwrap();

                        // Create bloom pass if bloom is enabled.
                        let bloom_bind_group_layout = if self.delegate.post_process_enabled() {
                            let bloom_pass = esox_gfx::BloomPass::new(
                                &gpu.device,
                                gpu.config.width,
                                gpu.config.height,
                                gpu.config.format,
                                bloom_view_ref, // temporary; updated below
                            );
                            let bloom_layout = bloom_pass.bind_group_layout().clone();
                            self.bloom_pass = Some(bloom_pass);
                            Some(bloom_layout)
                        } else {
                            None
                        };

                        // Get the bloom result view or fallback to black.
                        let effective_bloom_view = self
                            .bloom_pass
                            .as_ref()
                            .map(|b| b.result_view())
                            .unwrap_or(bloom_view_ref);

                        if self.delegate.post_process_enabled() {
                            let offscreen = esox_gfx::OffscreenTarget::new(
                                &gpu.device,
                                gpu.config.width,
                                gpu.config.height,
                                gpu.config.format,
                                &pp_layout,
                                &resources.uniform_buffer,
                                &pp_sampler,
                                effective_bloom_view,
                            );
                            // Update bloom pass to use the offscreen scene texture.
                            if let Some(bloom) = self.bloom_pass.as_mut() {
                                bloom.update_scene_texture(&gpu.device, &offscreen.sample_view);
                            }
                            self.offscreen = Some(offscreen);
                        }

                        // Spawn async pipeline compilation on background thread.
                        let proxy = self.event_proxy.clone();
                        self.pipeline_rx = Some(esox_gfx::spawn_pipeline_compilation(
                            esox_gfx::PipelineCompileConfig {
                                device: Arc::clone(&gpu.device),
                                format: gpu.config.format,
                                sample_count: gpu.sample_count,
                                scene_bind_group_layout: registry
                                    .scene_bind_group_layout()
                                    .expect("layout just created")
                                    .clone(),
                                pp_bind_group_layout: Some(pp_layout.clone()),
                                user_shader_source: user_shader,
                                bloom_bind_group_layout,
                            },
                            move || {
                                if let Some(ref p) = proxy {
                                    let _ = p.send_event(AppUserEvent::PipelineReady);
                                }
                            },
                        ));

                        self.pp_bind_group_layout = Some(pp_layout);
                        self.pp_sampler = Some(pp_sampler);

                        // Create MSAA texture if sample_count > 1.
                        if gpu.sample_count > 1 {
                            self.msaa_view = Some(create_msaa_texture(
                                &gpu.device,
                                gpu.config.width,
                                gpu.config.height,
                                gpu.config.format,
                                gpu.sample_count,
                            ));
                        }

                        // Create depth/stencil texture (always, even at sample_count=1).
                        self.depth_view = Some(create_depth_texture(
                            &gpu.device,
                            gpu.config.width,
                            gpu.config.height,
                            gpu.sample_count,
                        ));

                        self.delegate.register_pipelines(&gpu, &mut registry);
                        self.delegate.on_init(&gpu, &mut resources);
                        self.render_resources = Some(resources);
                        self.pipeline_registry = Some(registry);
                    }
                    Err(e) => {
                        tracing::error!("failed to create render resources: {e}");
                        event_loop.exit();
                        return;
                    }
                }
                self.gpu = Some(gpu);
            }
            Err(e) => {
                tracing::error!("failed to initialize GPU: {e}");
                event_loop.exit();
                return;
            }
        }

        let mut clear = esox_gfx::Color::from_hex(&self.config.background)
            .unwrap_or(esox_gfx::Color::BLACK);
        clear.a = self.config.opacity;
        self.clear_color = clear.premultiplied();

        // Query the monitor refresh rate for frame throttling.
        // Filter video modes to the monitor's native resolution so we don't
        // pick up a higher Hz from a lower resolution (e.g. 1080p@240 on a
        // 4K@144 panel). Fall back to 240Hz when no modes are reported
        // (common on Wayland) so the GPU present mode provides real sync.
        if let Some(monitor) = window.current_monitor() {
            let native_size = monitor.size();
            let modes: Vec<_> = monitor.video_modes().collect();

            // Best: max Hz at the monitor's current resolution.
            let hz_at_native = modes
                .iter()
                .filter(|m| m.size() == native_size)
                .map(|m| (m.refresh_rate_millihertz() + 999) / 1000)
                .max()
                .filter(|&hz| hz > 0);

            if let Some(hz) = hz_at_native {
                self.monitor_refresh_hz = hz;
                tracing::info!(
                    "monitor: {:?}, {}x{} @ {}Hz ({} modes total)",
                    monitor.name(),
                    native_size.width,
                    native_size.height,
                    hz,
                    modes.len(),
                );
            } else if let Some(hz) = modes
                .iter()
                .map(|m| (m.refresh_rate_millihertz() + 999) / 1000)
                .max()
                .filter(|&hz| hz > 0)
            {
                // No modes at native size — use max across all modes as last resort.
                self.monitor_refresh_hz = hz;
                tracing::info!(
                    "monitor: {:?}, no modes at {}x{}, best across all: {}Hz",
                    monitor.name(),
                    native_size.width,
                    native_size.height,
                    hz,
                );
            } else {
                self.monitor_refresh_hz = 240;
                tracing::info!(
                    "monitor: {:?}, no video modes reported (Wayland?), defaulting to {}Hz cap",
                    monitor.name(),
                    self.monitor_refresh_hz,
                );
            }
        } else {
            self.monitor_refresh_hz = 240;
            tracing::warn!("no monitor detected, defaulting to {}Hz cap", self.monitor_refresh_hz);
        }

        window.set_ime_allowed(true);
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.write_perf_report();
                self.delegate.on_close();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size.width, size.height);
                    self.delegate.on_resize(size.width, size.height, gpu);
                    // Recreate MSAA texture at new size.
                    if gpu.sample_count > 1 {
                        self.msaa_view = Some(create_msaa_texture(
                            &gpu.device,
                            size.width,
                            size.height,
                            gpu.config.format,
                            gpu.sample_count,
                        ));
                    }
                    // Recreate depth/stencil texture at new size.
                    self.depth_view = Some(create_depth_texture(
                        &gpu.device,
                        size.width,
                        size.height,
                        gpu.sample_count,
                    ));
                    // Resize bloom pass if present.
                    if let Some(bloom) = self.bloom_pass.as_mut() {
                        let scene_view = self
                            .offscreen
                            .as_ref()
                            .map(|o| &o.sample_view)
                            .unwrap_or_else(|| self.black_bloom_view.as_ref().unwrap());
                        bloom.resize(&gpu.device, size.width, size.height, scene_view);
                    }
                    // Resize offscreen target if present.
                    if let (Some(offscreen), Some(resources), Some(layout), Some(sampler)) = (
                        self.offscreen.as_mut(),
                        self.render_resources.as_ref(),
                        self.pp_bind_group_layout.as_ref(),
                        self.pp_sampler.as_ref(),
                    ) {
                        let bloom_view = self
                            .bloom_pass
                            .as_ref()
                            .map(|b| b.result_view())
                            .unwrap_or_else(|| self.black_bloom_view.as_ref().unwrap());
                        offscreen.resize(
                            &gpu.device,
                            size.width,
                            size.height,
                            gpu.config.format,
                            layout,
                            &resources.uniform_buffer,
                            sampler,
                            bloom_view,
                        );
                        // Update bloom pass scene texture binding after offscreen resize.
                        if let Some(bloom) = self.bloom_pass.as_mut() {
                            bloom.update_scene_texture(&gpu.device, &offscreen.sample_view);
                        }
                    }
                }
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.current_modifiers = mods.state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Intercept Ctrl+Shift+C/V only on press (not release).
                if event.state == winit::event::ElementState::Pressed {
                    let text_all_mods = {
                        use winit::platform::modifier_supplement::KeyEventExtModifierSupplement;
                        event.text_with_all_modifiers().map(|s| s.to_string())
                    };
                    let mods = self.current_modifiers;

                    let copy = is_copy_shortcut(
                        event.physical_key,
                        &event.logical_key,
                        mods,
                        text_all_mods.as_deref(),
                    );
                    let paste = is_paste_shortcut(
                        event.physical_key,
                        &event.logical_key,
                        mods,
                        text_all_mods.as_deref(),
                    );

                    if copy || paste {
                        if copy {
                            tracing::debug!("Ctrl+Shift+C intercepted for copy");
                            if let Some(text) = self.delegate.on_copy() {
                                tracing::debug!(len = text.len(), "copied text to clipboard");
                                if let Err(e) = Clipboard::write(&text) {
                                    tracing::warn!("clipboard write failed: {e}");
                                }
                            } else {
                                tracing::debug!("copy: no selection");
                            }
                            if let Some(window) = self.window.as_ref() {
                                window.request_redraw();
                            }
                            return;
                        }
                        if paste {
                            tracing::debug!("Ctrl+Shift+V intercepted for paste");
                            match Clipboard::read(self.config.security.max_paste_bytes) {
                                Ok(text) if !text.is_empty() => {
                                    tracing::debug!(len = text.len(), "pasting from clipboard");
                                    self.delegate.on_paste(&text);
                                }
                                Ok(_) => {
                                    tracing::debug!("paste: clipboard empty");
                                }
                                Err(e) => tracing::warn!("clipboard read failed: {e}"),
                            }
                            if let Some(window) = self.window.as_ref() {
                                window.request_redraw();
                            }
                            return;
                        }
                    }
                }

                // F12 — screenshot
                if event.state == winit::event::ElementState::Pressed {
                    use winit::keyboard::{KeyCode, PhysicalKey};
                    if event.physical_key == PhysicalKey::Code(KeyCode::F12) {
                        self.screenshot_pending = true;
                        tracing::info!("screenshot requested (F12)");
                    }
                }

                // Forward both press AND release events so the input system
                // can track key-up state (axes, held, just_released).
                let converted = convert_key_event(&event);
                let mods = convert_modifiers(self.current_modifiers);
                self.delegate.on_key(&converted, mods);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.delegate.on_mouse(MouseInputEvent::Left);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = (position.x, position.y);
                self.delegate.on_mouse(MouseInputEvent::Moved {
                    x: position.x,
                    y: position.y,
                });
                // Update OS cursor icon based on pointer position (skip when grabbed).
                if !self.cursor_grabbed {
                    if let Some(window) = self.window.as_ref() {
                        let icon = self.delegate.cursor_icon(position.x, position.y);
                        window.set_cursor(winit::window::Cursor::Icon(convert_cursor_icon(icon)));
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let btn = classify_mouse_button(button);
                let (x, y) = self.cursor_position;
                let event = match state {
                    winit::event::ElementState::Pressed => {
                        MouseInputEvent::Press { x, y, button: btn }
                    }
                    winit::event::ElementState::Released => {
                        MouseInputEvent::Release { x, y, button: btn }
                    }
                };
                self.delegate.on_mouse(event);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta_y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                };
                let (x, y) = self.cursor_position;
                self.delegate
                    .on_mouse(MouseInputEvent::Scroll { x, y, delta_y });
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::Ime(ime) => {
                match ime {
                    winit::event::Ime::Commit(text) => {
                        // IME composition committed — forward raw text (not a paste).
                        self.delegate.on_ime_commit(&text);
                        if let Some(window) = self.window.as_ref() {
                            window.request_redraw();
                        }
                    }
                    winit::event::Ime::Preedit(text, cursor) => {
                        self.delegate.on_ime_preedit(text, cursor);
                        if let Some(window) = self.window.as_ref() {
                            window.request_redraw();
                        }
                    }
                    winit::event::Ime::Enabled => {
                        self.delegate.on_ime_enabled(true);
                    }
                    winit::event::Ime::Disabled => {
                        self.delegate.on_ime_enabled(false);
                    }
                }
            }
            WindowEvent::DroppedFile(path) => {
                self.delegate.on_file_dropped(path);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::HoveredFile(path) => {
                let (x, y) = self.cursor_position;
                self.delegate.on_file_hover(Some(path), x, y);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::HoveredFileCancelled => {
                let (x, y) = self.cursor_position;
                self.delegate.on_file_hover(None, x, y);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::Focused(focused) => {
                // Release cursor grab on focus loss so the user can interact
                // with other windows. It will be re-acquired on the next
                // redraw if the delegate still wants it.
                if !focused && self.cursor_grabbed {
                    self.cursor_grabbed = false;
                    if let Some(window) = self.window.as_ref() {
                        use winit::window::CursorGrabMode;
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                    }
                }
                self.delegate.on_focus_changed(focused);
                if !self.redraw_pending
                    && let Some(window) = self.window.as_ref()
                {
                    window.request_redraw();
                    self.redraw_pending = true;
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(gpu) = self.gpu.as_ref() {
                    self.delegate.on_scale_changed(scale_factor, gpu);
                }
            }
            WindowEvent::RedrawRequested => {
                self.last_redraw = std::time::Instant::now();

                // Sync cursor grab/hide state with delegate.
                let want_grab = self.delegate.cursor_grabbed();
                if want_grab != self.cursor_grabbed {
                    self.cursor_grabbed = want_grab;
                    if let Some(window) = self.window.as_ref() {
                        if want_grab {
                            // Try Locked first (raw motion), fall back to Confined.
                            use winit::window::CursorGrabMode;
                            if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                            }
                            window.set_cursor_visible(false);
                        } else {
                            use winit::window::CursorGrabMode;
                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        }
                    }
                }

                // Hot-reload post-process shader if a file change was detected.
                if self.shader_reload_pending {
                    if let Some(source) = self.delegate.pending_shader_reload() {
                        if let Err(e) = esox_gfx::validate_user_shader(&source) {
                            tracing::warn!("shader reload failed validation: {e}");
                        } else if let (Some(gpu), Some(registry)) =
                            (self.gpu.as_ref(), self.pipeline_registry.as_mut())
                        {
                            let pp_layout = self
                                .pp_bind_group_layout
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| {
                                    esox_gfx::post_process_bind_group_layout(&gpu.device)
                                });
                            if let Err(e) = registry.create_post_process_pipeline(
                                gpu,
                                &pp_layout,
                                Some(&source),
                            ) {
                                tracing::warn!("shader reload pipeline creation failed: {e}");
                            } else {
                                tracing::info!("post-process shader hot-reloaded");
                            }
                        }
                    }
                    self.shader_reload_pending = false;
                }

                // Handle MSAA sample count change (requires full pipeline rebuild).
                if let Some(new_msaa) = self.delegate.take_msaa_change()
                    && let Some(gpu) = self.gpu.as_mut()
                {
                    gpu.sample_count = new_msaa;
                    let mut registry = esox_gfx::PipelineRegistry::new();
                    let _scene_layout = registry.create_scene_bind_group_layout(gpu);

                    let pp_layout = self
                        .pp_bind_group_layout
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(|| esox_gfx::post_process_bind_group_layout(&gpu.device));
                    let user_shader = self.delegate.post_process_shader_source();

                    // Recreate render resources (only needs bind group layout).
                    match esox_gfx::RenderResources::new(gpu, &registry) {
                        Ok(resources) => {
                            self.render_resources = Some(resources);
                        }
                        Err(e) => {
                            tracing::error!("failed to recreate render resources for MSAA: {e}");
                        }
                    }

                    // Recreate bloom pass for MSAA change.
                    let bloom_bind_group_layout = if self.delegate.post_process_enabled() {
                        let black_view = self.black_bloom_view.as_ref().unwrap();
                        let bloom_pass = esox_gfx::BloomPass::new(
                            &gpu.device,
                            gpu.config.width,
                            gpu.config.height,
                            gpu.config.format,
                            black_view,
                        );
                        let layout = bloom_pass.bind_group_layout().clone();
                        self.bloom_pass = Some(bloom_pass);
                        Some(layout)
                    } else {
                        self.bloom_pass = None;
                        None
                    };

                    // Spawn async pipeline compilation.
                    let proxy = self.event_proxy.clone();
                    self.pipeline_rx = Some(esox_gfx::spawn_pipeline_compilation(
                        esox_gfx::PipelineCompileConfig {
                            device: Arc::clone(&gpu.device),
                            format: gpu.config.format,
                            sample_count: gpu.sample_count,
                            scene_bind_group_layout: registry
                                .scene_bind_group_layout()
                                .expect("layout just created")
                                .clone(),
                            pp_bind_group_layout: Some(pp_layout.clone()),
                            user_shader_source: user_shader,
                            bloom_bind_group_layout,
                        },
                        move || {
                            if let Some(ref p) = proxy {
                                let _ = p.send_event(AppUserEvent::PipelineReady);
                            }
                        },
                    ));

                    // Recreate MSAA texture.
                    if new_msaa > 1 {
                        self.msaa_view = Some(create_msaa_texture(
                            &gpu.device,
                            gpu.config.width,
                            gpu.config.height,
                            gpu.config.format,
                            new_msaa,
                        ));
                    } else {
                        self.msaa_view = None;
                    }
                    // Recreate depth/stencil texture with new sample count.
                    self.depth_view = Some(create_depth_texture(
                        &gpu.device,
                        gpu.config.width,
                        gpu.config.height,
                        new_msaa,
                    ));
                    // Recreate offscreen target if post-process is active.
                    if self.delegate.post_process_enabled()
                        && let (Some(sampler), Some(resources)) =
                            (self.pp_sampler.as_ref(), self.render_resources.as_ref())
                    {
                        let bloom_view = self
                            .bloom_pass
                            .as_ref()
                            .map(|b| b.result_view())
                            .unwrap_or_else(|| self.black_bloom_view.as_ref().unwrap());
                        let offscreen = esox_gfx::OffscreenTarget::new(
                            &gpu.device,
                            gpu.config.width,
                            gpu.config.height,
                            gpu.config.format,
                            &pp_layout,
                            &resources.uniform_buffer,
                            sampler,
                            bloom_view,
                        );
                        if let Some(bloom) = self.bloom_pass.as_mut() {
                            bloom.update_scene_texture(&gpu.device, &offscreen.sample_view);
                        }
                        self.offscreen = Some(offscreen);
                    }
                    self.pp_bind_group_layout = Some(pp_layout);
                    self.pipeline_registry = Some(registry);
                }

                // Handle HDR mode change (requires surface reconfiguration + full rebuild).
                if let Some(new_hdr) = self.delegate.take_hdr_change()
                    && let Some(gpu) = self.gpu.as_mut()
                {
                    let caps = gpu.surface.get_capabilities(&gpu.adapter);
                    let srgb_fallback = caps
                        .formats
                        .iter()
                        .find(|f| f.is_srgb())
                        .copied()
                        .or_else(|| caps.formats.first().copied())
                        .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

                    let (format, hdr_active) = if new_hdr {
                        if let Some(&f) = caps
                            .formats
                            .iter()
                            .find(|f| **f == wgpu::TextureFormat::Rgba16Float)
                        {
                            tracing::info!("HDR enabled: switching to Rgba16Float");
                            (f, true)
                        } else {
                            tracing::warn!(
                                "HDR requested but Rgba16Float not supported; staying sRGB"
                            );
                            (srgb_fallback, false)
                        }
                    } else {
                        tracing::info!("HDR disabled: switching to sRGB");
                        (srgb_fallback, false)
                    };

                    gpu.config.format = format;
                    gpu.hdr_active = hdr_active;
                    gpu.surface.configure(&gpu.device, &gpu.config);

                    // Full pipeline rebuild (same pattern as MSAA change).
                    let mut registry = esox_gfx::PipelineRegistry::new();
                    let _scene_layout = registry.create_scene_bind_group_layout(gpu);

                    let pp_layout = self
                        .pp_bind_group_layout
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(|| esox_gfx::post_process_bind_group_layout(&gpu.device));
                    let user_shader = self.delegate.post_process_shader_source();

                    match esox_gfx::RenderResources::new(gpu, &registry) {
                        Ok(resources) => {
                            self.render_resources = Some(resources);
                        }
                        Err(e) => {
                            tracing::error!("failed to recreate render resources for HDR: {e}");
                        }
                    }

                    // Recreate black bloom placeholder for new format.
                    let (_black_tex, black_view) = esox_gfx::bloom::create_black_texture(
                        &gpu.device,
                        &gpu.queue,
                        gpu.config.format,
                    );
                    self.black_bloom_view = Some(black_view);

                    // Recreate bloom pass for HDR change.
                    let bloom_bind_group_layout = if self.delegate.post_process_enabled() {
                        let black_view = self.black_bloom_view.as_ref().unwrap();
                        let bloom_pass = esox_gfx::BloomPass::new(
                            &gpu.device,
                            gpu.config.width,
                            gpu.config.height,
                            gpu.config.format,
                            black_view,
                        );
                        let layout = bloom_pass.bind_group_layout().clone();
                        self.bloom_pass = Some(bloom_pass);
                        Some(layout)
                    } else {
                        self.bloom_pass = None;
                        None
                    };

                    // Spawn async pipeline compilation.
                    let proxy = self.event_proxy.clone();
                    self.pipeline_rx = Some(esox_gfx::spawn_pipeline_compilation(
                        esox_gfx::PipelineCompileConfig {
                            device: Arc::clone(&gpu.device),
                            format: gpu.config.format,
                            sample_count: gpu.sample_count,
                            scene_bind_group_layout: registry
                                .scene_bind_group_layout()
                                .expect("layout just created")
                                .clone(),
                            pp_bind_group_layout: Some(pp_layout.clone()),
                            user_shader_source: user_shader,
                            bloom_bind_group_layout,
                        },
                        move || {
                            if let Some(ref p) = proxy {
                                let _ = p.send_event(AppUserEvent::PipelineReady);
                            }
                        },
                    ));

                    // Recreate MSAA texture at current format.
                    if gpu.sample_count > 1 {
                        self.msaa_view = Some(create_msaa_texture(
                            &gpu.device,
                            gpu.config.width,
                            gpu.config.height,
                            gpu.config.format,
                            gpu.sample_count,
                        ));
                    }
                    // Recreate depth/stencil texture (sample count may differ).
                    self.depth_view = Some(create_depth_texture(
                        &gpu.device,
                        gpu.config.width,
                        gpu.config.height,
                        gpu.sample_count,
                    ));
                    // Recreate offscreen target if post-process is active.
                    if self.delegate.post_process_enabled()
                        && let (Some(sampler), Some(resources)) =
                            (self.pp_sampler.as_ref(), self.render_resources.as_ref())
                    {
                        let bloom_view = self
                            .bloom_pass
                            .as_ref()
                            .map(|b| b.result_view())
                            .unwrap_or_else(|| self.black_bloom_view.as_ref().unwrap());
                        let offscreen = esox_gfx::OffscreenTarget::new(
                            &gpu.device,
                            gpu.config.width,
                            gpu.config.height,
                            gpu.config.format,
                            &pp_layout,
                            &resources.uniform_buffer,
                            sampler,
                            bloom_view,
                        );
                        if let Some(bloom) = self.bloom_pass.as_mut() {
                            bloom.update_scene_texture(&gpu.device, &offscreen.sample_view);
                        }
                        self.offscreen = Some(offscreen);
                    }
                    self.pp_bind_group_layout = Some(pp_layout);
                    self.pipeline_registry = Some(registry);
                }

                // Poll for async-compiled pipelines from the background thread.
                if let (Some(registry), Some(rx)) =
                    (self.pipeline_registry.as_mut(), self.pipeline_rx.as_ref())
                {
                    registry.poll_ready_pipelines(rx);
                }

                if let (Some(gpu), Some(resources)) =
                    (self.gpu.as_ref(), self.render_resources.as_mut())
                {
                    // Skip GPU submission if the delegate reports no damage.
                    if !self.delegate.needs_redraw() {
                        self.perf.frames_skipped += 1;
                        tracing::trace!("frame skipped (no damage), total skipped: {}", self.perf.frames_skipped);
                        self.redraw_pending = false;
                        return;
                    }

                    self.perf.begin_frame();
                    self.frame.clear();
                    self.delegate.on_redraw(gpu, resources, &mut self.frame, &self.perf);

                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    let delta = elapsed - self.last_frame_elapsed;
                    self.last_frame_elapsed = elapsed;
                    let uniforms = esox_gfx::FrameUniforms {
                        viewport: [
                            gpu.config.width as f32,
                            gpu.config.height as f32,
                            1.0 / gpu.config.width as f32,
                            1.0 / gpu.config.height as f32,
                        ],
                        time: [elapsed, delta, (self.frame_number % (1 << 23)) as f32, 0.0],
                    };

                    let registry = match self.pipeline_registry.as_ref() {
                        Some(r) => r,
                        None => {
                            tracing::error!("pipeline registry not initialized; skipping frame");
                            return;
                        }
                    };

                    // Acquire surface early so 3D pre-render can target it.
                    let surface = match gpu.acquire_surface() {
                        Ok(s) => s,
                        Err(e) => {
                            tracing::error!("surface acquisition failed: {e}");
                            self.consecutive_render_failures += 1;
                            if self.consecutive_render_failures >= 3 {
                                tracing::error!("3 consecutive render failures, exiting");
                                self.write_perf_report();
                                event_loop.exit();
                            }
                            return;
                        }
                    };

                    // Call pre-render (3D pass) — default is no-op.
                    let pre_render_bufs = self.delegate.on_pre_render(gpu, &surface.view);
                    let has_pre_render = !pre_render_bufs.is_empty();

                    // Submit 3D command buffers before the 2D pass.
                    if has_pre_render {
                        gpu.queue.submit(pre_render_bufs);
                    }

                    // Choose load op: Load if 3D was rendered, Clear otherwise.
                    let color_load_op = if has_pre_render {
                        esox_gfx::ColorLoadOp::Load
                    } else {
                        let bg = &self.clear_color;
                        esox_gfx::ColorLoadOp::Clear(wgpu::Color {
                            r: bg.r as f64,
                            g: bg.g as f64,
                            b: bg.b as f64,
                            a: bg.a as f64,
                        })
                    };

                    let pp = if self.delegate.post_process_enabled() {
                        if let Some(offscreen) = self.offscreen.as_ref() {
                            let mut params = self.delegate.post_process_params();
                            params.time = elapsed;
                            offscreen.update_params(&gpu.queue, &params);
                        }
                        self.offscreen
                            .as_ref()
                            .map(|offscreen| esox_gfx::PostProcessPass {
                                offscreen,
                                pipeline_id: esox_gfx::PIPELINE_POST_PROCESS,
                                bloom: self.bloom_pass.as_ref(),
                            })
                    } else {
                        None
                    };

                    // Create screenshot capture buffer if requested.
                    let screenshot_capture = if self.screenshot_pending {
                        self.screenshot_pending = false;
                        Some(esox_gfx::ScreenshotCapture::new(
                            &gpu.device,
                            gpu.config.width,
                            gpu.config.height,
                            gpu.config.format,
                        ))
                    } else {
                        None
                    };

                    if let Err(e) = esox_gfx::FrameEncoder::encode_and_submit_with_surface(
                        gpu,
                        resources,
                        &mut self.frame,
                        &uniforms,
                        registry,
                        surface,
                        color_load_op,
                        pp,
                        self.msaa_view.as_ref(),
                        self.depth_view.as_ref(),
                        screenshot_capture.as_ref(),
                    ) {
                        tracing::error!("render error: {e}");
                        self.consecutive_render_failures += 1;
                        if self.consecutive_render_failures >= 3 {
                            tracing::error!("3 consecutive render failures, exiting");
                            self.write_perf_report();
                            event_loop.exit();
                        }
                        return;
                    }
                    self.consecutive_render_failures = 0;

                    // Save screenshot on a background thread.
                    if let Some(capture) = screenshot_capture {
                        let device = gpu.device.clone();
                        let path = screenshot_path();
                        std::thread::spawn(move || {
                            capture.save_blocking(&device, path);
                        });
                    }

                    // Read counts after encoding (build_batches runs inside the encoder).
                    let instance_count = self.frame.instance_count() as u32;
                    let batch_count = self.frame.batch_count() as u32;
                    self.perf.end_frame(instance_count, batch_count);
                    self.frame_number += 1;
                }
                // Check if the delegate wants to exit.
                if self.delegate.should_exit() {
                    self.write_perf_report();
                    self.delegate.on_close();
                    event_loop.exit();
                    return;
                }

                // Update window title if the delegate has a new one.
                if let Some(title) = self.delegate.take_title()
                    && let Some(window) = self.window.as_ref()
                {
                    window.set_title(&title);
                }

                // Apply settings-driven window changes.
                if let Some(title) = self.delegate.take_settings_title()
                    && let Some(window) = self.window.as_ref()
                {
                    window.set_title(&title);
                }
                if let Some(decorated) = self.delegate.take_decorations()
                    && let Some(window) = self.window.as_ref()
                {
                    window.set_decorations(decorated);
                }
                if let Some(rgba) = self.delegate.take_clear_color() {
                    self.clear_color = esox_gfx::Color::new(rgba[0], rgba[1], rgba[2], rgba[3]);
                }

                // Handle bell — request urgency hint from the window manager.
                if self.delegate.has_bell()
                    && let Some(window) = self.window.as_ref()
                {
                    window.request_user_attention(Some(
                        winit::window::UserAttentionType::Informational,
                    ));
                    self.delegate.on_bell();
                }
                self.redraw_pending = false;
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppUserEvent) {
        if matches!(event, AppUserEvent::ShaderFileChanged) {
            self.shader_reload_pending = true;
        }
        // A background thread (PTY watcher, blink timer, or shader watcher) wants a redraw.
        if !self.redraw_pending
            && let Some(window) = self.window.as_ref()
        {
            window.request_redraw();
            self.redraw_pending = true;
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        // When the cursor is grabbed, CursorMoved window events stop firing.
        // Raw device motion still arrives here, so forward it to the delegate.
        if self.cursor_grabbed {
            if let winit::event::DeviceEvent::MouseMotion { delta } = event {
                self.delegate
                    .on_mouse(MouseInputEvent::RawMotion { dx: delta.0, dy: delta.1 });
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Check for signal-triggered exit (SIGTERM/SIGINT).
        if SIGNAL_EXIT.load(Ordering::SeqCst) {
            self.write_perf_report();
            self.delegate.on_close();
            event_loop.exit();
            // Exit cleanly to avoid winit teardown issues from signal context.
            std::process::exit(0);
        }
        if self.redraw_pending {
            return;
        }
        if self.delegate.needs_continuous_redraw() {
            let effective_fps = self
                .config
                .frame
                .max_fps
                .map(|fps| fps.min(self.monitor_refresh_hz))
                .unwrap_or(self.monitor_refresh_hz);
            let target_interval =
                std::time::Duration::from_secs_f64(1.0 / effective_fps.max(1) as f64);
            let now = std::time::Instant::now();
            let next_redraw = self.last_redraw + target_interval;
            if now >= next_redraw {
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                    self.redraw_pending = true;
                }
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(next_redraw));
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

/// Run the application event loop.
///
/// This creates a winit event loop, builds an [`App`] from the given config
/// and delegate, and blocks until the window is closed.
/// Global flag set by signal handlers to request graceful shutdown.
static SIGNAL_EXIT: AtomicBool = AtomicBool::new(false);

/// Install signal handlers so SIGTERM/SIGINT trigger a graceful exit
/// (allowing perf report to be written).
/// Generate a timestamped screenshot file path in the user's Pictures directory
/// (or current directory as fallback).
fn screenshot_path() -> std::path::PathBuf {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let dir = std::env::var_os("XDG_PICTURES_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            dirs_fallback().map(|home| home.join("Pictures"))
        })
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    // Ensure the directory exists.
    let _ = std::fs::create_dir_all(&dir);

    dir.join(format!("screenshot_{timestamp}.png"))
}

/// Best-effort home directory lookup without adding a dependency.
fn dirs_fallback() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

fn install_signal_handlers() {
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" fn handler(_sig: libc::c_int) {
            SIGNAL_EXIT.store(true, Ordering::SeqCst);
        }
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handler as *const () as usize;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
    }
}

pub fn run(
    #[cfg_attr(not(feature = "settings"), allow(unused_mut))]
    mut config: crate::config::PlatformConfig,
    delegate: Box<dyn AppDelegate>,
) -> Result<(), Error> {
    install_signal_handlers();

    // Auto-restore window state from settings if app_name is set.
    #[cfg(feature = "settings")]
    if let Some(ref app_name) = config.app_name {
        let dirs = crate::xdg::AppDirs::new(app_name);
        if let Some(ws) = crate::settings::WindowState::load(&dirs) {
            tracing::info!("restored window state: {}x{}", ws.width, ws.height);
            ws.apply_to(&mut config.window);
        }
    }

    let event_loop = winit::event_loop::EventLoop::<AppUserEvent>::with_user_event()
        .build()
        .map_err(|e| Error::EventLoop(e.to_string()))?;
    let proxy = event_loop.create_proxy();
    let mut app = App::new(config, delegate);
    app.event_proxy = Some(proxy.clone());
    app.delegate.set_waker(Waker { proxy: proxy.clone() });

    // Start portal bridge if feature is enabled.
    #[cfg(feature = "portals")]
    {
        let handle = crate::portal::start_portal_bridge(proxy);
        app.delegate.set_portal_handle(handle);
    }

    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::EventLoop(e.to_string()))?;

    // Auto-save window state on exit.
    #[cfg(feature = "settings")]
    if let Some(ref app_name) = app.config.app_name {
        if let Some(ref window) = app.window {
            let dirs = crate::xdg::AppDirs::new(app_name);
            let ws = crate::settings::WindowState::from_window(window);
            if let Err(e) = ws.save(&dirs) {
                tracing::warn!("failed to save window state: {e}");
            }
        }
    }

    // Write report after event loop exits (covers normal close).
    app.write_perf_report();
    Ok(())
}

/// Clipboard access (read/write) via arboard.
pub struct Clipboard;

impl Clipboard {
    /// Read text from the system clipboard, truncated to `max_bytes`.
    ///
    /// Pass `0` for unlimited. Truncation avoids OOM when the clipboard holds
    /// very large data and a program queries it via OSC 52.
    pub fn read(max_bytes: usize) -> Result<String, Error> {
        let mut clip = arboard::Clipboard::new()
            .map_err(|e| Error::Clipboard(format!("failed to open clipboard: {e}")))?;
        let text = clip
            .get_text()
            .map_err(|e| Error::Clipboard(format!("failed to read clipboard: {e}")))?;
        if max_bytes > 0 && text.len() > max_bytes {
            // Truncate at a char boundary to avoid splitting a multi-byte character.
            let truncated = &text[..text.floor_char_boundary(max_bytes)];
            Ok(truncated.to_string())
        } else {
            Ok(text)
        }
    }

    /// Write text to the system clipboard.
    pub fn write(text: &str) -> Result<(), Error> {
        let mut clip = arboard::Clipboard::new()
            .map_err(|e| Error::Clipboard(format!("failed to open clipboard: {e}")))?;
        clip.set_text(text.to_owned())
            .map_err(|e| Error::Clipboard(format!("failed to write clipboard: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_window_creation() {
        let e = Error::WindowCreation("no display".into());
        assert_eq!(e.to_string(), "failed to create window: no display");
    }

    #[test]
    fn error_display_event_loop() {
        let e = Error::EventLoop("loop died".into());
        assert_eq!(e.to_string(), "event loop error: loop died");
    }

    #[test]
    fn error_display_clipboard() {
        let e = Error::Clipboard("no clipboard".into());
        assert_eq!(e.to_string(), "clipboard error: no clipboard");
    }

    // --- Keyboard shortcut detection tests ---

    #[test]
    fn copy_shortcut_with_modifiers() {
        use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
        let mods = ModifiersState::CONTROL | ModifiersState::SHIFT;
        let physical = PhysicalKey::Code(KeyCode::KeyC);
        let logical = winit::keyboard::Key::Character("C".into());
        assert!(is_copy_shortcut(physical, &logical, mods, Some("\x03")));
    }

    #[test]
    fn paste_shortcut_with_modifiers() {
        use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
        let mods = ModifiersState::CONTROL | ModifiersState::SHIFT;
        let physical = PhysicalKey::Code(KeyCode::KeyV);
        let logical = winit::keyboard::Key::Character("V".into());
        assert!(is_paste_shortcut(physical, &logical, mods, Some("\x16")));
    }

    #[test]
    fn copy_shortcut_without_ctrl_rejected() {
        use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
        // Only Shift, no Ctrl.
        let mods = ModifiersState::SHIFT;
        let physical = PhysicalKey::Code(KeyCode::KeyC);
        let logical = winit::keyboard::Key::Character("C".into());
        assert!(!is_copy_shortcut(physical, &logical, mods, Some("C")));
    }

    #[test]
    fn paste_shortcut_wrong_key_rejected() {
        use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
        let mods = ModifiersState::CONTROL | ModifiersState::SHIFT;
        // Physical key is C, not V.
        let physical = PhysicalKey::Code(KeyCode::KeyC);
        let logical = winit::keyboard::Key::Character("C".into());
        assert!(!is_paste_shortcut(physical, &logical, mods, Some("\x03")));
    }

    #[test]
    fn copy_shortcut_fallback_from_event() {
        use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
        // No modifier bits set (simulates Wayland late ModifiersChanged).
        let mods = ModifiersState::empty();
        let physical = PhysicalKey::Code(KeyCode::KeyC);
        let logical = winit::keyboard::Key::Character("C".into());
        // text_with_all_modifiers indicates Ctrl is held (control char < 0x20).
        assert!(is_copy_shortcut(physical, &logical, mods, Some("\x03")));
    }

    // --- Mouse button classification tests ---

    #[test]
    fn classify_mouse_button_left() {
        assert_eq!(classify_mouse_button(winit::event::MouseButton::Left), 0);
    }

    #[test]
    fn classify_mouse_button_middle() {
        assert_eq!(classify_mouse_button(winit::event::MouseButton::Middle), 1);
    }

    #[test]
    fn classify_mouse_button_right() {
        assert_eq!(classify_mouse_button(winit::event::MouseButton::Right), 2);
    }

    #[test]
    fn classify_mouse_button_other() {
        assert_eq!(
            classify_mouse_button(winit::event::MouseButton::Other(4)),
            3
        );
    }
}
