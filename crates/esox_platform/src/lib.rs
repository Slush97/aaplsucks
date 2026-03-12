//! `esox_platform` — Windowing, input dispatch, and platform integration.

pub mod config;
pub mod perf;
pub mod sandbox;

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
        event: &winit::event::KeyEvent,
        modifiers: winit::keyboard::ModifiersState,
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

    /// Called once before the event loop starts, providing the proxy for
    /// background threads to wake the event loop.
    fn set_event_loop_proxy(&mut self, _proxy: EventLoopProxy<AppUserEvent>) {}

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

    /// Maximum frames per second for continuous animations (0 = monitor refresh rate).
    fn max_fps(&self) -> u32 {
        0
    }

    /// Whether the application should exit (e.g., child shell has exited).
    fn should_exit(&self) -> bool {
        false
    }

    /// Called just before the window is destroyed (close or exit).
    fn on_close(&mut self, _window: &winit::window::Window) {}

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
    fn cursor_icon(&self, _x: f64, _y: f64) -> winit::window::CursorIcon {
        winit::window::CursorIcon::Text
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
    /// Cursor left the window surface.
    Left,
}

/// Detect whether a key event represents Ctrl+Shift+C (copy shortcut).
///
/// Uses the physical key to be layout-independent. Accepts two modifier
/// sources to work around Wayland timing issues where `ModifiersChanged`
/// may arrive late.
pub fn is_copy_shortcut(
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
pub fn is_paste_shortcut(
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
    /// Count of consecutive render failures (for device-lost recovery).
    consecutive_render_failures: u32,
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
            consecutive_render_failures: 0,
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
        if let Some(monitor) = window.current_monitor()
            && let Some(hz) = monitor
                .video_modes()
                .map(|m| m.refresh_rate_millihertz() / 1000)
                .max()
            && hz > 0
        {
            self.monitor_refresh_hz = hz;
            tracing::debug!("monitor refresh rate: {}Hz", hz);
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
                if let Some(w) = self.window.as_ref() {
                    self.delegate.on_close(w);
                }
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
                if event.state == winit::event::ElementState::Pressed {
                    // Intercept Ctrl+Shift+C (copy) and Ctrl+Shift+V (paste)
                    // using extracted helpers for testability.
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

                    self.delegate.on_key(&event, self.current_modifiers);
                    // Trigger immediate redraw so PTY response is picked up quickly.
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
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
                // Update OS cursor icon based on pointer position.
                if let Some(window) = self.window.as_ref() {
                    let icon = self.delegate.cursor_icon(position.x, position.y);
                    window.set_cursor(winit::window::Cursor::Icon(icon));
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
                    winit::event::Ime::Preedit(_, _)
                    | winit::event::Ime::Enabled
                    | winit::event::Ime::Disabled => {
                        // Preedit rendering could be added later.
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

                    let pp = if self.delegate.post_process_enabled() {
                        // Upload post-process params before encoding.
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

                    if let Err(e) = esox_gfx::FrameEncoder::encode_and_submit_with_post_process(
                        gpu,
                        resources,
                        &mut self.frame,
                        &uniforms,
                        &self.clear_color,
                        registry,
                        pp,
                        self.msaa_view.as_ref(),
                        self.depth_view.as_ref(),
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

                    // Read counts after encoding (build_batches runs inside the encoder).
                    let instance_count = self.frame.instance_count() as u32;
                    let batch_count = self.frame.batch_count() as u32;
                    self.perf.end_frame(instance_count, batch_count);
                    self.frame_number += 1;
                }
                // Check if the delegate wants to exit.
                if self.delegate.should_exit() {
                    self.write_perf_report();
                    if let Some(w) = self.window.as_ref() {
                        self.delegate.on_close(w);
                    }
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

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Check for signal-triggered exit (SIGTERM/SIGINT).
        if SIGNAL_EXIT.load(Ordering::SeqCst) {
            self.write_perf_report();
            if let Some(w) = self.window.as_ref() {
                self.delegate.on_close(w);
            }
            event_loop.exit();
            // Exit cleanly to avoid winit teardown issues from signal context.
            std::process::exit(0);
        }
        if self.redraw_pending {
            return;
        }
        if self.delegate.needs_continuous_redraw() {
            let max_fps = self.delegate.max_fps();
            let effective_fps = if max_fps == 0 {
                self.monitor_refresh_hz
            } else {
                max_fps.min(self.monitor_refresh_hz)
            };
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

pub fn run(config: crate::config::PlatformConfig, delegate: Box<dyn AppDelegate>) -> Result<(), Error> {
    install_signal_handlers();
    let event_loop = winit::event_loop::EventLoop::<AppUserEvent>::with_user_event()
        .build()
        .map_err(|e| Error::EventLoop(e.to_string()))?;
    let proxy = event_loop.create_proxy();
    let mut app = App::new(config, delegate);
    app.event_proxy = Some(proxy.clone());
    app.delegate.set_event_loop_proxy(proxy);
    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::EventLoop(e.to_string()))?;
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
