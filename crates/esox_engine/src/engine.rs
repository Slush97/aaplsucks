//! Engine — implements AppDelegate internally, bridges to Game trait.

use esox_gfx::mesh3d::Renderer3D;
use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::perf::PerfMonitor;
use esox_platform::{AppDelegate, MouseInputEvent};

use crate::assets::AssetManager;
#[cfg(feature = "ui")]
use crate::debug_overlay::{EngineStats, draw_debug_overlay};
use crate::ecs::{
    animation_system, camera_sync_system, chunked_render_extraction_system, hierarchy_system,
    light_collection_system, particle_system, physics_sync_system, render_extraction_system,
};
use crate::game::Game;
use crate::input::InputManager;
use crate::physics::{NullPhysics, PhysicsBackend};
use crate::physics::entity_map::PhysicsEntityMap;
use crate::time::FixedTimestep;
use crate::Ctx;

/// Engine configuration.
pub struct EngineConfig {
    /// Platform/window config.
    pub platform: esox_platform::config::PlatformConfig,
    /// 3D clear color.
    pub clear_color: wgpu::Color,
    /// Enable post-processing (bloom, tone mapping, SSAO).
    pub postprocess: bool,
    /// Post-process settings (bloom intensity, tone mapping, SSAO, fog).
    /// Only used when `postprocess` is `true`. Defaults to
    /// `PostProcess3DConfig::default()`.
    pub postprocess_config: esox_gfx::mesh3d::PostProcess3DConfig,
    /// Enable shadows.
    pub shadows: bool,
    /// Optional physics backend. Defaults to `NullPhysics` if `None`.
    pub physics: Option<Box<dyn PhysicsBackend>>,
    /// Optional chunk config for spatial world partitioning.
    pub chunk_config: Option<crate::chunk::ChunkConfig>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            platform: esox_platform::config::PlatformConfig {
                window: esox_platform::config::WindowConfig {
                    title: "esox engine".into(),
                    width: Some(1280),
                    height: Some(720),
                    ..Default::default()
                },
                ..Default::default()
            },
            clear_color: wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            },
            postprocess: true,
            postprocess_config: esox_gfx::mesh3d::PostProcess3DConfig {
                ssao_enabled: true,
                ..esox_gfx::mesh3d::PostProcess3DConfig::default()
            },
            shadows: true,
            physics: None,
            chunk_config: None,
        }
    }
}

// ── Engine subsystem state ──────────────────────────────────────────────────
//
// Separated from `Game` and `Renderer3D` so the borrow checker can split
// borrows cleanly: `self.game`, `self.renderer`, and `self.state` are three
// independent fields that can be borrowed simultaneously.

/// All engine subsystem state except the game logic and the 3D renderer.
struct EngineState {
    config: EngineConfig,
    world: hecs::World,
    input: InputManager,
    timestep: FixedTimestep,
    assets: AssetManager,
    physics: Box<dyn PhysicsBackend>,
    entity_map: PhysicsEntityMap,
    viewport: (u32, u32),
    frame_count: u32,
    initialized: bool,
    last_camera_pos: glam::Vec3,
    chunk_manager: Option<crate::chunk::ChunkManager>,
    #[cfg(feature = "audio")]
    audio: Option<crate::audio::AudioManager>,
    #[cfg(feature = "ui")]
    debug_overlay_visible: bool,
    #[cfg(feature = "ui")]
    last_batch_stats: esox_gfx::mesh3d::BatchStats3D,
    #[cfg(feature = "ui")]
    last_physics_us: u64,
    #[cfg(feature = "ui")]
    ui_state: esox_ui::UiState,
    #[cfg(feature = "ui")]
    text_renderer: Option<esox_ui::TextRenderer>,
    #[cfg(feature = "ui")]
    theme: esox_ui::Theme,
}

impl EngineState {
    /// Build a [`Ctx`] from engine state plus an externally-borrowed renderer.
    ///
    /// This borrows **all** of `EngineState` mutably, so nothing else on
    /// `self` can be accessed while the returned `Ctx` is alive. Use this
    /// for the game callbacks (`init`, `update`, `render`) where the game
    /// gets full access to every subsystem.
    fn make_ctx<'a>(
        &'a mut self,
        gpu: &'a GpuContext,
        renderer: &'a mut Renderer3D,
    ) -> Ctx<'a> {
        Ctx {
            world: &mut self.world,
            input: &mut self.input,
            time: &self.timestep.time_state_cache,
            renderer,
            gpu,
            assets: &mut self.assets,
            physics: &mut *self.physics,
            entity_map: &mut self.entity_map,
            viewport: self.viewport,
            chunks: self.chunk_manager.as_mut(),
            #[cfg(feature = "audio")]
            audio: self.audio.as_mut(),
        }
    }

    /// Process completed asset uploads and poll hot-reload.
    fn process_assets(&mut self, gpu: &GpuContext, renderer: &mut Renderer3D) {
        self.assets.process_uploads(gpu, renderer);
        #[cfg(feature = "hot-reload")]
        self.assets.poll_asset_reload(gpu, renderer);
    }

    /// Run built-in ECS systems: hierarchy, lights, render extraction,
    /// animation, and particles.
    fn run_ecs_systems(&mut self, gpu: &GpuContext, renderer: &mut Renderer3D) {
        hierarchy_system(&mut self.world);

        let lights = light_collection_system(&self.world);
        renderer.set_lights(&lights);

        if let Some(ref chunk_mgr) = self.chunk_manager {
            chunked_render_extraction_system(
                &self.world,
                renderer,
                self.last_camera_pos,
                chunk_mgr,
            );
        } else {
            render_extraction_system(&self.world, renderer, self.last_camera_pos);
        }

        let frame_dt = self.timestep.time_state_cache.frame_dt;
        animation_system(&mut self.world, renderer, gpu, frame_dt);
        particle_system(&mut self.world, renderer, gpu, frame_dt, self.frame_count);
        self.frame_count = self.frame_count.wrapping_add(1);
    }

    /// Dispatch GPU compute work (skinning, particles).
    fn dispatch_compute(
        &self,
        gpu: &GpuContext,
        renderer: &mut Renderer3D,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut cmd_bufs = Vec::new();
        if let Some(skin_cmd) = renderer.dispatch_skinning(gpu) {
            cmd_bufs.push(skin_cmd);
        }
        if let Some(particle_cmd) = renderer.dispatch_particles(gpu) {
            cmd_bufs.push(particle_cmd);
        }
        cmd_bufs
    }

    /// Sync the active camera from ECS and update the audio listener.
    fn sync_camera_and_audio(&mut self) -> esox_gfx::mesh3d::Camera {
        let camera = camera_sync_system(&self.world).unwrap_or_default();
        self.last_camera_pos = camera.position;
        #[cfg(feature = "audio")]
        if let Some(ref mut audio) = self.audio {
            let cam_forward = (camera.target - camera.position).normalize_or_zero();
            audio.set_listener(camera.position, cam_forward, camera.up);
        }
        camera
    }
}

// ── Engine ──────────────────────────────────────────────────────────────────

/// The engine manages the game loop, ECS world, and all subsystems.
pub(crate) struct Engine {
    game: Box<dyn Game>,
    renderer: Option<Renderer3D>,
    state: EngineState,
}

impl Engine {
    pub fn new(mut config: EngineConfig, game: Box<dyn Game>) -> Self {
        let tick_rate = config.platform.frame.tick_rate;
        let physics = config.physics.take().unwrap_or_else(|| Box::new(NullPhysics));
        let chunk_manager = config
            .chunk_config
            .take()
            .map(crate::chunk::ChunkManager::new);
        Self {
            game,
            renderer: None,
            state: EngineState {
                config,
                world: hecs::World::new(),
                input: InputManager::new(),
                timestep: FixedTimestep::new(tick_rate),
                assets: AssetManager::new(),
                physics,
                entity_map: PhysicsEntityMap::new(),
                viewport: (1280, 720),
                frame_count: 0,
                initialized: false,
                last_camera_pos: glam::Vec3::ZERO,
                chunk_manager,
                #[cfg(feature = "audio")]
                audio: crate::audio::AudioManager::new(),
                #[cfg(feature = "ui")]
                debug_overlay_visible: false,
                #[cfg(feature = "ui")]
                last_batch_stats: esox_gfx::mesh3d::BatchStats3D::default(),
                #[cfg(feature = "ui")]
                last_physics_us: 0,
                #[cfg(feature = "ui")]
                ui_state: esox_ui::UiState::new(),
                #[cfg(feature = "ui")]
                text_renderer: None,
                #[cfg(feature = "ui")]
                theme: esox_ui::Theme::dark(),
            },
        }
    }

}

// ── AppDelegate impl ────────────────────────────────────────────────────────

impl AppDelegate for Engine {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        let mut renderer = Renderer3D::new(gpu);

        if self.state.config.postprocess {
            renderer.enable_postprocess(gpu);
            renderer.set_postprocess(self.state.config.postprocess_config);
            if self.state.config.postprocess_config.ssao_enabled {
                renderer.enable_ssao(gpu);
            }
        }

        if self.state.config.shadows {
            renderer.enable_shadows(gpu);
            renderer.enable_point_shadows(gpu);
            renderer.enable_spot_shadows(gpu);
        }

        #[cfg(feature = "ui")]
        {
            match esox_ui::TextRenderer::new(gpu) {
                Ok(tr) => self.state.text_renderer = Some(tr),
                Err(e) => tracing::warn!("Failed to init TextRenderer: {e}"),
            }
        }

        self.state.timestep.time_state_cache = self.state.timestep.time_state(0);
        let mut ctx = self.state.make_ctx(gpu, &mut renderer);
        self.game.init(&mut ctx);

        self.renderer = Some(renderer);
        self.state.initialized = true;
    }

    fn on_pre_render(
        &mut self,
        gpu: &GpuContext,
        surface_view: &wgpu::TextureView,
    ) -> Vec<wgpu::CommandBuffer> {
        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return vec![],
        };

        // 1. Hot-reload.
        #[cfg(feature = "hot-reload")]
        renderer.poll_shader_reload(gpu);

        // 2. Advance timestep.
        let (tick_count, alpha) = self.state.timestep.advance();
        self.state.timestep.time_state_cache = self.state.timestep.time_state(tick_count);

        // 3. Process assets.
        self.state.process_assets(gpu, renderer);

        // 4. Fixed-rate update loop.
        #[cfg(feature = "ui")]
        let mut physics_us: u64 = 0;
        self.state.input.begin_frame(tick_count);
        for tick_i in 0..tick_count {
            self.state.input.pre_update();
            self.state.timestep.time_state_cache = self.state.timestep.time_state(tick_i + 1);
            {
                let mut ctx = self.state.make_ctx(gpu, renderer);
                self.game.update(&mut ctx);
            }

            #[cfg(feature = "ui")]
            let phys_start = std::time::Instant::now();
            self.state.physics.step(self.state.timestep.tick_dt);
            #[cfg(feature = "ui")]
            {
                physics_us += phys_start.elapsed().as_micros() as u64;
            }

            physics_sync_system(&mut self.state.world, &*self.state.physics);
            self.state.input.post_update();
        }
        self.state.input.end_frame();
        #[cfg(feature = "ui")]
        {
            self.state.last_physics_us = physics_us;
        }

        // 5. Variable-rate render.
        {
            self.state.timestep.time_state_cache = self.state.timestep.time_state(tick_count);
            let mut ctx = self.state.make_ctx(gpu, renderer);
            self.game.render(&mut ctx, alpha);
        }

        // 6. ECS systems.
        self.state.run_ecs_systems(gpu, renderer);

        // 7. Dispatch compute.
        let mut cmd_bufs = self.state.dispatch_compute(gpu, renderer);

        // 8. Camera + audio sync.
        let camera = self.state.sync_camera_and_audio();

        // 9. Encode 3D render pass.
        let elapsed = self.state.timestep.time_state_cache.elapsed;
        let dt = self.state.timestep.time_state_cache.frame_dt;
        let (render_cmd, batch_stats) = renderer.encode(
            gpu,
            surface_view,
            &camera,
            self.state.viewport.0,
            self.state.viewport.1,
            elapsed,
            dt,
            self.state.config.clear_color,
        );
        #[cfg(feature = "ui")]
        {
            self.state.last_batch_stats = batch_stats;
        }
        #[cfg(not(feature = "ui"))]
        let _ = batch_stats;
        cmd_bufs.push(render_cmd);

        cmd_bufs
    }

    fn on_redraw(
        &mut self,
        _gpu: &GpuContext,
        _resources: &mut RenderResources,
        _frame: &mut Frame,
        _perf: &PerfMonitor,
    ) {
        #[cfg(feature = "ui")]
        {
            let text = match self.state.text_renderer.as_mut() {
                Some(t) => t,
                None => return,
            };

            self.state.ui_state.update_blink(self.state.theme.cursor_blink_ms);

            let vp = esox_ui::Rect::new(
                0.0,
                0.0,
                self.state.viewport.0 as f32,
                self.state.viewport.1 as f32,
            );

            let renderer = self.renderer.as_mut().unwrap();
            let time_state = &self.state.timestep.time_state_cache;

            let ctx = Ctx {
                world: &mut self.state.world,
                input: &mut self.state.input,
                time: time_state,
                renderer,
                gpu: _gpu,
                assets: &mut self.state.assets,
                physics: &mut *self.state.physics,
                entity_map: &mut self.state.entity_map,
                viewport: self.state.viewport,
                chunks: self.state.chunk_manager.as_mut(),
                #[cfg(feature = "audio")]
                audio: self.state.audio.as_mut(),
            };

            let mut ui = esox_ui::Ui::begin(
                _frame, _gpu, _resources, text,
                &mut self.state.ui_state, &self.state.theme, vp,
            );

            self.game.ui(&mut ui, &ctx);

            ui.finish();

            if self.state.debug_overlay_visible {
                let text = self.state.text_renderer.as_mut().unwrap();
                let stats = EngineStats {
                    physics_step_us: self.state.last_physics_us,
                    batch_stats: self.state.last_batch_stats,
                    entity_count: self.state.world.len() as usize,
                };
                draw_debug_overlay(
                    _frame, _gpu, _resources, text,
                    &stats, _perf, self.state.viewport,
                );
            }
        }
    }

    fn on_key(
        &mut self,
        event: &esox_input::KeyEvent,
        _modifiers: esox_input::Modifiers,
    ) {
        #[cfg(feature = "ui")]
        {
            use esox_input::KeyCode;
            if event.pressed
                && event.physical_key == KeyCode::F3
            {
                self.state.debug_overlay_visible = !self.state.debug_overlay_visible;
            }
        }
        self.state.input.handle_key_event(event);
        #[cfg(feature = "ui")]
        self.state.ui_state.process_key(event.clone(), _modifiers);
    }

    fn on_resize(&mut self, width: u32, height: u32, _gpu: &GpuContext) {
        self.state.viewport = (width, height);
    }

    fn on_mouse(&mut self, event: MouseInputEvent) {
        match event {
            MouseInputEvent::Moved { x, y } => {
                self.state.input.handle_mouse_move(x, y);
                #[cfg(feature = "ui")]
                self.state.ui_state.process_mouse_move(
                    x as f32,
                    y as f32,
                    self.state.theme.item_height,
                    self.state.theme.dropdown_gap,
                );
            }
            MouseInputEvent::Press {
                button,
                x: _x,
                y: _y,
            } => {
                self.state.input.handle_mouse_button(button, true);
                #[cfg(feature = "ui")]
                if button == 0 {
                    self.state.ui_state.process_mouse_click(_x as f32, _y as f32);
                } else if button == 2 {
                    self.state.ui_state.process_right_click(_x as f32, _y as f32);
                }
            }
            MouseInputEvent::Release { button, .. } => {
                self.state.input.handle_mouse_button(button, false);
                #[cfg(feature = "ui")]
                self.state.ui_state.process_mouse_release();
            }
            MouseInputEvent::Scroll {
                x: _x,
                y: _y,
                delta_y: _delta_y,
            } => {
                self.state.input.handle_scroll(_delta_y);
                #[cfg(feature = "ui")]
                self.state.ui_state.process_scroll(_x as f32, _y as f32, _delta_y);
            }
            MouseInputEvent::RawMotion { dx, dy } => {
                self.state.input.handle_raw_mouse_motion(dx, dy);
            }
            MouseInputEvent::Left => {}
        }
    }

    fn on_focus_changed(&mut self, focused: bool) {
        if !focused {
            self.state.input.clear_all_state();
        }
    }

    fn on_scale_changed(&mut self, _scale_factor: f64, _gpu: &GpuContext) {}

    fn on_paste(&mut self, _text: &str) {
        #[cfg(feature = "ui")]
        self.state.ui_state.on_ime_commit(_text.to_string());
    }

    fn on_ime_commit(&mut self, _text: &str) {
        #[cfg(feature = "ui")]
        self.state.ui_state.on_ime_commit(_text.to_string());
    }

    fn on_ime_preedit(&mut self, _text: String, _cursor: Option<(usize, usize)>) {
        #[cfg(feature = "ui")]
        self.state.ui_state.on_ime_preedit(_text, _cursor);
    }

    fn on_ime_enabled(&mut self, _enabled: bool) {
        #[cfg(feature = "ui")]
        self.state.ui_state.on_ime_enabled(_enabled);
    }

    fn on_copy(&mut self) -> Option<String> {
        None
    }

    fn needs_continuous_redraw(&self) -> bool {
        true
    }

    fn cursor_grabbed(&self) -> bool {
        self.state.input.cursor_grabbed()
    }

    fn should_exit(&self) -> bool {
        self.game.should_exit()
    }
}
