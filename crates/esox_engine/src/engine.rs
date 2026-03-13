//! Engine — implements AppDelegate internally, bridges to Game trait.

use esox_gfx::mesh3d::Renderer3D;
use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::perf::PerfMonitor;
use esox_platform::{AppDelegate, MouseInputEvent};

use crate::assets::AssetManager;
#[cfg(feature = "ui")]
use crate::debug_overlay::{EngineStats, draw_debug_overlay};
use crate::ecs::{
    animation_system, camera_sync_system, hierarchy_system, light_collection_system,
    particle_system, physics_sync_system, render_extraction_system,
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
    /// Fixed update rate in Hz (default: 60).
    pub tick_rate: f32,
    /// 3D clear color.
    pub clear_color: wgpu::Color,
    /// Enable post-processing (bloom, tone mapping, SSAO).
    pub postprocess: bool,
    /// Enable shadows.
    pub shadows: bool,
    /// Optional physics backend. Defaults to `NullPhysics` if `None`.
    pub physics: Option<Box<dyn PhysicsBackend>>,
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
            tick_rate: 60.0,
            clear_color: wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            },
            postprocess: true,
            shadows: true,
            physics: None,
        }
    }
}

/// The engine manages the game loop, ECS world, and all subsystems.
pub(crate) struct Engine {
    pub config: EngineConfig,
    game: Box<dyn Game>,
    renderer: Option<Renderer3D>,
    world: hecs::World,
    input: InputManager,
    timestep: FixedTimestep,
    assets: AssetManager,
    physics: Box<dyn PhysicsBackend>,
    entity_map: PhysicsEntityMap,
    viewport: (u32, u32),
    initialized: bool,
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

impl Engine {
    pub fn new(mut config: EngineConfig, game: Box<dyn Game>) -> Self {
        let tick_rate = config.tick_rate;
        let physics = config.physics.take().unwrap_or_else(|| Box::new(NullPhysics));
        Self {
            config,
            game,
            renderer: None,
            world: hecs::World::new(),
            input: InputManager::new(),
            timestep: FixedTimestep::new(tick_rate),
            assets: AssetManager::new(),
            physics,
            entity_map: PhysicsEntityMap::new(),
            viewport: (1280, 720),
            initialized: false,
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
        }
    }

    #[allow(dead_code)]
    pub fn set_physics(&mut self, physics: Box<dyn PhysicsBackend>) {
        self.physics = physics;
    }
}

impl AppDelegate for Engine {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        let mut renderer = Renderer3D::new(gpu);

        if self.config.postprocess {
            renderer.enable_postprocess(gpu);
            renderer.set_postprocess(esox_gfx::mesh3d::PostProcess3DConfig {
                bloom_enabled: true,
                bloom_intensity: 0.3,
                bloom_threshold: 1.0,
                bloom_soft_knee: 0.5,
                tone_map_enabled: true,
                ssao_enabled: true,
                motion_blur_enabled: false,
            });
            renderer.enable_ssao(gpu);
        }

        if self.config.shadows {
            renderer.enable_shadows(gpu);
        }

        #[cfg(feature = "ui")]
        {
            match esox_ui::TextRenderer::new(gpu) {
                Ok(tr) => self.text_renderer = Some(tr),
                Err(e) => tracing::warn!("Failed to init TextRenderer: {e}"),
            }
        }

        // Call game init.
        self.timestep.time_state_cache = self.timestep.time_state(0);
        let mut ctx = Ctx {
            world: &mut self.world,
            input: &mut self.input,
            time: &self.timestep.time_state_cache,
            renderer: &mut renderer,
            gpu,
            assets: &mut self.assets,
            physics: &mut *self.physics,
            entity_map: &mut self.entity_map,
            viewport: self.viewport,
        };
        self.game.init(&mut ctx);

        self.renderer = Some(renderer);
        self.initialized = true;
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

        // 0. Poll shader hot-reload.
        #[cfg(feature = "hot-reload")]
        renderer.poll_shader_reload(gpu);

        // 1. Advance timestep.
        let (tick_count, alpha) = self.timestep.advance();
        self.timestep.time_state_cache = self.timestep.time_state(tick_count);

        // 2. Process completed asset uploads.
        self.assets.process_uploads(gpu, renderer);

        // 3. Fixed-rate update loop.
        #[cfg(feature = "ui")]
        let mut physics_us: u64 = 0;
        self.input.begin_frame(tick_count);
        for tick_i in 0..tick_count {
            self.input.pre_update();

            self.timestep.time_state_cache = self.timestep.time_state(tick_i + 1);
            {
                let mut ctx = Ctx {
                    world: &mut self.world,
                    input: &mut self.input,
                    time: &self.timestep.time_state_cache,
                    renderer,
                    gpu,
                    assets: &mut self.assets,
                    physics: &mut *self.physics,
                    entity_map: &mut self.entity_map,
                    viewport: self.viewport,
                };
                self.game.update(&mut ctx);
            }

            #[cfg(feature = "ui")]
            let phys_start = std::time::Instant::now();
            self.physics.step(self.timestep.tick_dt);
            #[cfg(feature = "ui")]
            {
                physics_us += phys_start.elapsed().as_micros() as u64;
            }

            physics_sync_system(&mut self.world, &*self.physics);

            self.input.post_update();
        }
        self.input.end_frame();
        #[cfg(feature = "ui")]
        {
            self.last_physics_us = physics_us;
        }

        // 4. Variable-rate render callback.
        {
            self.timestep.time_state_cache = self.timestep.time_state(tick_count);
            let mut ctx = Ctx {
                world: &mut self.world,
                input: &mut self.input,
                time: &self.timestep.time_state_cache,
                renderer,
                gpu,
                assets: &mut self.assets,
                physics: &mut *self.physics,
                entity_map: &mut self.entity_map,
                viewport: self.viewport,
            };
            self.game.render(&mut ctx, alpha);
        }

        // 5. Run ECS systems.
        hierarchy_system(&mut self.world);

        // 6. Light collection.
        let lights = light_collection_system(&self.world);
        renderer.set_lights(&lights);

        // 7. Render extraction — issue draw calls from ECS entities.
        render_extraction_system(&self.world, renderer);

        // 7.5. Animation — advance players and upload joint matrices.
        let frame_dt = self.timestep.time_state_cache.frame_dt;
        animation_system(&mut self.world, renderer, gpu, frame_dt);

        // 7.6. Particle system — advance emitters and queue particle draws.
        particle_system(&mut self.world, renderer, gpu, frame_dt);

        // 8. Dispatch skinning compute (if any).
        let mut cmd_bufs = Vec::new();
        if let Some(skin_cmd) = renderer.dispatch_skinning(gpu) {
            cmd_bufs.push(skin_cmd);
        }

        // 8.1. Dispatch particle compute (if any).
        if let Some(particle_cmd) = renderer.dispatch_particles(gpu) {
            cmd_bufs.push(particle_cmd);
        }

        // 9. Camera sync.
        let camera = camera_sync_system(&self.world).unwrap_or_default();

        // 10. Encode 3D render pass.
        let elapsed = self.timestep.time_state_cache.elapsed;
        let dt = self.timestep.time_state_cache.frame_dt;
        let (render_cmd, batch_stats) = renderer.encode(
            gpu,
            surface_view,
            &camera,
            self.viewport.0,
            self.viewport.1,
            elapsed,
            dt,
            self.config.clear_color,
        );
        #[cfg(feature = "ui")]
        {
            self.last_batch_stats = batch_stats;
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
            let text = match self.text_renderer.as_mut() {
                Some(t) => t,
                None => return,
            };

            self.ui_state.update_blink(self.theme.cursor_blink_ms);

            let vp = esox_ui::Rect::new(
                0.0,
                0.0,
                self.viewport.0 as f32,
                self.viewport.1 as f32,
            );

            let renderer = self.renderer.as_mut().unwrap();
            let time_state = &self.timestep.time_state_cache;

            let ctx = Ctx {
                world: &mut self.world,
                input: &mut self.input,
                time: time_state,
                renderer,
                gpu: _gpu,
                assets: &mut self.assets,
                physics: &mut *self.physics,
                entity_map: &mut self.entity_map,
                viewport: self.viewport,
            };

            let mut ui = esox_ui::Ui::begin(
                _frame, _gpu, _resources, text,
                &mut self.ui_state, &self.theme, vp,
            );

            self.game.ui(&mut ui, &ctx);

            ui.finish();

            if self.debug_overlay_visible {
                let text = self.text_renderer.as_mut().unwrap();
                let stats = EngineStats {
                    physics_step_us: self.last_physics_us,
                    batch_stats: self.last_batch_stats,
                    entity_count: self.world.len() as usize,
                };
                draw_debug_overlay(
                    _frame, _gpu, _resources, text,
                    &stats, _perf, self.viewport,
                );
            }
        }
    }

    fn on_key(
        &mut self,
        event: &winit::event::KeyEvent,
        _modifiers: winit::keyboard::ModifiersState,
    ) {
        #[cfg(feature = "ui")]
        {
            use winit::keyboard::{KeyCode, PhysicalKey};
            if event.state.is_pressed()
                && event.physical_key == PhysicalKey::Code(KeyCode::F3)
            {
                self.debug_overlay_visible = !self.debug_overlay_visible;
            }
        }
        self.input.handle_key_event(event);
        #[cfg(feature = "ui")]
        self.ui_state.process_key(event.clone(), _modifiers);
    }

    fn on_resize(&mut self, width: u32, height: u32, _gpu: &GpuContext) {
        self.viewport = (width, height);
    }

    fn on_mouse(&mut self, event: MouseInputEvent) {
        match event {
            MouseInputEvent::Moved { x, y } => {
                self.input.handle_mouse_move(x, y);
                #[cfg(feature = "ui")]
                self.ui_state.process_mouse_move(
                    x as f32,
                    y as f32,
                    self.theme.item_height,
                    self.theme.dropdown_gap,
                );
            }
            MouseInputEvent::Press {
                button,
                x: _x,
                y: _y,
            } => {
                self.input.handle_mouse_button(button, true);
                #[cfg(feature = "ui")]
                if button == 0 {
                    self.ui_state.process_mouse_click(_x as f32, _y as f32);
                } else if button == 2 {
                    self.ui_state.process_right_click(_x as f32, _y as f32);
                }
            }
            MouseInputEvent::Release { button, .. } => {
                self.input.handle_mouse_button(button, false);
                #[cfg(feature = "ui")]
                self.ui_state.process_mouse_release();
            }
            MouseInputEvent::Scroll {
                x: _x,
                y: _y,
                delta_y: _delta_y,
            } => {
                #[cfg(feature = "ui")]
                self.ui_state.process_scroll(_x as f32, _y as f32, _delta_y);
            }
            MouseInputEvent::Left => {}
        }
    }

    fn on_focus_changed(&mut self, focused: bool) {
        if !focused {
            self.input.clear_all_state();
        }
    }

    fn on_scale_changed(&mut self, _scale_factor: f64, _gpu: &GpuContext) {}

    fn on_paste(&mut self, _text: &str) {
        #[cfg(feature = "ui")]
        self.ui_state.on_ime_commit(_text.to_string());
    }

    fn on_ime_commit(&mut self, _text: &str) {
        #[cfg(feature = "ui")]
        self.ui_state.on_ime_commit(_text.to_string());
    }

    fn on_ime_preedit(&mut self, _text: String, _cursor: Option<(usize, usize)>) {
        #[cfg(feature = "ui")]
        self.ui_state.on_ime_preedit(_text, _cursor);
    }

    fn on_ime_enabled(&mut self, _enabled: bool) {
        #[cfg(feature = "ui")]
        self.ui_state.on_ime_enabled(_enabled);
    }

    fn on_copy(&mut self) -> Option<String> {
        None
    }

    fn needs_continuous_redraw(&self) -> bool {
        true
    }

    fn should_exit(&self) -> bool {
        self.game.should_exit()
    }
}
