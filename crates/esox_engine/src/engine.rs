//! Engine — implements AppDelegate internally, bridges to Game trait.

use esox_gfx::mesh3d::Renderer3D;
use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::perf::PerfMonitor;
use esox_platform::{AppDelegate, MouseInputEvent};

use crate::assets::AssetManager;
use crate::ecs::{
    camera_sync_system, hierarchy_system, light_collection_system, render_extraction_system,
};
use crate::game::Game;
use crate::input::InputManager;
use crate::physics::{NullPhysics, PhysicsBackend};
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
    viewport: (u32, u32),
    initialized: bool,
}

impl Engine {
    pub fn new(config: EngineConfig, game: Box<dyn Game>) -> Self {
        let tick_rate = config.tick_rate;
        Self {
            config,
            game,
            renderer: None,
            world: hecs::World::new(),
            input: InputManager::new(),
            timestep: FixedTimestep::new(tick_rate),
            assets: AssetManager::new(),
            physics: Box::new(NullPhysics),
            viewport: (1280, 720),
            initialized: false,
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
                tone_map_enabled: true,
                ssao_enabled: true,
                motion_blur_enabled: false,
            });
            renderer.enable_ssao(gpu);
        }

        if self.config.shadows {
            renderer.enable_shadows(gpu);
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

        // 1. Advance timestep.
        let (tick_count, alpha) = self.timestep.advance();
        self.timestep.time_state_cache = self.timestep.time_state(tick_count);

        // 2. Process completed asset uploads.
        self.assets.process_uploads(gpu, renderer);

        // 3. Fixed-rate update loop.
        for tick_i in 0..tick_count {
            self.input.pre_update();

            self.timestep.time_state_cache = self.timestep.time_state(tick_i + 1);
            let mut ctx = Ctx {
                world: &mut self.world,
                input: &mut self.input,
                time: &self.timestep.time_state_cache,
                renderer,
                gpu,
                assets: &mut self.assets,
                viewport: self.viewport,
            };
            self.game.update(&mut ctx);

            self.physics.step(self.timestep.tick_dt);

            self.input.post_update();
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

        // 8. Dispatch skinning compute (if any).
        let mut cmd_bufs = Vec::new();
        if let Some(skin_cmd) = renderer.dispatch_skinning(gpu) {
            cmd_bufs.push(skin_cmd);
        }

        // 9. Camera sync.
        let camera = camera_sync_system(&self.world).unwrap_or_default();

        // 10. Encode 3D render pass.
        let elapsed = self.timestep.time_state_cache.elapsed;
        let dt = self.timestep.time_state_cache.frame_dt;
        let (render_cmd, _stats) = renderer.encode(
            gpu,
            surface_view,
            &camera,
            self.viewport.0,
            self.viewport.1,
            elapsed,
            dt,
            self.config.clear_color,
        );
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
        // 2D overlay rendering via the `ui` feature would go here.
    }

    fn on_key(
        &mut self,
        event: &winit::event::KeyEvent,
        _modifiers: winit::keyboard::ModifiersState,
    ) {
        self.input.handle_key_event(event);
    }

    fn on_resize(&mut self, width: u32, height: u32, _gpu: &GpuContext) {
        self.viewport = (width, height);
    }

    fn on_mouse(&mut self, event: MouseInputEvent) {
        match event {
            MouseInputEvent::Moved { x, y } => self.input.handle_mouse_move(x, y),
            MouseInputEvent::Press { button, .. } => self.input.handle_mouse_button(button, true),
            MouseInputEvent::Release { button, .. } => {
                self.input.handle_mouse_button(button, false)
            }
            MouseInputEvent::Scroll { .. } | MouseInputEvent::Left => {}
        }
    }

    fn on_scale_changed(&mut self, _scale_factor: f64, _gpu: &GpuContext) {}
    fn on_paste(&mut self, _text: &str) {}
    fn on_ime_commit(&mut self, _text: &str) {}
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
