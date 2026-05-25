//! Minimal glTF/GLB viewer: loads one model from a path argument and shows it
//! on an auto-orbiting camera. Built to eyeball morphic's Deadlock skin exports.
//!
//! Usage: `cargo run -p glb_viewer -- [path.glb]`  (default: /tmp/hornet.glb)
//! Esc quits. Up/Down cycle animations if the model has any (skins with no
//! embedded clips render at bind pose).

use std::path::PathBuf;

use esox_gfx::mesh3d::{
    AnimationClip, AnimationPlayer, Camera, DirectionalLight, GltfScene, GltfSceneHandles,
    InstanceData, LightEnvironment, MaterialDescriptor, MaterialHandle, MaterialType, MeshData,
    MeshHandle, PostProcess3DConfig, Renderer3D, Scene3D, ShadowConfig, Transform,
};
use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::config::{PlatformConfig, WindowConfig};
use esox_platform::{AppDelegate, MouseInputEvent};

fn model_arg() -> PathBuf {
    std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/hornet.glb"))
}

struct Loaded {
    scene: Scene3D,
    handles: GltfSceneHandles,
    player: Option<AnimationPlayer>,
    clips: Vec<AnimationClip>,
    current: usize,
}

struct App {
    path: PathBuf,
    renderer: Option<Renderer3D>,
    loaded: Option<Loaded>,
    ground_mesh: Option<MeshHandle>,
    ground_material: Option<MaterialHandle>,
    camera: Camera,
    start: std::time::Instant,
    last_frame: std::time::Instant,
    viewport: (u32, u32),
    loaded_once: bool,
}

impl App {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            renderer: None,
            loaded: None,
            ground_mesh: None,
            ground_material: None,
            // Framed for a ~2.3m humanoid standing on the ground plane.
            camera: Camera {
                position: glam::Vec3::new(3.5, 1.5, 3.5),
                target: glam::Vec3::new(0.0, 1.1, 0.0),
                ..Camera::default()
            },
            start: std::time::Instant::now(),
            last_frame: std::time::Instant::now(),
            viewport: (1280, 720),
            loaded_once: false,
        }
    }

    fn load(&mut self, gpu: &GpuContext) {
        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return,
        };

        let gltf_scene = match GltfScene::load(&self.path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[glb_viewer] failed to load {}: {e}", self.path.display());
                return;
            }
        };

        let has_skins = !gltf_scene.skins.is_empty();
        let has_anims = !gltf_scene.animations.is_empty();
        let mut handles = renderer.upload_gltf_scene(gpu, gltf_scene);
        let scene = Scene3D::from_gltf(&handles);

        // Build a player for any skin so a clip-less model still poses at bind
        // (advance with no current clip yields the bind-pose skinning matrices).
        let mut player = None;
        if has_skins && !handles.skins.is_empty() {
            let mut p = AnimationPlayer::new(&handles.skins[0]);
            if has_anims && !handles.animations.is_empty() {
                p.play(0, true);
            }
            player = Some(p);
        }

        let clips = std::mem::take(&mut handles.animations);
        eprintln!(
            "[glb_viewer] loaded {} (skin: {}, {} animation(s))",
            self.path.display(),
            has_skins,
            clips.len(),
        );

        self.loaded = Some(Loaded {
            scene,
            handles,
            player,
            clips,
            current: 0,
        });
    }

    fn switch_anim(&mut self, delta: i32) {
        let Some(loaded) = self.loaded.as_mut() else {
            return;
        };
        if loaded.clips.is_empty() {
            return;
        }
        let len = loaded.clips.len() as i32;
        loaded.current = (loaded.current as i32 + delta).rem_euclid(len) as usize;
        if let Some(p) = loaded.player.as_mut() {
            p.play(loaded.current, true);
        }
        eprintln!(
            "[glb_viewer] anim {}/{}",
            loaded.current + 1,
            loaded.clips.len()
        );
    }
}

impl AppDelegate for App {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        let mut renderer = Renderer3D::new(gpu);

        renderer.enable_postprocess(gpu);
        renderer.set_postprocess(PostProcess3DConfig {
            bloom_enabled: true,
            bloom_intensity: 0.15,
            bloom_threshold: 2.0,
            bloom_soft_knee: 0.5,
            tone_map_enabled: true,
            ssao_enabled: true,
            fog_enabled: false,
            fog_color: [0.75, 0.82, 0.90],
            fog_start: 50.0,
            fog_end: 200.0,
        });
        renderer.enable_shadows(gpu);
        renderer.set_shadow_config(ShadowConfig {
            shadow_distance: 15.0,
            ..ShadowConfig::default()
        });
        renderer.enable_ssao(gpu);

        renderer.set_lights(&LightEnvironment {
            ambient_color: [0.16, 0.16, 0.2],
            ambient_intensity: 1.0,
            directional: DirectionalLight {
                direction: [-0.4, -1.0, -0.3],
                color: [1.0, 0.96, 0.88],
                intensity: 2.5,
            },
            point_lights: vec![],
            spot_lights: vec![],
        });
        renderer.generate_procedural_ibl(gpu);

        let ground = MeshData::plane(20.0, 20.0, 1);
        self.ground_mesh = Some(renderer.upload_mesh(gpu, &ground));
        self.ground_material = Some(renderer.create_material(
            gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [0.35, 0.35, 0.32, 1.0],
                roughness: 0.85,
                metallic: 0.0,
                ..MaterialDescriptor::default()
            },
        ));

        self.renderer = Some(renderer);
    }

    fn on_redraw(
        &mut self,
        _gpu: &GpuContext,
        _resources: &mut RenderResources,
        _frame: &mut Frame,
        _perf: &esox_platform::perf::PerfMonitor,
    ) {
    }

    fn on_pre_render(
        &mut self,
        gpu: &GpuContext,
        surface_view: &wgpu::TextureView,
    ) -> Vec<wgpu::CommandBuffer> {
        if !self.loaded_once {
            self.loaded_once = true;
            self.load(gpu);
        }

        let now = std::time::Instant::now();
        let elapsed = self.start.elapsed().as_secs_f32();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return vec![],
        };

        let mut cmd_bufs = Vec::new();

        if let Some(loaded) = &mut self.loaded {
            if let Some(player) = loaded.player.as_mut() {
                player.advance(dt, &loaded.clips);
                for &si in loaded.handles.skinned_mesh_indices.iter().flatten() {
                    renderer.update_joints(gpu, si, player.skinning_matrices());
                }
            }
        }
        if let Some(skin_cmd) = renderer.dispatch_skinning(gpu) {
            cmd_bufs.push(skin_cmd);
        }

        if let Some(loaded) = &self.loaded {
            loaded.scene.draw(renderer);
        }
        if let (Some(gm), Some(gmat)) = (self.ground_mesh, self.ground_material) {
            let ground = Transform {
                position: glam::Vec3::new(0.0, -0.02, 0.0),
                ..Transform::IDENTITY
            };
            renderer.draw_with_material(gm, gmat, &[InstanceData::from_transform(&ground)]);
        }

        // Slow auto-orbit so all sides are visible.
        let radius = 3.5;
        let speed = 0.3;
        self.camera.position = glam::Vec3::new(
            radius * (elapsed * speed).cos(),
            1.5,
            radius * (elapsed * speed).sin(),
        );
        self.camera.target = glam::Vec3::new(0.0, 1.1, 0.0);

        let (render_cmd, _stats) = renderer.encode(
            gpu,
            surface_view,
            &self.camera,
            self.viewport.0,
            self.viewport.1,
            elapsed,
            dt,
            wgpu::Color {
                r: 0.08,
                g: 0.08,
                b: 0.1,
                a: 1.0,
            },
        );
        cmd_bufs.push(render_cmd);
        cmd_bufs
    }

    fn on_key(
        &mut self,
        event: &esox_platform::esox_input::KeyEvent,
        _modifiers: esox_platform::esox_input::Modifiers,
    ) {
        use esox_platform::esox_input::{Key, NamedKey};
        if !event.pressed {
            return;
        }
        match &event.key {
            Key::Named(NamedKey::Escape) => std::process::exit(0),
            Key::Named(NamedKey::ArrowUp) => self.switch_anim(1),
            Key::Named(NamedKey::ArrowDown) => self.switch_anim(-1),
            _ => {}
        }
    }

    fn on_resize(&mut self, width: u32, height: u32, _gpu: &GpuContext) {
        self.viewport = (width, height);
    }

    fn on_scale_changed(&mut self, _scale_factor: f64, _gpu: &GpuContext) {}
    fn on_mouse(&mut self, _event: MouseInputEvent) {}
    fn on_paste(&mut self, _text: &str) {}
    fn on_ime_commit(&mut self, _text: &str) {}
    fn on_copy(&mut self) -> Option<String> {
        None
    }
    fn needs_continuous_redraw(&self) -> bool {
        true
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("glb_viewer=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap()),
        )
        .init();

    let path = model_arg();
    let config = PlatformConfig {
        window: WindowConfig {
            title: format!("glb_viewer — {}", path.display()),
            width: Some(1280),
            height: Some(720),
            ..WindowConfig::default()
        },
        ..PlatformConfig::default()
    };

    if let Err(e) = esox_platform::run(config, Box::new(App::new(path))) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
