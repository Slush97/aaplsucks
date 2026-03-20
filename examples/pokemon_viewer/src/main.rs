//! Pokemon Animation Viewer — Dragonite & Mewtwo.
//!
//! Left/Right: switch Pokemon. Up/Down: cycle animations. Escape: exit.

use std::path::{Path, PathBuf};

use esox_gfx::mesh3d::{
    AnimationClip, AnimationPlayer, Camera, DirectionalLight, GltfScene, GltfSceneHandles,
    InstanceData, LightEnvironment, MaterialDescriptor, MaterialType, MeshData, MeshHandle,
    MaterialHandle, PostProcess3DConfig, Renderer3D, Scene3D, ShadowConfig, Transform,
};
use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::config::{PlatformConfig, WindowConfig};
use esox_platform::{AppDelegate, MouseInputEvent};

const ROSTER: &[(usize, &str)] = &[
    (1, "Bulbasaur"),
    (4, "Charmander"),
    (7, "Squirtle"),
    (25, "Pikachu"),
    (149, "Dragonite"),
    (150, "Mewtwo"),
];

fn models_dir() -> PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));

    let candidates = [
        PathBuf::from("examples/pokemon/assets/models"),
        PathBuf::from("../pokemon/assets/models"),
    ];

    if let Some(exe) = &exe_dir {
        let p = exe.join("../../../examples/pokemon/assets/models");
        if p.is_dir() {
            return p;
        }
    }

    for c in &candidates {
        if c.is_dir() {
            return c.clone();
        }
    }

    candidates[0].clone()
}

fn model_path(base: &Path, dex: usize) -> Option<PathBuf> {
    let gltf_dir = base.join(format!("{:03}/glTF", dex));
    let gltf = gltf_dir.join("model.gltf");
    if gltf.exists() {
        return Some(gltf);
    }
    let glb = gltf_dir.join("model.glb");
    if glb.exists() {
        return Some(glb);
    }
    None
}

/// Extract a human-readable name from the raw animation clip name.
/// e.g. "pm0150_00_00_00400_attack01" -> "attack01"
fn pretty_anim_name(raw: &str) -> &str {
    // Names are like "pm0150_00_00_00400_attack01" — grab after last numeric segment.
    // Find the last '_' followed by an alpha char.
    let bytes = raw.as_bytes();
    let mut last_label_start = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'_' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_alphabetic() {
            last_label_start = i + 1;
        }
        i += 1;
    }
    if last_label_start > 0 {
        &raw[last_label_start..]
    } else {
        raw
    }
}

struct LoadedModel {
    scene: Scene3D,
    handles: GltfSceneHandles,
    anim_player: Option<AnimationPlayer>,
    anim_clips: Vec<AnimationClip>,
    anim_names: Vec<String>,
    current_anim: usize,
}

struct App {
    renderer: Option<Renderer3D>,
    models_base: PathBuf,
    roster_index: usize,
    loaded: Option<LoadedModel>,
    ground_mesh: Option<MeshHandle>,
    ground_material: Option<MaterialHandle>,
    camera: Camera,
    start: std::time::Instant,
    last_frame: std::time::Instant,
    viewport: (u32, u32),
    pending_load: Option<usize>,
    pending_title: Option<String>,
}

impl App {
    fn new() -> Self {
        Self {
            renderer: None,
            models_base: models_dir(),
            roster_index: 0,
            loaded: None,
            ground_mesh: None,
            ground_material: None,
            camera: Camera {
                position: glam::Vec3::new(5.0, 2.5, 5.0),
                target: glam::Vec3::new(0.0, 1.0, 0.0),
                ..Camera::default()
            },
            start: std::time::Instant::now(),
            last_frame: std::time::Instant::now(),
            viewport: (1280, 720),
            pending_load: Some(0),
            pending_title: None,
        }
    }

    fn load_current(&mut self, gpu: &GpuContext) {
        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return,
        };

        self.loaded = None;
        let (dex, name) = ROSTER[self.roster_index];

        let path = match model_path(&self.models_base, dex) {
            Some(p) => p,
            None => {
                eprintln!("[viewer] no model for #{:03} {}", dex, name);
                return;
            }
        };

        match GltfScene::load(&path) {
            Ok(gltf_scene) => {
                let has_anims = !gltf_scene.animations.is_empty();
                let has_skins = !gltf_scene.skins.is_empty();

                let mut handles = renderer.upload_gltf_scene(gpu, gltf_scene);
                let scene = Scene3D::from_gltf(&handles);

                let mut anim_player = None;
                if has_anims && has_skins && !handles.skins.is_empty() {
                    let mut player = AnimationPlayer::new(&handles.skins[0]);
                    if !handles.animations.is_empty() {
                        player.play(0, true);
                    }
                    anim_player = Some(player);
                }

                let anim_clips = std::mem::take(&mut handles.animations);
                let anim_names: Vec<String> = anim_clips
                    .iter()
                    .enumerate()
                    .map(|(i, c)| match &c.name {
                        Some(name) => pretty_anim_name(name).to_string(),
                        None => format!("anim_{i}"),
                    })
                    .collect();

                let num_anims = anim_clips.len();
                self.loaded = Some(LoadedModel {
                    scene,
                    handles,
                    anim_player,
                    anim_clips,
                    anim_names,
                    current_anim: 0,
                });

                eprintln!("[viewer] loaded #{:03} {} ({} animations)", dex, name, num_anims);
                self.update_title();
            }
            Err(e) => {
                eprintln!("[viewer] failed to load #{:03} {}: {e}", dex, name);
            }
        }
    }

    fn update_title(&mut self) {
        let (dex, name) = ROSTER[self.roster_index];
        let anim_info = if let Some(loaded) = &self.loaded {
            if loaded.anim_clips.is_empty() {
                "no animations".to_string()
            } else {
                let anim_name = &loaded.anim_names[loaded.current_anim];
                format!(
                    "anim {}/{}: {}",
                    loaded.current_anim + 1,
                    loaded.anim_clips.len(),
                    anim_name,
                )
            }
        } else {
            "loading...".to_string()
        };

        self.pending_title = Some(format!(
            "#{:03} {} — {}  |  [Left/Right] pokemon  [Up/Down] anim",
            dex, name, anim_info,
        ));
    }

    fn switch_pokemon(&mut self, delta: i32) {
        let len = ROSTER.len() as i32;
        self.roster_index = ((self.roster_index as i32 + delta).rem_euclid(len)) as usize;
        self.pending_load = Some(self.roster_index);
    }

    fn switch_anim(&mut self, delta: i32) {
        let loaded = match self.loaded.as_mut() {
            Some(l) => l,
            None => return,
        };
        if loaded.anim_clips.is_empty() {
            return;
        }

        let len = loaded.anim_clips.len() as i32;
        let next = (loaded.current_anim as i32 + delta).rem_euclid(len) as usize;
        loaded.current_anim = next;

        if let Some(player) = loaded.anim_player.as_mut() {
            player.play(next, true);
        }

        let (dex, name) = ROSTER[self.roster_index];
        let anim_name = &loaded.anim_names[next];
        eprintln!(
            "[viewer] #{:03} {} — anim {}/{}: {}",
            dex, name, next + 1, loaded.anim_clips.len(), anim_name,
        );
        self.update_title();
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

        let lights = LightEnvironment {
            ambient_color: [0.15, 0.15, 0.18],
            ambient_intensity: 1.0,
            directional: DirectionalLight {
                direction: [-0.4, -1.0, -0.3],
                color: [1.0, 0.96, 0.88],
                intensity: 2.5,
            },
            point_lights: vec![],
            spot_lights: vec![],
        };
        renderer.set_lights(&lights);
        renderer.generate_procedural_ibl(gpu);

        let ground = MeshData::plane(20.0, 20.0, 1);
        let ground_mesh = renderer.upload_mesh(gpu, &ground);
        let ground_mat = renderer.create_material(
            gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [0.35, 0.35, 0.32, 1.0],
                roughness: 0.85,
                metallic: 0.0,
                ..MaterialDescriptor::default()
            },
        );
        self.ground_mesh = Some(ground_mesh);
        self.ground_material = Some(ground_mat);

        self.renderer = Some(renderer);
        eprintln!("[viewer] models dir: {}", self.models_base.display());
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
        if let Some(_idx) = self.pending_load.take() {
            self.load_current(gpu);
        }

        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return vec![],
        };

        let now = std::time::Instant::now();
        let elapsed = self.start.elapsed().as_secs_f32();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        let mut cmd_bufs = Vec::new();

        // Advance animation.
        if let Some(loaded) = &mut self.loaded {
            if let Some(player) = loaded.anim_player.as_mut() {
                player.advance(dt, &loaded.anim_clips);
                for &si in loaded.handles.skinned_mesh_indices.iter().flatten() {
                    renderer.update_joints(gpu, si, player.skinning_matrices());
                }
            }
        }

        if let Some(skin_cmd) = renderer.dispatch_skinning(gpu) {
            cmd_bufs.push(skin_cmd);
        }

        // Draw model.
        if let Some(loaded) = &self.loaded {
            loaded.scene.draw(renderer);
        }

        // Draw ground.
        if let (Some(gm), Some(gmat)) = (self.ground_mesh, self.ground_material) {
            let ground_transform = Transform {
                position: glam::Vec3::new(0.0, -0.05, 0.0),
                ..Transform::IDENTITY
            };
            let ground_instance = InstanceData::from_transform(&ground_transform);
            renderer.draw_with_material(gm, gmat, &[ground_instance]);
        }

        // Orbit camera.
        let radius = 5.0;
        let orbit_speed = 0.3;
        self.camera.position = glam::Vec3::new(
            radius * (elapsed * orbit_speed).cos(),
            2.5,
            radius * (elapsed * orbit_speed).sin(),
        );
        self.camera.target = glam::Vec3::new(0.0, 1.0, 0.0);

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
                b: 0.10,
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
            Key::Named(NamedKey::ArrowRight) => self.switch_pokemon(1),
            Key::Named(NamedKey::ArrowLeft) => self.switch_pokemon(-1),
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
    fn on_copy(&mut self) -> Option<String> { None }
    fn needs_continuous_redraw(&self) -> bool { true }

    fn take_title(&mut self) -> Option<String> {
        self.pending_title.take()
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("pokemon_viewer=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let (dex, name) = ROSTER[0];
    let config = PlatformConfig {
        window: WindowConfig {
            title: format!("#{:03} {} — Pokemon Animation Viewer", dex, name),
            width: Some(1280),
            height: Some(720),
            ..WindowConfig::default()
        },
        ..PlatformConfig::default()
    };

    if let Err(e) = esox_platform::run(config, Box::new(App::new())) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
