//! Deadlock model viewer with direct VPK input and hot reload.
//!
//! Wraps vpkmerge's `.vmdl_c` -> GLB exporter and esox's GLB renderer into one
//! iteration loop: point it at a skin VPK (or a plain .glb), it exports and
//! displays the model with live skeletal animation, then re-exports and
//! reloads whenever the watched file changes on disk (e.g. a reskin builder
//! re-bakes the addon).
//!
//! Usage:
//!   vpk_viewer model.glb
//!   vpk_viewer --vpk pak01_dir.vpk --hero hornet
//!   vpk_viewer --vpk addon_dir.vpk --hero hornet --base pak01_dir.vpk
//!   vpk_viewer --vpk pak01_dir.vpk --hero yamato --clip primary_stand_idle
//!
//! Keys: Up/Down cycle clips, Space pause/resume, R force reload, Esc quit.
//!
//! Each reload clears the previous model's GPU resources back to a
//! post-ground-plane [`SceneCheckpoint`] before uploading the new one, so
//! long watch sessions don't accumulate VRAM or stale skinning dispatches.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use esox_gfx::mesh3d::{
    AnimationClip, AnimationPlayer, Camera, DirectionalLight, GltfScene, GltfSceneHandles,
    InstanceData, LightEnvironment, MaterialDescriptor, MaterialHandle, MaterialType, MeshData,
    MeshHandle, PostProcess3DConfig, Renderer3D, Scene3D, SceneCheckpoint, ShadowConfig, Transform,
};
use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::config::{PlatformConfig, WindowConfig};
use esox_platform::{AppDelegate, MouseInputEvent};
use notify::Watcher;
use vpkmerge_core::{AnimOptions, PoseSelection};

const RELOAD_DEBOUNCE: Duration = Duration::from_millis(500);

#[derive(Parser)]
#[command(about = "View a Deadlock model straight from a VPK (or a .glb), with hot reload")]
struct Args {
    /// A .glb to view directly (alternative to --vpk).
    glb: Option<PathBuf>,
    /// Skin or base VPK containing the model.
    #[arg(long, conflicts_with = "glb")]
    vpk: Option<PathBuf>,
    /// Hero codename to auto-discover (e.g. hornet).
    #[arg(long, requires = "vpk", conflicts_with = "entry")]
    hero: Option<String>,
    /// Explicit .vmdl_c entry path inside the VPK.
    #[arg(long, requires = "vpk")]
    entry: Option<String>,
    /// Base pak01_dir.vpk for materials/textures the skin doesn't ship.
    #[arg(long, requires = "vpk")]
    base: Option<PathBuf>,
    /// Keep only these clips (repeatable). Trimming makes re-exports much faster.
    #[arg(long, conflicts_with = "no_anim")]
    clip: Vec<String>,
    /// Strip all animation clips (static mesh + skeleton).
    #[arg(long)]
    no_anim: bool,
    /// Bake a static single-frame pose: bare for menu/idle, or CLIP[@FRAME].
    #[arg(long, num_args = 0..=1, default_missing_value = "", conflicts_with_all = ["clip", "no_anim"])]
    pose: Option<String>,
    /// Disable the file watcher (load once, like glb_viewer).
    #[arg(long)]
    no_watch: bool,
}

enum Target {
    Hero(String),
    Entry(String),
}

enum Source {
    Glb(PathBuf),
    Vpk {
        vpk: PathBuf,
        base: Option<PathBuf>,
        target: Target,
        anim: AnimOptions,
    },
}

impl Source {
    /// The on-disk file whose changes should trigger a reload.
    fn watched_file(&self) -> &Path {
        match self {
            Source::Glb(p) => p,
            Source::Vpk { vpk, .. } => vpk,
        }
    }
}

fn parse_pose(spec: &str) -> PoseSelection {
    if spec.is_empty() {
        return PoseSelection::default();
    }
    let (clip, frame) = match spec.split_once('@') {
        Some((c, f)) => (c, f.parse().unwrap_or(0)),
        None => (spec, 0),
    };
    PoseSelection {
        clips: vec![clip.to_string()],
        frame,
        require: false,
    }
}

fn source_from_args(args: &Args) -> Result<Source, String> {
    if let Some(glb) = &args.glb {
        return Ok(Source::Glb(glb.clone()));
    }
    let Some(vpk) = &args.vpk else {
        return Err("pass a .glb path or --vpk (see --help)".into());
    };
    let target = match (&args.hero, &args.entry) {
        (Some(h), None) => Target::Hero(h.clone()),
        (None, Some(e)) => Target::Entry(e.clone()),
        _ => return Err("--vpk needs exactly one of --hero or --entry".into()),
    };
    Ok(Source::Vpk {
        vpk: vpk.clone(),
        base: args.base.clone(),
        target,
        anim: AnimOptions {
            no_anim: args.no_anim,
            clips: args.clip.clone(),
            pose: args.pose.as_deref().map(parse_pose),
        },
    })
}

/// Export (VPK mode) + parse the GLB. Runs on a worker thread so the window
/// stays live during multi-second hero exports.
fn produce_scene(source: &Source, tmp_glb: &Path) -> anyhow::Result<GltfScene> {
    let glb_path = match source {
        Source::Glb(p) => p.clone(),
        Source::Vpk {
            vpk,
            base,
            target,
            anim,
        } => {
            let t0 = Instant::now();
            match target {
                Target::Hero(h) => {
                    vpkmerge_core::export_hero_model(vpk, h, base.as_deref(), anim, tmp_glb)?;
                }
                Target::Entry(e) => {
                    vpkmerge_core::export_model(vpk, e, base.as_deref(), anim, tmp_glb)?;
                }
            }
            eprintln!("[vpk_viewer] exported in {:.1}s", t0.elapsed().as_secs_f32());
            tmp_glb.to_path_buf()
        }
    };
    let t0 = Instant::now();
    let scene = GltfScene::load(&glb_path)?;
    eprintln!("[vpk_viewer] parsed GLB in {:.1}s", t0.elapsed().as_secs_f32());
    Ok(scene)
}

struct Loaded {
    scene: Scene3D,
    handles: GltfSceneHandles,
    player: Option<AnimationPlayer>,
    clips: Vec<AnimationClip>,
    current: usize,
}

struct App {
    source: Arc<Source>,
    tmp_glb: PathBuf,
    tx: mpsc::Sender<anyhow::Result<GltfScene>>,
    rx: mpsc::Receiver<anyhow::Result<GltfScene>>,
    in_flight: bool,
    dirty: Arc<Mutex<Option<Instant>>>,
    _watcher: Option<notify::RecommendedWatcher>,
    renderer: Option<Renderer3D>,
    loaded: Option<Loaded>,
    checkpoint: Option<SceneCheckpoint>,
    ground_mesh: Option<MeshHandle>,
    ground_material: Option<MaterialHandle>,
    camera: Camera,
    start: Instant,
    last_frame: Instant,
    viewport: (u32, u32),
    paused: bool,
    pending_title: Option<String>,
}

impl App {
    fn new(source: Source, watch: bool) -> Self {
        let (tx, rx) = mpsc::channel();
        let dirty: Arc<Mutex<Option<Instant>>> = Arc::new(Mutex::new(None));

        let watcher = if watch {
            match spawn_watcher(source.watched_file(), Arc::clone(&dirty)) {
                Ok(w) => Some(w),
                Err(e) => {
                    eprintln!("[vpk_viewer] file watcher disabled: {e}");
                    None
                }
            }
        } else {
            None
        };

        let mut app = Self {
            source: Arc::new(source),
            tmp_glb: std::env::temp_dir().join(format!("vpk_viewer_{}.glb", std::process::id())),
            tx,
            rx,
            in_flight: false,
            dirty,
            _watcher: watcher,
            renderer: None,
            loaded: None,
            checkpoint: None,
            ground_mesh: None,
            ground_material: None,
            // Framed for a ~2.3m humanoid standing on the ground plane.
            camera: Camera {
                position: glam::Vec3::new(3.5, 1.5, 3.5),
                target: glam::Vec3::new(0.0, 1.1, 0.0),
                ..Camera::default()
            },
            start: Instant::now(),
            last_frame: Instant::now(),
            viewport: (1280, 720),
            paused: false,
            pending_title: None,
        };
        app.spawn_load();
        app
    }

    fn spawn_load(&mut self) {
        if self.in_flight {
            return;
        }
        self.in_flight = true;
        let source = Arc::clone(&self.source);
        let tmp = self.tmp_glb.clone();
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(produce_scene(&source, &tmp));
        });
    }

    fn install_scene(&mut self, gpu: &GpuContext, gltf_scene: GltfScene) {
        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return,
        };

        // Resume the clip the previous load was playing, matched by name so
        // reordered exports don't jump to an unrelated animation.
        let prev_name = self.loaded.as_ref().and_then(|l| {
            l.clips
                .get(l.current)
                .and_then(|c| c.name.clone())
        });

        // Drop the old model's handles, then release its GPU resources so
        // reloads don't accumulate; the upload below reuses the freed space.
        self.loaded = None;
        if let Some(cp) = &self.checkpoint {
            renderer.clear_scene(cp);
        }

        let has_skins = !gltf_scene.skins.is_empty();
        let mut handles = renderer.upload_gltf_scene(gpu, gltf_scene);
        let scene = Scene3D::from_gltf(&handles);
        let clips = std::mem::take(&mut handles.animations);
        let current = prev_name
            .and_then(|n| clips.iter().position(|c| c.name.as_deref() == Some(&n)))
            .unwrap_or(0);

        let mut player = None;
        if has_skins && !handles.skins.is_empty() {
            let mut p = AnimationPlayer::new(&handles.skins[0]);
            if !clips.is_empty() {
                p.play(current, true);
            }
            p.speed = if self.paused { 0.0 } else { 1.0 };
            player = Some(p);
        }

        eprintln!(
            "[vpk_viewer] loaded (skin: {has_skins}, {} clip(s))",
            clips.len()
        );
        self.loaded = Some(Loaded {
            scene,
            handles,
            player,
            clips,
            current,
        });
        self.refresh_title();
    }

    fn refresh_title(&mut self) {
        let what = match self.source.as_ref() {
            Source::Glb(p) => p.display().to_string(),
            Source::Vpk { target, .. } => match target {
                Target::Hero(h) => h.clone(),
                Target::Entry(e) => e.clone(),
            },
        };
        let anim = match &self.loaded {
            Some(l) if !l.clips.is_empty() => {
                let name = l.clips[l.current].name.as_deref().unwrap_or("?");
                format!(" [{}/{} {name}]", l.current + 1, l.clips.len())
            }
            _ => String::new(),
        };
        let paused = if self.paused { " (paused)" } else { "" };
        self.pending_title = Some(format!("vpk_viewer — {what}{anim}{paused}"));
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
            p.speed = if self.paused { 0.0 } else { 1.0 };
        }
        let name = loaded.clips[loaded.current].name.as_deref().unwrap_or("?");
        eprintln!(
            "[vpk_viewer] anim {}/{} {name}",
            loaded.current + 1,
            loaded.clips.len()
        );
        self.refresh_title();
    }
}

/// Watches the parent directory of `file` (non-recursive) and flags `dirty` on
/// any event touching the file's chunk family: `x_dir.vpk` also matches
/// `x_000.vpk` etc., and atomic rename-replace still lands (the file's own
/// inode would be lost by watching the path directly).
fn spawn_watcher(
    file: &Path,
    dirty: Arc<Mutex<Option<Instant>>>,
) -> anyhow::Result<notify::RecommendedWatcher> {
    let dir = file
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let prefix = stem.strip_suffix("dir").unwrap_or(stem).to_string();
    let ext = file.extension().map(|e| e.to_os_string());

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        let Ok(event) = res else { return };
        // Content changes only. Access events must not count: the export reads
        // the watched VPK, so reacting to reads would loop reload -> read ->
        // reload forever. Metadata-only events (atime) are skipped for the
        // same reason; use the R key to force a reload by hand.
        use notify::event::ModifyKind;
        let content_change = matches!(
            event.kind,
            notify::EventKind::Create(_)
                | notify::EventKind::Modify(
                    ModifyKind::Data(_) | ModifyKind::Name(_) | ModifyKind::Any
                )
        );
        if !content_change {
            return;
        }
        // Same extension AND same stem family. The extension check matters:
        // prefix alone also caught unrelated files in the watched dir (e.g. a
        // `live_demo.log` next to `live_dir.vpk` that the viewer's own output
        // was piped into, which retriggered the watcher on every log line).
        let relevant = event.paths.iter().any(|p| {
            p.extension().map(|e| e.to_os_string()) == ext
                && p.file_stem()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(&prefix))
        });
        if relevant {
            *dirty.lock().unwrap() = Some(Instant::now());
        }
    })?;
    watcher.watch(&dir, notify::RecursiveMode::NonRecursive)?;
    eprintln!(
        "[vpk_viewer] watching {} for changes to {}*",
        dir.display(),
        file.file_stem().and_then(|s| s.to_str()).unwrap_or("?")
    );
    Ok(watcher)
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

        // Everything above survives reloads; each model upload is cleared
        // back to this point before the next one lands.
        self.checkpoint = Some(renderer.scene_checkpoint());

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
        // Finished background loads land here (upload needs &mut renderer).
        while let Ok(msg) = self.rx.try_recv() {
            self.in_flight = false;
            match msg {
                Ok(scene) => self.install_scene(gpu, scene),
                Err(e) => eprintln!("[vpk_viewer] load failed: {e:#}"),
            }
        }

        // Debounced hot reload. If a load is already running the flag stays
        // set, so the change is picked up as soon as the current load lands.
        let due = self
            .dirty
            .lock()
            .unwrap()
            .is_some_and(|t| t.elapsed() >= RELOAD_DEBOUNCE);
        if due && !self.in_flight {
            *self.dirty.lock().unwrap() = None;
            eprintln!("[vpk_viewer] change detected, reloading");
            self.spawn_load();
        }

        let now = Instant::now();
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
            Key::Named(NamedKey::Space) => {
                self.paused = !self.paused;
                if let Some(p) = self.loaded.as_mut().and_then(|l| l.player.as_mut()) {
                    p.speed = if self.paused { 0.0 } else { 1.0 };
                }
                self.refresh_title();
            }
            Key::Character(c) if c == "r" => {
                eprintln!("[vpk_viewer] manual reload");
                self.spawn_load();
            }
            _ => {}
        }
    }

    fn on_resize(&mut self, width: u32, height: u32, _gpu: &GpuContext) {
        self.viewport = (width, height);
    }

    fn take_title(&mut self) -> Option<String> {
        self.pending_title.take()
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
                .add_directive("vpk_viewer=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();
    let source = match source_from_args(&args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    let config = PlatformConfig {
        window: WindowConfig {
            title: "vpk_viewer".to_string(),
            width: Some(1280),
            height: Some(720),
            ..WindowConfig::default()
        },
        ..PlatformConfig::default()
    };

    if let Err(e) = esox_platform::run(config, Box::new(App::new(source, !args.no_watch))) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
