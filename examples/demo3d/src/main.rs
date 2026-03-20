use std::path::PathBuf;

use esox_gfx::mesh3d::{
    AnimationClip, AnimationPlayer, Camera, DirectionalLight, GltfScene, GltfSceneHandles,
    InstanceData, LightEnvironment, MaterialDescriptor, MaterialHandle, MaterialType, MeshData,
    MeshHandle, PointLight, PostProcess3DConfig, Renderer3D, Scene3D, ShadowConfig, SpotLight,
    Transform,
};
use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::config::{PlatformConfig, WindowConfig};
use esox_platform::{AppDelegate, MouseInputEvent};

struct Demo3dApp {
    renderer: Option<Renderer3D>,
    // Fallback: spinning cube + ground plane (when no glTF file is provided).
    cube_mesh: Option<MeshHandle>,
    cube_material: Option<MaterialHandle>,
    ground_mesh: Option<MeshHandle>,
    ground_material: Option<MaterialHandle>,
    // glTF scene.
    scene: Option<Scene3D>,
    gltf_handles: Option<GltfSceneHandles>,
    anim_player: Option<AnimationPlayer>,
    anim_clips: Vec<AnimationClip>,
    // Shared state.
    camera: Camera,
    angle: f32,
    start: std::time::Instant,
    last_frame: std::time::Instant,
    viewport: (u32, u32),
    gltf_path: Option<PathBuf>,
}

impl Demo3dApp {
    fn new(gltf_path: Option<PathBuf>) -> Self {
        Self {
            renderer: None,
            cube_mesh: None,
            cube_material: None,
            ground_mesh: None,
            ground_material: None,
            scene: None,
            gltf_handles: None,
            anim_player: None,
            anim_clips: Vec::new(),
            camera: Camera {
                position: glam::Vec3::new(3.0, 2.5, 3.0),
                target: glam::Vec3::ZERO,
                ..Camera::default()
            },
            angle: 0.0,
            start: std::time::Instant::now(),
            last_frame: std::time::Instant::now(),
            viewport: (800, 600),
            gltf_path,
        }
    }
}

impl AppDelegate for Demo3dApp {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        let mut renderer = Renderer3D::new(gpu);

        // Enable post-processing: bloom + tone mapping.
        renderer.enable_postprocess(gpu);
        renderer.set_postprocess(PostProcess3DConfig {
            bloom_enabled: true,
            bloom_intensity: 0.3,
            bloom_threshold: 1.0,
            bloom_soft_knee: 0.5,
            tone_map_enabled: true,
            ssao_enabled: true,
            fog_enabled: false,
            fog_color: [0.75, 0.82, 0.90],
            fog_start: 50.0,
            fog_end: 200.0,
        });

        // Enable shadows.
        renderer.enable_shadows(gpu);
        renderer.set_shadow_config(ShadowConfig {
            shadow_distance: 15.0,
            ..ShadowConfig::default()
        });

        // Enable SSAO.
        renderer.enable_ssao(gpu);

        // Set up lighting with a directional light, a point light, and a spot light.
        let lights = LightEnvironment {
            ambient_color: [0.08, 0.08, 0.12],
            ambient_intensity: 1.0,
            directional: DirectionalLight {
                direction: [-0.5, -1.0, -0.3],
                color: [1.0, 0.95, 0.85],
                intensity: 2.0,
            },
            point_lights: vec![
                PointLight {
                    position: [2.0, 3.0, 2.0],
                    color: [0.4, 0.7, 1.0],
                    intensity: 8.0,
                    range: 12.0,
                    cast_shadows: false,
                },
            ],
            spot_lights: vec![
                SpotLight {
                    position: [-3.0, 4.0, 0.0],
                    direction: [0.3, -1.0, 0.0],
                    color: [1.0, 0.8, 0.3],
                    intensity: 15.0,
                    range: 15.0,
                    inner_cone_angle: 15.0_f32.to_radians(),
                    outer_cone_angle: 30.0_f32.to_radians(),
                    cast_shadows: false,
                },
            ],
        };
        renderer.set_lights(&lights);
        renderer.generate_procedural_ibl(gpu);

        // Try loading a glTF file if provided.
        if let Some(path) = &self.gltf_path {
            match GltfScene::load(path) {
                Ok(gltf_scene) => {
                    let has_animations = !gltf_scene.animations.is_empty();
                    let has_skins = !gltf_scene.skins.is_empty();

                    let mut handles = renderer.upload_gltf_scene(gpu, gltf_scene);
                    let scene = Scene3D::from_gltf(&handles);

                    // Set up animation if available.
                    if has_animations && has_skins && !handles.skins.is_empty() {
                        let mut player = AnimationPlayer::new(&handles.skins[0]);
                        if !handles.animations.is_empty() {
                            player.play(0, true);
                        }
                        self.anim_player = Some(player);
                    }

                    self.scene = Some(scene);
                    // Store animation clips separately for the player.
                    self.anim_clips = std::mem::take(&mut handles.animations);
                    self.gltf_handles = Some(handles);

                    // Pull camera back for larger scenes.
                    self.camera.position = glam::Vec3::new(5.0, 3.0, 5.0);

                    eprintln!("[demo3d] loaded glTF: {}", path.display());
                }
                Err(e) => {
                    eprintln!("[demo3d] failed to load glTF {}: {e}", path.display());
                    self.setup_fallback_cube(&mut renderer, gpu);
                }
            }
        } else {
            self.setup_fallback_cube(&mut renderer, gpu);
        }

        self.renderer = Some(renderer);
    }

    fn on_redraw(
        &mut self,
        gpu: &GpuContext,
        _resources: &mut RenderResources,
        frame: &mut Frame,
        perf: &esox_platform::perf::PerfMonitor,
    ) {
        let w = gpu.config.width as f32;

        // Semi-transparent black bar at the top.
        frame.push(esox_gfx::QuadInstance {
            rect: [0.0, 0.0, w, 32.0],
            uv: [0.0; 4],
            color: [0.0, 0.0, 0.0, 0.6],
            border_radius: [0.0; 4],
            sdf_params: [0.0; 4],
            flags: [esox_gfx::ShapeType::Rect.to_f32(), 0.0, 1.0, 0.0],
            clip_rect: [0.0; 4],
            color2: [0.0; 4],
            extra: [0.0; 4],
        });

        // FPS bar indicator.
        let fps = perf.fps;
        let bar_w = (fps as f32).min(120.0);
        frame.push(esox_gfx::QuadInstance {
            rect: [8.0, 8.0, bar_w, 16.0],
            uv: [0.0; 4],
            color: [0.3, 1.0, 0.3, 0.9],
            border_radius: [4.0; 4],
            sdf_params: [0.0; 4],
            flags: [esox_gfx::ShapeType::Rect.to_f32(), 0.0, 1.0, 0.0],
            clip_rect: [0.0; 4],
            color2: [0.0; 4],
            extra: [0.0; 4],
        });
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

        let now = std::time::Instant::now();
        let elapsed = self.start.elapsed().as_secs_f32();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.angle = elapsed * 0.8;

        let mut cmd_bufs = Vec::new();

        // Advance animation if playing.
        if let Some(player) = self.anim_player.as_mut() {
            player.advance(dt, &self.anim_clips);

            // Upload joint matrices to GPU for all skinned meshes.
            if let Some(handles) = &self.gltf_handles {
                for &si in handles.skinned_mesh_indices.iter().flatten() {
                    renderer.update_joints(gpu, si, player.skinning_matrices());
                }
            }
        }

        // Dispatch compute skinning if needed.
        if let Some(skin_cmd) = renderer.dispatch_skinning(gpu) {
            cmd_bufs.push(skin_cmd);
        }

        if let Some(scene) = &self.scene {
            // Draw the loaded scene.
            scene.draw(renderer);
        } else if let (Some(mesh), Some(mat)) = (self.cube_mesh, self.cube_material) {
            // Draw the fallback spinning cube.
            let rotation = glam::Quat::from_euler(
                glam::EulerRot::YXZ,
                self.angle,
                self.angle * 0.6,
                0.0,
            );
            let transform = Transform {
                position: glam::Vec3::new(0.0, 1.0, 0.0),
                rotation,
                ..Transform::IDENTITY
            };
            let instance = InstanceData::from_transform(&transform);
            renderer.draw_with_material(mesh, mat, &[instance]);

            // Draw the ground plane.
            if let (Some(gm), Some(gmat)) = (self.ground_mesh, self.ground_material) {
                let ground_instance = InstanceData::from_transform(&Transform::IDENTITY);
                renderer.draw_with_material(gm, gmat, &[ground_instance]);
            }
        }

        // Orbit camera around the scene.
        if self.scene.is_some() {
            let radius = 5.0;
            let orbit_speed = 0.3;
            self.camera.position = glam::Vec3::new(
                radius * (elapsed * orbit_speed).cos(),
                2.5,
                radius * (elapsed * orbit_speed).sin(),
            );
        }

        let (render_cmd, _stats) = renderer.encode(
            gpu,
            surface_view,
            &self.camera,
            self.viewport.0,
            self.viewport.1,
            elapsed,
            dt,
            wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
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
        if event.pressed {
            if let Key::Named(NamedKey::Escape) = &event.key {
                std::process::exit(0);
            }
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

impl Demo3dApp {
    fn setup_fallback_cube(&mut self, renderer: &mut Renderer3D, gpu: &GpuContext) {
        let cube = MeshData::cube(1.5);
        let mesh = renderer.upload_mesh(gpu, &cube);

        let mat = renderer.create_material(
            gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [0.2, 0.5, 1.0, 1.0],
                roughness: 0.3,
                metallic: 0.7,
                emissive: [0.05, 0.1, 0.3],
                ..MaterialDescriptor::default()
            },
        );

        // Ground plane.
        let ground = MeshData::plane(20.0, 20.0, 1);
        let ground_mesh = renderer.upload_mesh(gpu, &ground);
        let ground_mat = renderer.create_material(
            gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [0.4, 0.4, 0.35, 1.0],
                roughness: 0.8,
                metallic: 0.0,
                ..MaterialDescriptor::default()
            },
        );

        self.cube_mesh = Some(mesh);
        self.cube_material = Some(mat);
        self.ground_mesh = Some(ground_mesh);
        self.ground_material = Some(ground_mat);
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("demo3d=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    // Accept a .glb/.gltf path as the first argument.
    let gltf_path = std::env::args().nth(1).map(PathBuf::from);

    let title = match &gltf_path {
        Some(path) => format!(
            "esox 3D — {}",
            path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "glTF".into())
        ),
        None => "esox 3D demo — spinning cube".into(),
    };

    let config = PlatformConfig {
        window: WindowConfig {
            title,
            width: Some(800),
            height: Some(600),
            ..WindowConfig::default()
        },
        ..PlatformConfig::default()
    };

    if let Err(e) = esox_platform::run(config, Box::new(Demo3dApp::new(gltf_path))) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
