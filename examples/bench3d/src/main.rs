//! 3D performance benchmark — spawns N objects in a grid with mixed meshes/materials
//! and an orbiting camera. Measures frame time, CPU encode time, draw calls, cull rate,
//! and triangle count.
//!
//! Usage: cargo run --release -p bench3d -- [count=10000]

use std::time::Instant;

use esox_gfx::mesh3d::{
    Camera, InstanceData, LightEnvironment, MaterialDescriptor, MaterialHandle,
    MaterialType, MeshData, MeshHandle, Renderer3D, Transform,
};
use esox_gfx::{Frame, GpuContext, RenderResources};
use esox_platform::config::{PlatformConfig, WindowConfig};
use esox_platform::{AppDelegate, MouseInputEvent};

struct BenchApp {
    renderer: Option<Renderer3D>,
    meshes: Vec<MeshHandle>,
    materials: Vec<MaterialHandle>,
    object_count: usize,
    camera: Camera,
    start: Instant,
    frame_count: u64,
    last_report: Instant,
    viewport: (u32, u32),
}

impl BenchApp {
    fn new(object_count: usize) -> Self {
        Self {
            renderer: None,
            meshes: Vec::new(),
            materials: Vec::new(),
            object_count,
            camera: Camera {
                position: glam::Vec3::new(50.0, 30.0, 50.0),
                target: glam::Vec3::ZERO,
                ..Camera::default()
            },
            start: Instant::now(),
            frame_count: 0,
            last_report: Instant::now(),
            viewport: (1280, 720),
        }
    }
}

impl AppDelegate for BenchApp {
    fn on_init(&mut self, gpu: &GpuContext, _resources: &mut RenderResources) {
        let mut renderer = Renderer3D::new(gpu);

        let lights = LightEnvironment::default();
        renderer.set_lights(&lights);

        // Upload 4 different mesh types for variety.
        let cube = renderer.upload_mesh(gpu, &MeshData::cube(1.0));
        let sphere = renderer.upload_mesh(gpu, &MeshData::sphere(0.5, 16, 8));
        let cylinder = renderer.upload_mesh(gpu, &MeshData::cylinder(0.3, 1.0, 12));
        let cone = renderer.upload_mesh(gpu, &MeshData::cone(0.4, 1.0, 12));
        self.meshes = vec![cube, sphere, cylinder, cone];

        // Create a few materials.
        let mat_red = renderer.create_material(
            gpu,
            &MaterialDescriptor {
                albedo: [1.0, 0.2, 0.1, 1.0],
                material_type: MaterialType::Lit,
                ..Default::default()
            },
        );
        let mat_blue = renderer.create_material(
            gpu,
            &MaterialDescriptor {
                albedo: [0.1, 0.3, 1.0, 1.0],
                material_type: MaterialType::Lit,
                ..Default::default()
            },
        );
        let mat_green = renderer.create_material(
            gpu,
            &MaterialDescriptor {
                albedo: [0.1, 0.8, 0.2, 1.0],
                material_type: MaterialType::Lit,
                ..Default::default()
            },
        );
        self.materials = vec![mat_red, mat_blue, mat_green];

        self.renderer = Some(renderer);
        self.start = Instant::now();
        self.last_report = Instant::now();

        eprintln!(
            "bench3d: {} objects, {} meshes, {} materials",
            self.object_count,
            self.meshes.len(),
            self.materials.len()
        );
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
        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return vec![],
        };

        let elapsed = self.start.elapsed().as_secs_f32();
        let delta = 1.0 / 60.0;

        // Orbit camera.
        let orbit_radius = 60.0;
        let angle = elapsed * 0.3;
        self.camera.position = glam::Vec3::new(
            angle.cos() * orbit_radius,
            30.0 + (elapsed * 0.5).sin() * 10.0,
            angle.sin() * orbit_radius,
        );
        self.camera.target = glam::Vec3::ZERO;

        // Spawn objects in a grid.
        let side = (self.object_count as f32).cbrt().ceil() as usize;
        let spacing = 2.5;

        let encode_start = Instant::now();

        for i in 0..self.object_count {
            let x = (i % side) as f32 * spacing - (side as f32 * spacing * 0.5);
            let y = ((i / side) % side) as f32 * spacing;
            let z = (i / (side * side)) as f32 * spacing - (side as f32 * spacing * 0.5);

            let mesh = self.meshes[i % self.meshes.len()];
            let material = self.materials[i % self.materials.len()];

            let instance = InstanceData::from_transform(&Transform {
                position: glam::Vec3::new(x, y, z),
                ..Transform::default()
            });

            renderer.draw_with_material(mesh, material, &[instance]);
        }

        let (cmd_buf, stats) = renderer.encode(
            gpu,
            surface_view,
            &self.camera,
            self.viewport.0,
            self.viewport.1,
            elapsed,
            delta,
            wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.08,
                a: 1.0,
            },
        );

        let encode_time = encode_start.elapsed();

        self.frame_count += 1;

        // Report every 2 seconds.
        if self.last_report.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.frame_count as f32 / self.start.elapsed().as_secs_f32();
            eprintln!(
                "bench3d: {:.1} fps | encode {:.2}ms | draws {} | culled {} | instances {} | tris {} | pipelines {} | materials {}",
                fps,
                encode_time.as_secs_f64() * 1000.0,
                stats.draw_calls,
                stats.culled_draws,
                stats.total_instances,
                stats.total_triangles,
                stats.pipeline_switches,
                stats.material_switches,
            );
            self.last_report = Instant::now();
        }

        vec![cmd_buf]
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
        if width > 0 && height > 0 {
            self.viewport = (width, height);
        }
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
    tracing_subscriber::fmt::init();

    let count: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let config = PlatformConfig {
        window: WindowConfig {
            title: format!("bench3d — {count} objects"),
            width: Some(1280),
            height: Some(720),
            ..WindowConfig::default()
        },
        ..PlatformConfig::default()
    };

    if let Err(e) = esox_platform::run(config, Box::new(BenchApp::new(count))) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
