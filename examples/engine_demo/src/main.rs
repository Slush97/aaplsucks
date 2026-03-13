//! Engine demo — validates Phase 5 (Game trait, ECS, input mapping, fixed timestep).
//!
//! Spawns a PBR scene using ECS entities: orbiting cubes, lights, and a camera.
//! WASD orbits the camera, Escape exits.

use esox_engine::*;
use esox_engine::glam::{Quat, Vec3};

struct EngineDemo {
    orbit_angle: f32,
    orbit_radius: f32,
    exit: bool,
}

/// Tag component for cubes that spin.
struct Spinner {
    speed: f32,
}

impl Game for EngineDemo {
    fn init(&mut self, ctx: &mut Ctx) {
        // ── Input bindings ──
        ctx.input
            .bind_action("exit", ActionBinding::Key(esox_engine::winit::keyboard::KeyCode::Escape));
        ctx.input.bind_axis(
            "orbit",
            AxisBinding::Keys {
                negative: esox_engine::winit::keyboard::KeyCode::KeyA,
                positive: esox_engine::winit::keyboard::KeyCode::KeyD,
            },
        );
        ctx.input.bind_axis(
            "zoom",
            AxisBinding::Keys {
                negative: esox_engine::winit::keyboard::KeyCode::KeyS,
                positive: esox_engine::winit::keyboard::KeyCode::KeyW,
            },
        );

        // ── Materials ──
        use esox_gfx::mesh3d::{MaterialDescriptor, MaterialType, MeshData};

        let blue_mat = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [0.2, 0.4, 1.0, 1.0],
                roughness: 0.3,
                metallic: 0.8,
                emissive: [0.02, 0.04, 0.15],
                ..MaterialDescriptor::default()
            },
        );

        let red_mat = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [1.0, 0.2, 0.15, 1.0],
                roughness: 0.5,
                metallic: 0.3,
                ..MaterialDescriptor::default()
            },
        );

        let ground_mat = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [0.35, 0.35, 0.3, 1.0],
                roughness: 0.9,
                metallic: 0.0,
                ..MaterialDescriptor::default()
            },
        );

        // ── Meshes ──
        let cube_data = MeshData::cube(1.0);
        let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &cube_data);

        let sphere_data = MeshData::sphere(0.6, 24, 16);
        let sphere_mesh = ctx.renderer.upload_mesh(ctx.gpu, &sphere_data);

        let ground_data = MeshData::plane(20.0, 20.0, 1);
        let ground_mesh = ctx.renderer.upload_mesh(ctx.gpu, &ground_data);

        // Register with asset manager so handles are tracked.
        ctx.assets.register_mesh(cube_mesh);
        ctx.assets.register_mesh(sphere_mesh);
        ctx.assets.register_mesh(ground_mesh);

        // ── ECS: spawn ground ──
        ctx.world.spawn((
            Transform3D {
                position: Vec3::ZERO,
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            MeshRenderer {
                mesh: ground_mesh,
                material: ground_mat,
                tint: [1.0; 4],
                visible: true,
            },
        ));

        // ── ECS: spawn spinning cubes in a ring ──
        let count = 6;
        for i in 0..count {
            let angle = (i as f32 / count as f32) * std::f32::consts::TAU;
            let radius = 3.0;
            let pos = Vec3::new(angle.cos() * radius, 0.8, angle.sin() * radius);
            let speed = 0.5 + (i as f32) * 0.3;

            ctx.world.spawn((
                Transform3D {
                    position: pos,
                    ..Transform3D::default()
                },
                GlobalTransform::default(),
                MeshRenderer {
                    mesh: cube_mesh,
                    material: if i % 2 == 0 { blue_mat } else { red_mat },
                    tint: [1.0; 4],
                    visible: true,
                },
                Spinner { speed },
            ));
        }

        // ── ECS: spawn a sphere at center ──
        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(0.0, 1.5, 0.0),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            MeshRenderer {
                mesh: sphere_mesh,
                material: blue_mat,
                tint: [1.0, 1.0, 1.0, 1.0],
                visible: true,
            },
        ));

        // ── ECS: spawn lights ──
        ctx.world.spawn((
            Transform3D {
                position: Vec3::ZERO,
                rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_3),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            DirectionalLightComponent {
                color: [1.0, 0.95, 0.85],
                intensity: 2.0,
            },
        ));

        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(2.0, 4.0, 2.0),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            PointLightComponent {
                color: [0.4, 0.7, 1.0],
                intensity: 10.0,
                range: 15.0,
                cast_shadows: false,
            },
        ));

        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(-3.0, 5.0, 0.0),
                rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2 * 0.8),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            SpotLightComponent {
                color: [1.0, 0.8, 0.3],
                intensity: 15.0,
                range: 15.0,
                inner_cone_angle: 15.0_f32.to_radians(),
                outer_cone_angle: 30.0_f32.to_radians(),
                cast_shadows: false,
            },
        ));

        // ── ECS: spawn camera ──
        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(
                    self.orbit_radius,
                    3.0,
                    self.orbit_radius,
                ),
                rotation: look_at_quat(
                    Vec3::new(self.orbit_radius, 3.0, self.orbit_radius),
                    Vec3::ZERO,
                ),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            Camera3D {
                fov_y: std::f32::consts::FRAC_PI_4,
                near: 0.1,
                far: 200.0,
                active: true,
            },
        ));
    }

    fn update(&mut self, ctx: &mut Ctx) {
        // ── Input ──
        if ctx.input.just_pressed("exit") {
            self.exit = true;
            return;
        }

        let orbit_input = ctx.input.axis("orbit");
        let zoom_input = ctx.input.axis("zoom");

        self.orbit_angle += orbit_input * 2.0 * ctx.time.tick_dt;
        self.orbit_radius = (self.orbit_radius - zoom_input * 5.0 * ctx.time.tick_dt).clamp(2.0, 20.0);

        // ── Spin cubes ──
        for (_e, (t, spinner)) in ctx
            .world
            .query_mut::<(&mut Transform3D, &Spinner)>()
        {
            t.rotation *= Quat::from_rotation_y(spinner.speed * ctx.time.tick_dt);
        }

        // ── Update camera position from orbit ──
        for (_e, (t, cam)) in ctx
            .world
            .query_mut::<(&mut Transform3D, &Camera3D)>()
        {
            if cam.active {
                let pos = Vec3::new(
                    self.orbit_angle.cos() * self.orbit_radius,
                    3.0,
                    self.orbit_angle.sin() * self.orbit_radius,
                );
                t.position = pos;
                t.rotation = look_at_quat(pos, Vec3::new(0.0, 0.5, 0.0));
            }
        }
    }

    fn should_exit(&self) -> bool {
        self.exit
    }
}

/// Compute a quaternion that looks from `eye` toward `target` (Y-up).
fn look_at_quat(eye: Vec3, target: Vec3) -> Quat {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward))
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("engine_demo=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "esox engine demo".into(),
                width: Some(1280),
                height: Some(720),
                ..Default::default()
            },
            ..Default::default()
        },
        ..EngineConfig::default()
    };

    let game = EngineDemo {
        orbit_angle: 0.0,
        orbit_radius: 6.0,
        exit: false,
    };

    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
