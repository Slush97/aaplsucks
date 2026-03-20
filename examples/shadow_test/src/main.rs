//! Shadow test — isolated scenes for debugging shadow mapping.
//!
//! Press 1–6 to switch scenes. WASD + mouse to move. Esc to quit.
//!
//! Scenes:
//!   1. Single point light, a few primitives on a ground plane
//!   2. Single spot light aimed at objects
//!   3. Directional light only (CSM cascades)
//!   4. Two point lights with overlapping ranges
//!   5. Thin wall geometry (shadow acne / light leak stress test)
//!   6. All light types combined

use esox_engine::*;
use esox_engine::glam::{Quat, Vec3};
use esox_engine::esox_gfx::mesh3d::{
    MaterialDescriptor, MaterialType, MeshData, MeshHandle, PostProcess3DConfig,
};

use std::f32::consts::FRAC_PI_4;

const SCENE_COUNT: usize = 6;

struct ShadowTest {
    yaw: f32,
    pitch: f32,
    camera_pos: Vec3,
    prev_camera_pos: Vec3,
    camera_entity: Option<hecs::Entity>,
    scene_entities: Vec<hecs::Entity>,
    current_scene: usize,
    cube_mesh: Option<MeshHandle>,
    sphere_mesh: Option<MeshHandle>,
    cylinder_mesh: Option<MeshHandle>,
    plane_mesh: Option<MeshHandle>,
    exit: bool,
}

impl ShadowTest {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: -0.3,
            camera_pos: Vec3::new(0.0, 5.0, 10.0),
            prev_camera_pos: Vec3::new(0.0, 5.0, 10.0),
            camera_entity: None,
            scene_entities: Vec::new(),
            current_scene: 0,
            cube_mesh: None,
            sphere_mesh: None,
            cylinder_mesh: None,
            plane_mesh: None,
            exit: false,
        }
    }

    fn clear_scene(&mut self, world: &mut hecs::World) {
        for e in self.scene_entities.drain(..) {
            let _ = world.despawn(e);
        }
    }

    fn load_scene(&mut self, scene: usize, ctx: &mut Ctx) {
        self.clear_scene(ctx.world);
        self.current_scene = scene;

        let cube = self.cube_mesh.unwrap();
        let sphere = self.sphere_mesh.unwrap();
        let cylinder = self.cylinder_mesh.unwrap();
        let white_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.8, 0.8, 0.8, 1.0],
            roughness: 0.7,
            metallic: 0.0,
            ..MaterialDescriptor::default()
        });

        let red_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.8, 0.15, 0.1, 1.0],
            roughness: 0.5,
            metallic: 0.0,
            ..MaterialDescriptor::default()
        });

        let blue_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.1, 0.2, 0.8, 1.0],
            roughness: 0.4,
            metallic: 0.1,
            ..MaterialDescriptor::default()
        });

        let green_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.15, 0.7, 0.2, 1.0],
            roughness: 0.6,
            metallic: 0.0,
            ..MaterialDescriptor::default()
        });

        let gold_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [1.0, 0.85, 0.3, 1.0],
            roughness: 0.3,
            metallic: 0.9,
            ..MaterialDescriptor::default()
        });

        // Ground plane (shared by all scenes) — oversized so edges are never visible.
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::ZERO, scale: Vec3::new(200.0, 0.2, 200.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: white_mat, tint: [1.0; 4], visible: true },
        )));

        match scene {
            0 => self.scene_single_point(ctx, cube, sphere, cylinder, white_mat, red_mat, blue_mat),
            1 => self.scene_single_spot(ctx, cube, sphere, cylinder, white_mat, red_mat, green_mat),
            2 => self.scene_directional(ctx, cube, sphere, cylinder, white_mat, red_mat, blue_mat),
            3 => self.scene_two_points(ctx, cube, sphere, cylinder, white_mat, red_mat, gold_mat),
            4 => self.scene_thin_walls(ctx, cube, sphere, white_mat, red_mat, blue_mat),
            5 => self.scene_combined(ctx, cube, sphere, cylinder, white_mat, red_mat, blue_mat, green_mat),
            _ => {}
        }

        eprintln!("[shadow_test] loaded scene {} — {}", scene + 1, scene_name(scene));
    }

    // Scene 1: Single point light
    fn scene_single_point(
        &mut self, ctx: &mut Ctx,
        cube: MeshHandle, sphere: MeshHandle, cylinder: MeshHandle,
        white_mat: esox_gfx::mesh3d::MaterialHandle,
        red_mat: esox_gfx::mesh3d::MaterialHandle,
        blue_mat: esox_gfx::mesh3d::MaterialHandle,
    ) {
        // Objects
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 1.0, 0.0), scale: Vec3::splat(2.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: red_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(4.0, 1.0, -2.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: sphere, material: blue_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(-3.0, 1.5, -1.0), scale: Vec3::new(1.0, 3.0, 1.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cylinder, material: white_mat, tint: [1.0; 4], visible: true },
        )));

        // Point light
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 6.0, 0.0), ..Default::default() },
            GlobalTransform::default(),
            PointLightComponent { color: [1.0, 0.95, 0.9], intensity: 20.0, range: 25.0, cast_shadows: true },
        )));
    }

    // Scene 2: Single spot light
    fn scene_single_spot(
        &mut self, ctx: &mut Ctx,
        cube: MeshHandle, sphere: MeshHandle, cylinder: MeshHandle,
        white_mat: esox_gfx::mesh3d::MaterialHandle,
        red_mat: esox_gfx::mesh3d::MaterialHandle,
        green_mat: esox_gfx::mesh3d::MaterialHandle,
    ) {
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 1.0, 0.0), scale: Vec3::splat(2.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: red_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(3.0, 0.75, -3.0), scale: Vec3::splat(1.5), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: sphere, material: green_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(-2.5, 1.0, 2.0), scale: Vec3::new(1.0, 2.0, 1.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cylinder, material: white_mat, tint: [1.0; 4], visible: true },
        )));

        // Spot light aimed downward at an angle
        self.scene_entities.push(ctx.world.spawn((
            Transform3D {
                position: Vec3::new(5.0, 8.0, 5.0),
                rotation: Quat::from_rotation_arc(
                    Vec3::NEG_Z,
                    (Vec3::new(0.0, 0.0, 0.0) - Vec3::new(5.0, 8.0, 5.0)).normalize(),
                ),
                ..Default::default()
            },
            GlobalTransform::default(),
            SpotLightComponent {
                color: [1.0, 1.0, 0.95],
                intensity: 30.0,
                range: 25.0,
                inner_cone_angle: 0.3,
                outer_cone_angle: 0.6,
                cast_shadows: true,
            },
        )));
    }

    // Scene 3: Directional light only
    fn scene_directional(
        &mut self, ctx: &mut Ctx,
        cube: MeshHandle, sphere: MeshHandle, cylinder: MeshHandle,
        white_mat: esox_gfx::mesh3d::MaterialHandle,
        red_mat: esox_gfx::mesh3d::MaterialHandle,
        blue_mat: esox_gfx::mesh3d::MaterialHandle,
    ) {
        // Spread objects at different distances to test CSM cascade splits
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 1.0, 0.0), scale: Vec3::splat(2.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: red_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(6.0, 0.75, -8.0), scale: Vec3::splat(1.5), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: sphere, material: blue_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(-4.0, 2.0, -4.0), scale: Vec3::new(1.0, 4.0, 1.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cylinder, material: white_mat, tint: [1.0; 4], visible: true },
        )));
        // Far object to test distant cascade
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(-8.0, 1.5, -12.0), scale: Vec3::splat(3.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: blue_mat, tint: [1.0; 4], visible: true },
        )));

        // Directional light (sun-like, angled)
        self.scene_entities.push(ctx.world.spawn((
            Transform3D {
                rotation: Quat::from_rotation_arc(
                    Vec3::NEG_Z,
                    Vec3::new(-0.4, -0.8, -0.4).normalize(),
                ),
                ..Default::default()
            },
            GlobalTransform::default(),
            DirectionalLightComponent { color: [1.0, 0.98, 0.92], intensity: 3.0 },
        )));
    }

    // Scene 4: Two overlapping point lights
    fn scene_two_points(
        &mut self, ctx: &mut Ctx,
        cube: MeshHandle, sphere: MeshHandle, cylinder: MeshHandle,
        white_mat: esox_gfx::mesh3d::MaterialHandle,
        red_mat: esox_gfx::mesh3d::MaterialHandle,
        gold_mat: esox_gfx::mesh3d::MaterialHandle,
    ) {
        // Central objects between two lights
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 1.0, 0.0), scale: Vec3::splat(2.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: white_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 2.5, 0.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: sphere, material: gold_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(3.0, 1.5, -2.0), scale: Vec3::new(0.5, 3.0, 0.5), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cylinder, material: red_mat, tint: [1.0; 4], visible: true },
        )));

        // Warm light (left)
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(-5.0, 5.0, 0.0), ..Default::default() },
            GlobalTransform::default(),
            PointLightComponent { color: [1.0, 0.6, 0.2], intensity: 18.0, range: 20.0, cast_shadows: true },
        )));

        // Cool light (right)
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(5.0, 5.0, 0.0), ..Default::default() },
            GlobalTransform::default(),
            PointLightComponent { color: [0.3, 0.5, 1.0], intensity: 18.0, range: 20.0, cast_shadows: true },
        )));
    }

    // Scene 5: Thin wall stress test
    fn scene_thin_walls(
        &mut self, ctx: &mut Ctx,
        cube: MeshHandle, sphere: MeshHandle,
        white_mat: esox_gfx::mesh3d::MaterialHandle,
        red_mat: esox_gfx::mesh3d::MaterialHandle,
        blue_mat: esox_gfx::mesh3d::MaterialHandle,
    ) {
        // Thin walls at various thicknesses
        let thicknesses = [0.05, 0.1, 0.2, 0.5];
        for (i, &thick) in thicknesses.iter().enumerate() {
            let x = (i as f32 - 1.5) * 4.0;
            self.scene_entities.push(ctx.world.spawn((
                Transform3D {
                    position: Vec3::new(x, 2.0, -2.0),
                    scale: Vec3::new(thick, 4.0, 3.0),
                    ..Default::default()
                },
                GlobalTransform::default(),
                MeshRenderer { mesh: cube, material: white_mat, tint: [1.0; 4], visible: true },
            )));
        }

        // Object behind the thinnest wall to check light leak
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(-6.0, 1.0, -3.5), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: sphere, material: red_mat, tint: [1.0; 4], visible: true },
        )));

        // Floor-touching cube to check contact shadows / acne
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(3.0, 0.5, 2.0), scale: Vec3::splat(1.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: blue_mat, tint: [1.0; 4], visible: true },
        )));

        // Point light above
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 7.0, 0.0), ..Default::default() },
            GlobalTransform::default(),
            PointLightComponent { color: [1.0, 1.0, 1.0], intensity: 25.0, range: 25.0, cast_shadows: true },
        )));
    }

    // Scene 6: All light types combined
    fn scene_combined(
        &mut self, ctx: &mut Ctx,
        cube: MeshHandle, sphere: MeshHandle, cylinder: MeshHandle,
        white_mat: esox_gfx::mesh3d::MaterialHandle,
        red_mat: esox_gfx::mesh3d::MaterialHandle,
        blue_mat: esox_gfx::mesh3d::MaterialHandle,
        green_mat: esox_gfx::mesh3d::MaterialHandle,
    ) {
        // Various objects
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 1.0, 0.0), scale: Vec3::splat(2.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: red_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(5.0, 1.0, -3.0), scale: Vec3::splat(1.5), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: sphere, material: blue_mat, tint: [1.0; 4], visible: true },
        )));
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(-4.0, 1.5, 2.0), scale: Vec3::new(1.0, 3.0, 1.0), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cylinder, material: green_mat, tint: [1.0; 4], visible: true },
        )));
        // Wall to cast interesting multi-source shadows
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 2.0, -5.0), scale: Vec3::new(6.0, 4.0, 0.3), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube, material: white_mat, tint: [1.0; 4], visible: true },
        )));

        // Directional (sun)
        self.scene_entities.push(ctx.world.spawn((
            Transform3D {
                rotation: Quat::from_rotation_arc(
                    Vec3::NEG_Z,
                    Vec3::new(-0.3, -0.9, -0.3).normalize(),
                ),
                ..Default::default()
            },
            GlobalTransform::default(),
            DirectionalLightComponent { color: [1.0, 0.95, 0.85], intensity: 1.5 },
        )));

        // Point light
        self.scene_entities.push(ctx.world.spawn((
            Transform3D { position: Vec3::new(-5.0, 4.0, 3.0), ..Default::default() },
            GlobalTransform::default(),
            PointLightComponent { color: [1.0, 0.6, 0.2], intensity: 15.0, range: 18.0, cast_shadows: true },
        )));

        // Spot light
        self.scene_entities.push(ctx.world.spawn((
            Transform3D {
                position: Vec3::new(6.0, 7.0, 4.0),
                rotation: Quat::from_rotation_arc(
                    Vec3::NEG_Z,
                    (Vec3::new(0.0, 0.0, 0.0) - Vec3::new(6.0, 7.0, 4.0)).normalize(),
                ),
                ..Default::default()
            },
            GlobalTransform::default(),
            SpotLightComponent {
                color: [0.4, 0.6, 1.0],
                intensity: 25.0,
                range: 20.0,
                inner_cone_angle: 0.2,
                outer_cone_angle: 0.5,
                cast_shadows: true,
            },
        )));
    }
}

fn scene_name(scene: usize) -> &'static str {
    match scene {
        0 => "Single point light",
        1 => "Single spot light",
        2 => "Directional light (CSM)",
        3 => "Two overlapping point lights",
        4 => "Thin wall stress test",
        5 => "All lights combined",
        _ => "Unknown",
    }
}

impl Game for ShadowTest {
    fn init(&mut self, ctx: &mut Ctx) {
        use esox_engine::esox_input::KeyCode;

        ctx.input.bind_axis("move_x", AxisBinding::Keys { negative: KeyCode::KeyA, positive: KeyCode::KeyD });
        ctx.input.bind_axis("move_z", AxisBinding::Keys { negative: KeyCode::KeyS, positive: KeyCode::KeyW });
        ctx.input.bind_axis("look_x", AxisBinding::MouseDelta(MouseAxis::X));
        ctx.input.bind_axis("look_y", AxisBinding::MouseDelta(MouseAxis::Y));
        ctx.input.bind_action("exit", ActionBinding::Key(KeyCode::Escape));

        ctx.input.bind_action("scene_1", ActionBinding::Key(KeyCode::Digit1));
        ctx.input.bind_action("scene_2", ActionBinding::Key(KeyCode::Digit2));
        ctx.input.bind_action("scene_3", ActionBinding::Key(KeyCode::Digit3));
        ctx.input.bind_action("scene_4", ActionBinding::Key(KeyCode::Digit4));
        ctx.input.bind_action("scene_5", ActionBinding::Key(KeyCode::Digit5));
        ctx.input.bind_action("scene_6", ActionBinding::Key(KeyCode::Digit6));

        // Grab cursor for first-person mouse look.
        ctx.input.set_cursor_grab(true);

        // Upload shared meshes
        let cube_data = MeshData::cube(1.0);
        let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &cube_data);
        ctx.assets.register_mesh(cube_mesh);
        self.cube_mesh = Some(cube_mesh);

        let sphere_data = MeshData::sphere(1.0, 24, 16);
        let sphere_mesh = ctx.renderer.upload_mesh(ctx.gpu, &sphere_data);
        ctx.assets.register_mesh(sphere_mesh);
        self.sphere_mesh = Some(sphere_mesh);

        let cylinder_data = MeshData::cylinder(0.5, 1.0, 20);
        let cylinder_mesh = ctx.renderer.upload_mesh(ctx.gpu, &cylinder_data);
        ctx.assets.register_mesh(cylinder_mesh);
        self.cylinder_mesh = Some(cylinder_mesh);

        let plane_data = MeshData::plane(1.0, 1.0, 1);
        let plane_mesh = ctx.renderer.upload_mesh(ctx.gpu, &plane_data);
        ctx.assets.register_mesh(plane_mesh);
        self.plane_mesh = Some(plane_mesh);

        // Camera
        let cam_entity = ctx.world.spawn((
            Transform3D { position: self.camera_pos, ..Default::default() },
            GlobalTransform::default(),
            Camera3D { active: true, fov_y: FRAC_PI_4, near: 0.1, far: 100.0 },
        ));
        self.camera_entity = Some(cam_entity);

        ctx.renderer.set_postprocess(PostProcess3DConfig {
            bloom_enabled: false,
            bloom_intensity: 0.0,
            bloom_threshold: 10.0,
            bloom_soft_knee: 0.0,
            tone_map_enabled: true,
            ssao_enabled: true,
            fog_enabled: false,
            fog_color: [0.75, 0.82, 0.90],
            fog_start: 50.0,
            fog_end: 200.0,
        });

        // Start with scene 1
        self.load_scene(0, ctx);
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.input.just_pressed("exit") {
            self.exit = true;
            return;
        }

        // Scene switching
        for i in 0..SCENE_COUNT {
            let action = format!("scene_{}", i + 1);
            if ctx.input.just_pressed(&action) && self.current_scene != i {
                self.load_scene(i, ctx);
            }
        }

        let dt = ctx.time.tick_dt;

        // Mouse look
        let look_x = ctx.input.axis("look_x");
        let look_y = ctx.input.axis("look_y");
        let sensitivity = 0.003;
        self.yaw -= look_x * sensitivity;
        self.pitch = (self.pitch - look_y * sensitivity).clamp(-1.4, 1.4);

        // Movement
        let move_x = ctx.input.axis("move_x");
        let move_z = ctx.input.axis("move_z");
        let move_speed = 8.0;

        let (sin_y, cos_y) = self.yaw.sin_cos();
        let forward = Vec3::new(-sin_y, 0.0, -cos_y);
        let right = Vec3::new(cos_y, 0.0, -sin_y);
        let move_dir = (forward * move_z + right * move_x).normalize_or_zero();

        self.prev_camera_pos = self.camera_pos;
        self.camera_pos += move_dir * move_speed * dt;
    }

    fn render(&mut self, ctx: &mut Ctx, alpha: f32) {
        let visual_pos = self.prev_camera_pos.lerp(self.camera_pos, alpha);

        if let Some(ce) = self.camera_entity {
            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(ce) {
                t.position = visual_pos;
                t.rotation = Quat::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0);
            }
        }
    }

    fn ui(&mut self, ui: &mut esox_engine::esox_ui::Ui, _ctx: &Ctx) {
        use esox_gfx::Color;

        ui.padding(12.0, |ui| {
            ui.label_colored(
                &format!("Scene {}: {}", self.current_scene + 1, scene_name(self.current_scene)),
                Color::WHITE,
            );
        });
        ui.padding(12.0, |ui| {
            ui.label_colored("[1-6] Switch scene  [WASD] Move  [Esc] Quit", Color::new(0.6, 0.6, 0.6, 1.0));
        });
    }

    fn should_exit(&self) -> bool {
        self.exit
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("shadow_test=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "esox shadow test".into(),
                width: Some(1280),
                height: Some(720),
                ..Default::default()
            },
            msaa: 4,
            ..Default::default()
        },
        shadows: true,
        ..EngineConfig::default()
    };

    let game = ShadowTest::new();

    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
