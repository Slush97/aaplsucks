//! Dungeon walkthrough — first-person demo validating PBR + scene serialization.

use esox_engine::*;
use esox_engine::glam::{Quat, Vec3};
use esox_engine::esox_gfx::mesh3d::{
    MaterialDescriptor, MaterialType, MeshData, MeshHandle, PostProcess3DConfig,
};

use std::f32::consts::FRAC_PI_4;

// ── Interactive objects ──

#[derive(Clone)]
enum Interactive {
    Torch { entity: hecs::Entity, lit: bool },
    Pickup { entity: hecs::Entity, collected: bool },
}

// ── Room state ──

struct RoomEntities {
    entities: Vec<hecs::Entity>,
}

// ── Game state ──

struct Dungeon {
    yaw: f32,
    pitch: f32,
    camera_pos: Vec3,
    prev_camera_pos: Vec3,
    camera_entity: Option<hecs::Entity>,
    rooms: Vec<RoomEntities>,
    interactive: Vec<Interactive>,
    collected: u32,
    total_pickups: u32,
    exit: bool,
}

impl Dungeon {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            camera_pos: Vec3::new(0.0, 1.7, 0.0),
            prev_camera_pos: Vec3::new(0.0, 1.7, 0.0),
            camera_entity: None,
            rooms: Vec::new(),
            interactive: Vec::new(),
            collected: 0,
            total_pickups: 0,
            exit: false,
        }
    }
}

// ── Room construction ──

fn build_entry_hall(ctx: &mut Ctx, cube_mesh: MeshHandle) -> (Vec<hecs::Entity>, Vec<Interactive>) {
    let stone_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.35, 0.32, 0.28, 1.0],
        roughness: 0.95,
        metallic: 0.0,
        ..MaterialDescriptor::default()
    });

    let mut entities = Vec::new();
    let mut interactives = Vec::new();

    let room_w = 10.0_f32;
    let room_h = 5.0_f32;
    let room_d = 12.0_f32;

    // Floor
    entities.push(ctx.world.spawn((
        Transform3D { position: Vec3::new(0.0, 0.0, -room_d / 2.0), scale: Vec3::new(room_w, 0.2, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Ceiling
    entities.push(ctx.world.spawn((
        Transform3D { position: Vec3::new(0.0, room_h, -room_d / 2.0), scale: Vec3::new(room_w, 0.2, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Left wall
    entities.push(ctx.world.spawn((
        Transform3D { position: Vec3::new(-room_w / 2.0, room_h / 2.0, -room_d / 2.0), scale: Vec3::new(0.3, room_h, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Right wall
    entities.push(ctx.world.spawn((
        Transform3D { position: Vec3::new(room_w / 2.0, room_h / 2.0, -room_d / 2.0), scale: Vec3::new(0.3, room_h, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Back wall
    entities.push(ctx.world.spawn((
        Transform3D { position: Vec3::new(0.0, room_h / 2.0, -room_d), scale: Vec3::new(room_w, room_h, 0.3), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Front wall
    entities.push(ctx.world.spawn((
        Transform3D { position: Vec3::new(0.0, room_h / 2.0, 0.0), scale: Vec3::new(room_w, room_h, 0.3), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Torch lights on walls
    let torch_positions = [
        Vec3::new(-room_w / 2.0 + 0.5, 3.0, -room_d / 4.0),
        Vec3::new(room_w / 2.0 - 0.5, 3.0, -room_d * 3.0 / 4.0),
    ];

    for pos in torch_positions {
        let torch_entity = ctx.world.spawn((
            Transform3D { position: pos, ..Default::default() },
            GlobalTransform::default(),
            PointLightComponent { color: [1.0, 0.7, 0.3], intensity: 12.0, range: 15.0, cast_shadows: true },
            Tag("torch".into()),
        ));
        entities.push(torch_entity);
        interactives.push(Interactive::Torch { entity: torch_entity, lit: true });
    }

    (entities, interactives)
}

fn build_corridor(ctx: &mut Ctx, offset: Vec3, cube_mesh: MeshHandle) -> Vec<hecs::Entity> {
    let stone_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.3, 0.28, 0.25, 1.0],
        roughness: 0.9,
        metallic: 0.05,
        ..MaterialDescriptor::default()
    });

    let mut entities = Vec::new();

    let corr_w = 3.0_f32;
    let corr_h = 3.5_f32;
    let corr_d = 8.0_f32;

    // Floor
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, 0.0, -corr_d / 2.0), scale: Vec3::new(corr_w, 0.2, corr_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Ceiling
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, corr_h, -corr_d / 2.0), scale: Vec3::new(corr_w, 0.2, corr_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Left wall
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(-corr_w / 2.0, corr_h / 2.0, -corr_d / 2.0), scale: Vec3::new(0.2, corr_h, corr_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Right wall
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(corr_w / 2.0, corr_h / 2.0, -corr_d / 2.0), scale: Vec3::new(0.2, corr_h, corr_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    entities
}

fn build_treasure_room(ctx: &mut Ctx, offset: Vec3, cube_mesh: MeshHandle) -> (Vec<hecs::Entity>, Vec<Interactive>) {
    let stone_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.3, 0.28, 0.22, 1.0],
        roughness: 0.85,
        metallic: 0.1,
        ..MaterialDescriptor::default()
    });

    let gold_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [1.0, 0.85, 0.2, 1.0],
        roughness: 0.2,
        metallic: 0.95,
        emissive: [0.05, 0.04, 0.01],
        ..MaterialDescriptor::default()
    });

    let sphere_data = MeshData::sphere(0.3, 16, 12);
    let sphere_mesh = ctx.renderer.upload_mesh(ctx.gpu, &sphere_data);
    ctx.assets.register_mesh(sphere_mesh);

    let mut entities = Vec::new();
    let mut interactives = Vec::new();

    let room_w = 8.0_f32;
    let room_h = 4.5_f32;
    let room_d = 8.0_f32;

    // Floor
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, 0.0, -room_d / 2.0), scale: Vec3::new(room_w, 0.2, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Ceiling
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, room_h, -room_d / 2.0), scale: Vec3::new(room_w, 0.2, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Left wall
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(-room_w / 2.0, room_h / 2.0, -room_d / 2.0), scale: Vec3::new(0.3, room_h, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Right wall
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(room_w / 2.0, room_h / 2.0, -room_d / 2.0), scale: Vec3::new(0.3, room_h, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Back wall
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, room_h / 2.0, -room_d), scale: Vec3::new(room_w, room_h, 0.3), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Golden pickups
    let pickup_positions = [
        Vec3::new(-2.0, 0.8, -room_d / 2.0),
        Vec3::new(0.0, 0.8, -room_d * 0.7),
        Vec3::new(2.0, 0.8, -room_d / 2.0),
    ];

    for pos in pickup_positions {
        let pickup_entity = ctx.world.spawn((
            Transform3D { position: offset + pos, ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: sphere_mesh, material: gold_mat, tint: [1.0; 4], visible: true },
            Tag("pickup".into()),
        ));
        entities.push(pickup_entity);
        interactives.push(Interactive::Pickup { entity: pickup_entity, collected: false });
    }

    // Ambient warm light for the room
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, room_h - 0.5, -room_d / 2.0), ..Default::default() },
        GlobalTransform::default(),
        PointLightComponent { color: [1.0, 0.85, 0.5], intensity: 10.0, range: 12.0, cast_shadows: true },
    )));

    (entities, interactives)
}

fn build_exit_chamber(ctx: &mut Ctx, offset: Vec3, cube_mesh: MeshHandle) -> Vec<hecs::Entity> {
    let stone_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.4, 0.38, 0.35, 1.0],
        roughness: 0.8,
        metallic: 0.05,
        ..MaterialDescriptor::default()
    });

    let mut entities = Vec::new();

    let room_w = 6.0_f32;
    let room_h = 8.0_f32;
    let room_d = 6.0_f32;

    // Floor
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, 0.0, -room_d / 2.0), scale: Vec3::new(room_w, 0.2, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Ceiling (with gap for light shaft)
    // Left part
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(-room_w / 4.0, room_h, -room_d / 2.0), scale: Vec3::new(room_w / 2.0 - 0.5, 0.2, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Right part
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(room_w / 4.0, room_h, -room_d / 2.0), scale: Vec3::new(room_w / 2.0 - 0.5, 0.2, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Walls
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(-room_w / 2.0, room_h / 2.0, -room_d / 2.0), scale: Vec3::new(0.3, room_h, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(room_w / 2.0, room_h / 2.0, -room_d / 2.0), scale: Vec3::new(0.3, room_h, room_d), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, room_h / 2.0, -room_d), scale: Vec3::new(room_w, room_h, 0.3), ..Default::default() },
        GlobalTransform::default(),
        MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
    )));

    // Light shaft through ceiling gap
    entities.push(ctx.world.spawn((
        Transform3D { position: offset + Vec3::new(0.0, room_h - 0.5, -room_d / 2.0), ..Default::default() },
        GlobalTransform::default(),
        PointLightComponent { color: [0.9, 0.95, 1.0], intensity: 12.0, range: 15.0, cast_shadows: true },
    )));

    entities
}

// ── Game implementation ──

impl Game for Dungeon {
    fn init(&mut self, ctx: &mut Ctx) {
        use esox_engine::winit::keyboard::KeyCode;

        // Input bindings: WASD + mouse look + E interact + F5/F9 save/load + Esc
        ctx.input.bind_axis("move_x", AxisBinding::Keys { negative: KeyCode::KeyA, positive: KeyCode::KeyD });
        ctx.input.bind_axis("move_z", AxisBinding::Keys { negative: KeyCode::KeyS, positive: KeyCode::KeyW });
        ctx.input.bind_axis("look_x", AxisBinding::MouseDelta(MouseAxis::X));
        ctx.input.bind_axis("look_y", AxisBinding::MouseDelta(MouseAxis::Y));
        ctx.input.bind_action("interact", ActionBinding::Key(KeyCode::KeyE));
        ctx.input.bind_action("save", ActionBinding::Key(KeyCode::F5));
        ctx.input.bind_action("load", ActionBinding::Key(KeyCode::F9));
        ctx.input.bind_action("exit", ActionBinding::Key(KeyCode::Escape));

        // Shared cube mesh — uploaded once, reused by all room builders.
        let cube_data = MeshData::cube(1.0);
        let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &cube_data);
        ctx.assets.register_mesh(cube_mesh);

        // Build rooms
        let (hall_entities, hall_inter) = build_entry_hall(ctx, cube_mesh);
        self.rooms.push(RoomEntities { entities: hall_entities });
        self.interactive.extend(hall_inter);

        let corridor_offset = Vec3::new(0.0, 0.0, -12.0);
        let corr_entities = build_corridor(ctx, corridor_offset, cube_mesh);
        self.rooms.push(RoomEntities { entities: corr_entities });

        let treasure_offset = Vec3::new(0.0, 0.0, -20.0);
        let (treasure_entities, treasure_inter) = build_treasure_room(ctx, treasure_offset, cube_mesh);
        self.rooms.push(RoomEntities { entities: treasure_entities });
        self.interactive.extend(treasure_inter);

        let exit_offset = Vec3::new(0.0, 0.0, -28.0);
        let exit_entities = build_exit_chamber(ctx, exit_offset, cube_mesh);
        self.rooms.push(RoomEntities { entities: exit_entities });

        // Count pickups
        self.total_pickups = self.interactive.iter().filter(|i| matches!(i, Interactive::Pickup { .. })).count() as u32;

        // Camera entity
        let cam_entity = ctx.world.spawn((
            Transform3D { position: self.camera_pos, ..Default::default() },
            GlobalTransform::default(),
            Camera3D { active: true, fov_y: FRAC_PI_4, near: 0.1, far: 100.0 },
        ));
        self.camera_entity = Some(cam_entity);

        // No directional light — this is an enclosed dungeon, CSM shadows from a
        // directional light would just shadow everything below the ceilings.
        // Illumination comes from point/spot lights in each room.

        ctx.renderer.set_postprocess(PostProcess3DConfig {
            bloom_enabled: true,
            bloom_intensity: 0.06,
            bloom_threshold: 2.0,
            bloom_soft_knee: 0.2,
            tone_map_enabled: true,
            ssao_enabled: true,
            motion_blur_enabled: false,
        });

        eprintln!("[dungeon] init complete — {} rooms, {} pickups", self.rooms.len(), self.total_pickups);
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.input.just_pressed("exit") {
            self.exit = true;
            return;
        }

        let dt = ctx.time.tick_dt;

        // ── Mouse look ──
        let look_x = ctx.input.axis("look_x");
        let look_y = ctx.input.axis("look_y");
        let sensitivity = 0.003;

        self.yaw -= look_x * sensitivity;
        self.pitch = (self.pitch - look_y * sensitivity).clamp(-1.4, 1.4);

        // ── Movement (first-person, floor-clamped) ──
        let move_x = ctx.input.axis("move_x");
        let move_z = ctx.input.axis("move_z");
        let move_speed = 5.0;

        let (sin_y, cos_y) = self.yaw.sin_cos();
        let forward = Vec3::new(-sin_y, 0.0, -cos_y);
        let right = Vec3::new(cos_y, 0.0, -sin_y);

        let move_dir = (forward * move_z + right * move_x).normalize_or_zero();

        self.prev_camera_pos = self.camera_pos;
        self.camera_pos += move_dir * move_speed * dt;
        self.camera_pos.y = 1.7; // Floor-clamped eye height

        // ── Interaction ──
        if ctx.input.just_pressed("interact") {
            let interact_range = 3.0;
            for item in &mut self.interactive {
                match item {
                    Interactive::Torch { entity, lit } => {
                        if let Ok(t) = ctx.world.get::<&Transform3D>(*entity) {
                            if t.position.distance(self.camera_pos) < interact_range {
                                *lit = !*lit;
                                if let Ok(mut pl) = ctx.world.get::<&mut PointLightComponent>(*entity) {
                                    pl.intensity = if *lit { 8.0 } else { 0.0 };
                                }
                            }
                        }
                    }
                    Interactive::Pickup { entity, collected } => {
                        if *collected { continue; }
                        if let Ok(t) = ctx.world.get::<&Transform3D>(*entity) {
                            if t.position.distance(self.camera_pos) < interact_range {
                                *collected = true;
                                if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(*entity) {
                                    mr.visible = false;
                                }
                                self.collected += 1;
                                eprintln!("[dungeon] collected {}/{}", self.collected, self.total_pickups);
                            }
                        }
                    }
                }
            }
        }

        // ── Scene save/load ──
        if ctx.input.just_pressed("save") {
            let scene = esox_engine::scene::save_scene(ctx.world, ctx.assets);
            match esox_engine::scene::scene_to_ron(&scene) {
                Ok(ron_str) => {
                    if let Err(e) = std::fs::write("dungeon.scene.ron", &ron_str) {
                        eprintln!("[dungeon] save failed: {e}");
                    } else {
                        eprintln!("[dungeon] scene saved ({} entities)", scene.entities.len());
                    }
                }
                Err(e) => eprintln!("[dungeon] serialize failed: {e}"),
            }
        }

        if ctx.input.just_pressed("load") {
            match std::fs::read_to_string("dungeon.scene.ron") {
                Ok(ron_str) => {
                    match esox_engine::scene::scene_from_ron(&ron_str) {
                        Ok(scene) => {
                            // Clear world
                            let all: Vec<_> = ctx.world.iter().map(|e| e.entity()).collect();
                            for e in all { let _ = ctx.world.despawn(e); }

                            let id_map = esox_engine::scene::load_scene(&scene, ctx.world, ctx.assets, None, None);
                            eprintln!("[dungeon] scene loaded ({} entities)", id_map.len());

                            // Re-find camera entity
                            self.camera_entity = None;
                            for (entity, _cam) in ctx.world.query::<&Camera3D>().iter() {
                                self.camera_entity = Some(entity);
                                if let Ok(t) = ctx.world.get::<&Transform3D>(entity) {
                                    self.camera_pos = t.position;
                                    self.prev_camera_pos = t.position;
                                }
                                break;
                            }

                            // Rebuild interactive list from loaded tags.
                            self.interactive.clear();
                            self.collected = 0;
                            self.total_pickups = 0;
                            for (entity, tag) in ctx.world.query::<&Tag>().iter() {
                                match tag.0.as_str() {
                                    "torch" => {
                                        let lit = ctx.world.get::<&PointLightComponent>(entity)
                                            .map(|pl| pl.intensity > 0.0).unwrap_or(true);
                                        self.interactive.push(Interactive::Torch { entity, lit });
                                    }
                                    "pickup" => {
                                        self.total_pickups += 1;
                                        let collected = ctx.world.get::<&MeshRenderer>(entity)
                                            .map(|mr| !mr.visible).unwrap_or(false);
                                        if collected { self.collected += 1; }
                                        self.interactive.push(Interactive::Pickup { entity, collected });
                                    }
                                    _ => {}
                                }
                            }
                        }
                        Err(e) => eprintln!("[dungeon] deserialize failed: {e}"),
                    }
                }
                Err(e) => eprintln!("[dungeon] load failed: {e}"),
            }
        }
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

    fn ui(&mut self, ui: &mut esox_engine::esox_ui::Ui, ctx: &Ctx) {
        use esox_gfx::Color;

        // Crosshair (center dot)
        let center_y = ctx.viewport.1 as f32 / 2.0;
        let current_y = ui.cursor_y();
        if center_y > current_y {
            ui.add_space(center_y - current_y);
        }
        let crosshair = "+";
        let approx_w = crosshair.len() as f32 * 8.0;
        ui.center_horizontal(approx_w, |ui| {
            ui.label_colored(crosshair, Color::WHITE);
        });

        // Pickup counter (top-left)
        // Reset to top
        // Note: Since ui doesn't support arbitrary positioning easily,
        // we'll put the counter after the crosshair
        ui.padding(16.0, |ui| {
            ui.label_colored(
                &format!("Pickups: {} / {}", self.collected, self.total_pickups),
                Color::WHITE,
            );
        });

        // Interaction prompt
        let interact_range = 3.0;
        let near_interactive = self.interactive.iter().any(|item| {
            let (entity, active) = match item {
                Interactive::Torch { entity, .. } => (entity, true),
                Interactive::Pickup { entity, collected } => (entity, !collected),
            };
            if !active { return false; }
            if let Ok(t) = ctx.world.get::<&Transform3D>(*entity) {
                t.position.distance(self.camera_pos) < interact_range
            } else {
                false
            }
        });

        if near_interactive {
            ui.padding(16.0, |ui| {
                ui.label_colored("[E] Interact", Color::new(0.8, 0.8, 0.8, 1.0));
            });
        }

        // Save/load hints
        ui.padding(16.0, |ui| {
            ui.label_colored("[F5] Save  [F9] Load  [Esc] Quit", Color::new(0.5, 0.5, 0.5, 1.0));
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
                .add_directive("dungeon=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "esox dungeon".into(),
                width: Some(1280),
                height: Some(720),
                ..Default::default()
            },
            msaa: 4,
            ..Default::default()
        },
        // Enable shadows for point/spot lights. CSM shadows for the directional
        // light are still effectively inactive (intensity 0), but the shadow pass
        // infrastructure is needed for the point/spot shadow pipeline.
        shadows: true,
        ..EngineConfig::default()
    };

    let game = Dungeon::new();

    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
