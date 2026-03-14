//! Dungeon Explorer — flagship demo combining glTF loading (with procedural
//! fallback), PBR, physics-interactive crates, spatial audio, fire particles,
//! flickering torches, scene serialization, and editor compatibility.

use esox_engine::*;
use esox_engine::glam::{Quat, Vec3};
use esox_engine::esox_gfx::mesh3d::{
    GltfScene, MaterialDescriptor, MaterialType, MeshData,
    MaterialHandle, ParticlePoolHandle, PostProcess3DConfig,
};

use std::f32::consts::FRAC_PI_4;
use std::path::Path;

// ── Asset loading ──

/// Register all dungeon assets (glTF if present, procedural fallbacks always).
/// After this, `load_scene` can resolve mesh/material names like "stone_cube", "gold", etc.
fn register_dungeon_assets(ctx: &mut Ctx) {
    // Try loading glTF assets from assets/dungeon/
    let asset_dir = Path::new("assets/dungeon");
    if asset_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(asset_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "glb" || e == "gltf") {
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
                    match GltfScene::load(&path) {
                        Ok(scene) => {
                            let handles = ctx.renderer.upload_gltf_scene(ctx.gpu, scene);
                            if let (Some(&mesh), Some(&mat)) =
                                (handles.meshes.first(), handles.materials.first())
                            {
                                ctx.assets.register_mesh_named(format!("gltf_{name}"), mesh);
                                ctx.assets.register_material_named(format!("gltf_{name}_mat"), mat);
                                eprintln!("[dungeon_explorer] loaded glTF asset: {name}");
                            }
                        }
                        Err(e) => eprintln!("[dungeon_explorer] failed to load {}: {e}", path.display()),
                    }
                }
            }
        }
    }

    // Generate procedural fallback meshes + materials
    let cube_data = MeshData::cube(1.0);
    let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &cube_data);
    ctx.assets.register_mesh_named("stone_cube", cube_mesh);

    let sphere_data = MeshData::sphere(0.3, 16, 12);
    let sphere_mesh = ctx.renderer.upload_mesh(ctx.gpu, &sphere_data);
    ctx.assets.register_mesh_named("gold_sphere", sphere_mesh);

    let stone_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.35, 0.32, 0.28, 1.0],
        roughness: 0.95,
        metallic: 0.0,
        ..MaterialDescriptor::default()
    });
    ctx.assets.register_material_named("stone", stone_mat);

    let stone_dark = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.25, 0.22, 0.2, 1.0],
        roughness: 0.9,
        metallic: 0.05,
        ..MaterialDescriptor::default()
    });
    ctx.assets.register_material_named("stone_dark", stone_dark);

    let stone_gold = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.3, 0.28, 0.22, 1.0],
        roughness: 0.85,
        metallic: 0.1,
        ..MaterialDescriptor::default()
    });
    ctx.assets.register_material_named("stone_gold", stone_gold);

    let gold_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [1.0, 0.85, 0.2, 1.0],
        roughness: 0.2,
        metallic: 0.95,
        emissive: [0.05, 0.04, 0.01],
        ..MaterialDescriptor::default()
    });
    ctx.assets.register_material_named("gold", gold_mat);

    let wood_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.55, 0.35, 0.18, 1.0],
        roughness: 0.75,
        metallic: 0.0,
        ..MaterialDescriptor::default()
    });
    ctx.assets.register_material_named("wood", wood_mat);

    let torch_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
        material_type: MaterialType::PBR,
        albedo: [0.15, 0.12, 0.1, 1.0],
        roughness: 0.9,
        metallic: 0.0,
        ..MaterialDescriptor::default()
    });
    ctx.assets.register_material_named("torch_mat", torch_mat);
}

// ── Interactive objects ──

#[derive(Clone)]
enum Interactive {
    Torch { entity: hecs::Entity, lit: bool },
    Pickup { entity: hecs::Entity, collected: bool },
}

// ── Game state ──

struct DungeonExplorer {
    yaw: f32,
    pitch: f32,
    camera_pos: Vec3,
    prev_camera_pos: Vec3,
    camera_entity: Option<hecs::Entity>,
    interactive: Vec<Interactive>,
    torch_phases: Vec<(hecs::Entity, f32)>,
    collected: u32,
    total_pickups: u32,
    footstep_timer: f32,
    fire_pool: Option<ParticlePoolHandle>,
    particle_mat: Option<MaterialHandle>,
    audio_handles: AudioHandles,
    exit: bool,
}

#[allow(dead_code)]
struct AudioHandles {
    footstep_sfx: Option<audio::spatial::SoundHandle>,
    ambient_sfx: Option<audio::spatial::SoundHandle>,
    torch_sfx: Option<audio::spatial::SoundHandle>,
    torch_spatial: Vec<audio::spatial::SpatialSoundHandle>,
}

impl Default for AudioHandles {
    fn default() -> Self {
        Self {
            footstep_sfx: None,
            ambient_sfx: None,
            torch_sfx: None,
            torch_spatial: Vec::new(),
        }
    }
}

impl DungeonExplorer {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            camera_pos: Vec3::new(0.0, 1.7, -2.0),
            prev_camera_pos: Vec3::new(0.0, 1.7, -2.0),
            camera_entity: None,
            interactive: Vec::new(),
            torch_phases: Vec::new(),
            collected: 0,
            total_pickups: 0,
            footstep_timer: 0.0,
            fire_pool: None,
            particle_mat: None,
            audio_handles: AudioHandles::default(),
            exit: false,
        }
    }
}

impl Game for DungeonExplorer {
    fn init(&mut self, ctx: &mut Ctx) {
        use esox_engine::winit::keyboard::KeyCode;

        // ── Input bindings ──
        ctx.input.bind_axis("move_x", AxisBinding::Keys { negative: KeyCode::KeyA, positive: KeyCode::KeyD });
        ctx.input.bind_axis("move_z", AxisBinding::Keys { negative: KeyCode::KeyS, positive: KeyCode::KeyW });
        ctx.input.bind_axis("look_x", AxisBinding::MouseDelta(MouseAxis::X));
        ctx.input.bind_axis("look_y", AxisBinding::MouseDelta(MouseAxis::Y));
        ctx.input.bind_action("interact", ActionBinding::Key(KeyCode::KeyE));
        ctx.input.bind_action("save", ActionBinding::Key(KeyCode::F5));
        ctx.input.bind_action("load", ActionBinding::Key(KeyCode::F9));
        ctx.input.bind_action("exit", ActionBinding::Key(KeyCode::Escape));

        // ── Register assets ──
        register_dungeon_assets(ctx);

        // ── Particle setup ──
        let particle_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::Unlit,
            albedo: [1.0, 1.0, 1.0, 1.0],
            ..MaterialDescriptor::default()
        });
        self.particle_mat = Some(particle_mat);
        let fire_pool = ctx.renderer.create_particle_pool(ctx.gpu, 1024);
        self.fire_pool = Some(fire_pool);

        // ── Load scene from .ron ──
        let scene_path = "examples/dungeon_explorer/dungeon_explorer.scene.ron";
        match std::fs::read_to_string(scene_path) {
            Ok(ron_str) => {
                match esox_engine::scene::scene_from_ron(&ron_str) {
                    Ok(scene) => {
                        let id_map = esox_engine::scene::load_scene(
                            &scene, ctx.world, ctx.assets,
                            Some(ctx.physics), Some(ctx.entity_map),
                        );
                        eprintln!("[dungeon_explorer] scene loaded ({} entities)", id_map.len());
                        self.post_load_fixup(ctx, fire_pool, particle_mat);
                    }
                    Err(e) => {
                        eprintln!("[dungeon_explorer] scene parse failed: {e}");
                        eprintln!("[dungeon_explorer] building fallback scene");
                        self.build_fallback_scene(ctx);
                    }
                }
            }
            Err(e) => {
                eprintln!("[dungeon_explorer] scene load failed: {e}");
                eprintln!("[dungeon_explorer] building fallback scene");
                self.build_fallback_scene(ctx);
            }
        }

        // ── Post-processing ──
        ctx.renderer.set_postprocess(PostProcess3DConfig {
            bloom_enabled: true,
            bloom_intensity: 0.08,
            bloom_threshold: 2.0,
            bloom_soft_knee: 0.2,
            tone_map_enabled: true,
            ssao_enabled: true,
            motion_blur_enabled: false,
        });

        // ── Audio ──
        if let Some(ref mut audio) = ctx.audio {
            self.audio_handles.footstep_sfx = audio.load("assets/audio/footstep.ogg").ok();
            self.audio_handles.ambient_sfx = audio.load("assets/audio/ambient.ogg").ok();
            self.audio_handles.torch_sfx = audio.load("assets/audio/fire_crackle.ogg").ok();

            // Play ambient music if available
            if let Some(ambient) = self.audio_handles.ambient_sfx {
                let _ = audio.play_music(ambient, 2.0);
            }

            // Play spatial torch crackles
            if let Some(torch_sfx) = self.audio_handles.torch_sfx {
                for (entity, _phase) in &self.torch_phases {
                    if let Ok(t) = ctx.world.get::<&Transform3D>(*entity) {
                        if let Some(handle) = audio.play_spatial(torch_sfx, t.position, 15.0) {
                            self.audio_handles.torch_spatial.push(handle);
                        }
                    }
                }
            }
        }

        eprintln!(
            "[dungeon_explorer] init complete — {} interactives, {} pickups, {} torches",
            self.interactive.len(), self.total_pickups, self.torch_phases.len()
        );
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.input.just_pressed("exit") {
            self.exit = true;
            return;
        }

        let dt = ctx.time.tick_dt;
        let elapsed = ctx.time.elapsed;

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
        let moving = move_dir.length_squared() > 0.01;

        self.prev_camera_pos = self.camera_pos;
        self.camera_pos += move_dir * move_speed * dt;
        self.camera_pos.y = 1.7;

        // ── Footstep audio ──
        if moving {
            self.footstep_timer -= dt;
            if self.footstep_timer <= 0.0 {
                self.footstep_timer = 0.4;
                if let Some(ref mut audio) = ctx.audio {
                    if let Some(sfx) = self.audio_handles.footstep_sfx {
                        audio.play_at_volume(sfx, 0.3);
                    }
                }
            }
        } else {
            self.footstep_timer = 0.0;
        }

        // ── Update listener ──
        if let Some(ref mut audio) = ctx.audio {
            let (sin_y, cos_y) = self.yaw.sin_cos();
            let fwd = Vec3::new(-sin_y, 0.0, -cos_y);
            audio.set_listener(self.camera_pos, fwd, Vec3::Y);
        }

        // ── Torch flicker ──
        for (entity, phase) in &self.torch_phases {
            let flicker = 1.0
                + 0.15 * (elapsed * 12.0 + *phase).sin()
                + 0.08 * (elapsed * 7.3 + *phase * 2.7).sin()
                + 0.05 * (elapsed * 23.0 + *phase * 0.4).sin();
            let base_intensity = 12.0;
            if let Ok(mut pl) = ctx.world.get::<&mut PointLightComponent>(*entity) {
                if pl.intensity > 0.5 {
                    pl.intensity = base_intensity * flicker;
                }
            }
        }

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
                                    pl.intensity = if *lit { 12.0 } else { 0.0 };
                                }
                                if let Ok(mut pe) = ctx.world.get::<&mut ParticleEmitter>(*entity) {
                                    pe.active = *lit;
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
                                eprintln!("[dungeon_explorer] collected {}/{}", self.collected, self.total_pickups);
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
                    if let Err(e) = std::fs::write("dungeon_explorer.scene.ron", &ron_str) {
                        eprintln!("[dungeon_explorer] save failed: {e}");
                    } else {
                        eprintln!("[dungeon_explorer] scene saved ({} entities)", scene.entities.len());
                    }
                }
                Err(e) => eprintln!("[dungeon_explorer] serialize failed: {e}"),
            }
        }

        if ctx.input.just_pressed("load") {
            match std::fs::read_to_string("dungeon_explorer.scene.ron") {
                Ok(ron_str) => {
                    match esox_engine::scene::scene_from_ron(&ron_str) {
                        Ok(scene) => {
                            // Clear world
                            let all: Vec<_> = ctx.world.iter().map(|e| e.entity()).collect();
                            for e in all { let _ = ctx.world.despawn(e); }

                            let id_map = esox_engine::scene::load_scene(
                                &scene, ctx.world, ctx.assets,
                                Some(ctx.physics), Some(ctx.entity_map),
                            );
                            eprintln!("[dungeon_explorer] scene loaded ({} entities)", id_map.len());

                            if let (Some(pool), Some(mat)) = (self.fire_pool, self.particle_mat) {
                                self.post_load_fixup(ctx, pool, mat);
                            }
                        }
                        Err(e) => eprintln!("[dungeon_explorer] deserialize failed: {e}"),
                    }
                }
                Err(e) => eprintln!("[dungeon_explorer] load failed: {e}"),
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

        // FPS counter
        let fps = if ctx.time.frame_dt > 0.0 { (1.0 / ctx.time.frame_dt) as u32 } else { 0 };
        ui.padding(8.0, |ui| {
            ui.label_colored(&format!("FPS: {fps}"), Color::new(0.6, 0.6, 0.6, 1.0));
        });

        // Crosshair (center)
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

        // Pickup counter
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

        // Controls hint
        ui.padding(16.0, |ui| {
            ui.label_colored(
                "WASD: Move  Mouse: Look  [F5] Save  [F9] Load  [Esc] Quit",
                Color::new(0.4, 0.4, 0.4, 1.0),
            );
        });
    }

    fn should_exit(&self) -> bool {
        self.exit
    }
}

impl DungeonExplorer {
    /// Post-load fixup: rebuild interactive list, find camera, fix particle handles.
    fn post_load_fixup(
        &mut self,
        ctx: &mut Ctx,
        fire_pool: ParticlePoolHandle,
        particle_mat: MaterialHandle,
    ) {
        self.interactive.clear();
        self.torch_phases.clear();
        self.collected = 0;
        self.total_pickups = 0;
        self.camera_entity = None;

        // Fix up particle emitters with real GPU handles
        let emitter_entities: Vec<hecs::Entity> = ctx.world
            .query::<&ParticleEmitter>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        for entity in emitter_entities {
            if let Ok(mut pe) = ctx.world.get::<&mut ParticleEmitter>(entity) {
                pe.pool = fire_pool;
                pe.material = particle_mat;
            }
        }

        // Rebuild interactive list and find camera from tags
        let tagged: Vec<(hecs::Entity, String)> = ctx.world
            .query::<&Tag>()
            .iter()
            .map(|(e, t)| (e, t.0.clone()))
            .collect();

        let mut phase_counter = 0.0_f32;
        for (entity, tag) in &tagged {
            match tag.as_str() {
                "camera" => {
                    self.camera_entity = Some(*entity);
                    if let Ok(t) = ctx.world.get::<&Transform3D>(*entity) {
                        self.camera_pos = t.position;
                        self.prev_camera_pos = t.position;
                    }
                }
                "torch" => {
                    let lit = ctx.world.get::<&PointLightComponent>(*entity)
                        .map(|pl| pl.intensity > 0.5).unwrap_or(true);
                    self.interactive.push(Interactive::Torch { entity: *entity, lit });
                    self.torch_phases.push((*entity, phase_counter));
                    phase_counter += 1.7; // stagger phases
                }
                "pickup" => {
                    self.total_pickups += 1;
                    let collected = ctx.world.get::<&MeshRenderer>(*entity)
                        .map(|mr| !mr.visible).unwrap_or(false);
                    if collected { self.collected += 1; }
                    self.interactive.push(Interactive::Pickup { entity: *entity, collected });
                }
                _ => {}
            }
        }

        // If no camera entity found in scene, spawn one
        if self.camera_entity.is_none() {
            let cam_entity = ctx.world.spawn((
                Transform3D { position: self.camera_pos, ..Default::default() },
                GlobalTransform::default(),
                Camera3D { active: true, fov_y: FRAC_PI_4, near: 0.1, far: 100.0 },
            ));
            self.camera_entity = Some(cam_entity);
        }
    }

    /// Build a minimal procedural scene if .ron loading fails.
    fn build_fallback_scene(&mut self, ctx: &mut Ctx) {
        let cube_data = MeshData::cube(1.0);
        let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &cube_data);
        ctx.assets.register_mesh(cube_mesh);

        let stone_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.35, 0.32, 0.28, 1.0],
            roughness: 0.95,
            metallic: 0.0,
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

        // Simple room: floor, ceiling, 4 walls
        let room_w = 10.0_f32;
        let room_h = 5.0_f32;
        let room_d = 12.0_f32;

        // Floor
        ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, 0.0, -room_d / 2.0), scale: Vec3::new(room_w, 0.2, room_d), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
        ));
        // Ceiling
        ctx.world.spawn((
            Transform3D { position: Vec3::new(0.0, room_h, -room_d / 2.0), scale: Vec3::new(room_w, 0.2, room_d), ..Default::default() },
            GlobalTransform::default(),
            MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
        ));
        // Walls
        for &(pos, scale) in &[
            (Vec3::new(-room_w / 2.0, room_h / 2.0, -room_d / 2.0), Vec3::new(0.3, room_h, room_d)),
            (Vec3::new(room_w / 2.0, room_h / 2.0, -room_d / 2.0), Vec3::new(0.3, room_h, room_d)),
            (Vec3::new(0.0, room_h / 2.0, -room_d), Vec3::new(room_w, room_h, 0.3)),
            (Vec3::new(0.0, room_h / 2.0, 0.0), Vec3::new(room_w, room_h, 0.3)),
        ] {
            ctx.world.spawn((
                Transform3D { position: pos, scale, ..Default::default() },
                GlobalTransform::default(),
                MeshRenderer { mesh: cube_mesh, material: stone_mat, tint: [1.0; 4], visible: true },
            ));
        }

        // Torches
        let torch_positions = [
            Vec3::new(-room_w / 2.0 + 0.5, 3.0, -room_d / 4.0),
            Vec3::new(room_w / 2.0 - 0.5, 3.0, -room_d * 3.0 / 4.0),
        ];
        let mut phase = 0.0_f32;
        for pos in torch_positions {
            let torch_entity = ctx.world.spawn((
                Transform3D { position: pos, ..Default::default() },
                GlobalTransform::default(),
                PointLightComponent { color: [1.0, 0.7, 0.3], intensity: 12.0, range: 15.0, cast_shadows: true },
                Tag("torch".into()),
            ));
            self.interactive.push(Interactive::Torch { entity: torch_entity, lit: true });
            self.torch_phases.push((torch_entity, phase));
            phase += 1.7;
        }

        // Pickups
        let pickup_positions = [
            Vec3::new(-2.0, 0.8, -room_d / 2.0),
            Vec3::new(0.0, 0.8, -room_d * 0.7),
            Vec3::new(2.0, 0.8, -room_d / 2.0),
        ];
        for pos in pickup_positions {
            let pickup_entity = ctx.world.spawn((
                Transform3D { position: pos, ..Default::default() },
                GlobalTransform::default(),
                MeshRenderer { mesh: sphere_mesh, material: gold_mat, tint: [1.0; 4], visible: true },
                Tag("pickup".into()),
            ));
            self.interactive.push(Interactive::Pickup { entity: pickup_entity, collected: false });
        }
        self.total_pickups = pickup_positions.len() as u32;

        // Camera
        let cam_entity = ctx.world.spawn((
            Transform3D { position: self.camera_pos, ..Default::default() },
            GlobalTransform::default(),
            Camera3D { active: true, fov_y: FRAC_PI_4, near: 0.1, far: 100.0 },
        ));
        self.camera_entity = Some(cam_entity);
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("dungeon_explorer=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "esox dungeon explorer".into(),
                width: Some(1280),
                height: Some(720),
                ..Default::default()
            },
            msaa: 4,
            ..Default::default()
        },
        shadows: true,
        physics: Some(Box::new(RapierPhysics::new(Vec3::new(0.0, -9.81, 0.0)))),
        ..EngineConfig::default()
    };

    let game = DungeonExplorer::new();

    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
