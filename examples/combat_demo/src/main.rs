//! Combat Demo — knight vs spiders with animation-graph-driven combat.
//!
//! WASD to move, Space to attack, Escape to exit.
//! Spiders chase the knight and die on hit.

use esox_engine::*;
use esox_engine::glam::{Mat3, Quat, Vec3, Vec3Swizzles};
use esox_gfx::mesh3d::{
    AnimationClip, AnimationPlayer, GltfScene, GltfSceneHandles, MaterialDescriptor, MaterialType,
    MeshData, MeshHandle, MaterialHandle, PostProcess3DConfig, ShadowConfig,
};

use std::f32::consts::FRAC_PI_4;
use std::path::Path;

// ── Constants ──

const KNIGHT_SPEED: f32 = 6.0;
const SPIDER_SPEED: f32 = 3.5;
const ATTACK_RADIUS: f32 = 2.5;
const ATTACK_DURATION: f32 = 0.6;
const SPAWN_INTERVAL: f32 = 4.0;
const ARENA_RADIUS: f32 = 20.0;
const MAX_ENEMIES: usize = 12;

// ── Components ──

struct Player;
struct Enemy {
    health: f32,
}
struct DeathTimer(f32);

/// One primitive of a multi-material skinned model.
#[derive(Clone)]
struct PrimitivePart {
    mesh: MeshHandle,
    material: MaterialHandle,
    skinned_mesh_index: Option<usize>,
}

// ── Game State ──

struct CombatDemo {
    exit: bool,
    attack_timer: f32,
    spawn_timer: f32,
    kills: u32,
    // Spider model data for runtime spawning.
    spider_parts: Vec<PrimitivePart>,
    spider_clips: Vec<AnimationClip>,
    spider_skin: Option<esox_gfx::mesh3d::gltf_loader::GltfSkin>,
    spider_anim_graph: Option<AnimGraphDef>,
}

/// Extract per-primitive (mesh, material, skinned_index) tuples from uploaded glTF handles.
fn collect_parts(handles: &GltfSceneHandles) -> Vec<PrimitivePart> {
    handles
        .meshes
        .iter()
        .enumerate()
        .map(|(i, &mesh)| {
            let mat_idx = handles.mesh_material_indices[i].unwrap_or(0);
            let material = handles.materials[mat_idx];
            PrimitivePart {
                mesh,
                material,
                skinned_mesh_index: handles.skinned_mesh_indices[i],
            }
        })
        .collect()
}

/// Spawn a multi-primitive skinned entity. The first primitive gets the
/// AnimGraphController; the remaining primitives are spawned as child entities
/// that inherit the parent's transform. All skinned indices share joint matrices
/// via `extra_skinned_indices`.
fn spawn_skinned_entity(
    ctx: &mut Ctx,
    parts: &[PrimitivePart],
    skin: &esox_gfx::mesh3d::gltf_loader::GltfSkin,
    anim_graph_def: &AnimGraphDef,
    clips: &[AnimationClip],
    position: Vec3,
    scale: Vec3,
    extra_components: impl FnOnce(hecs::Entity, &mut hecs::World),
) -> hecs::Entity {
    let player = AnimationPlayer::new(skin);
    let graph = AnimGraphRuntime::new(anim_graph_def.clone(), player);

    let primary = &parts[0];
    let primary_skinned = primary.skinned_mesh_index.unwrap_or(0);
    let extra_skinned: Vec<usize> = parts[1..]
        .iter()
        .filter_map(|p| p.skinned_mesh_index)
        .collect();

    let parent = ctx.world.spawn((
        Transform3D {
            position,
            scale,
            ..Transform3D::default()
        },
        GlobalTransform::default(),
        MeshRenderer {
            mesh: primary.mesh,
            material: primary.material,
            tint: [1.0; 4],
            visible: true,
        },
        AnimGraphController {
            graph,
            clips: clips.to_vec(),
            skinned_mesh_index: primary_skinned,
            extra_skinned_indices: extra_skinned,
        },
    ));

    // Spawn child entities for additional primitives.
    let mut child_ids = Vec::new();
    for part in &parts[1..] {
        let child = ctx.world.spawn((
            Transform3D::default(), // identity — inherits parent transform
            GlobalTransform::default(),
            MeshRenderer {
                mesh: part.mesh,
                material: part.material,
                tint: [1.0; 4],
                visible: true,
            },
            Parent(parent),
        ));
        child_ids.push(child);
    }

    if !child_ids.is_empty() {
        let _ = ctx.world.insert_one(parent, Children(child_ids));
    }

    extra_components(parent, &mut ctx.world);
    parent
}

impl Game for CombatDemo {
    fn init(&mut self, ctx: &mut Ctx) {
        // ── Input bindings ──
        ctx.input.bind_action("exit", ActionBinding::Key(esox_engine::esox_input::KeyCode::Escape));
        ctx.input.bind_action("attack", ActionBinding::Key(esox_engine::esox_input::KeyCode::Space));
        ctx.input.bind_axis("move_x", AxisBinding::Keys {
            negative: esox_engine::esox_input::KeyCode::KeyA,
            positive: esox_engine::esox_input::KeyCode::KeyD,
        });
        ctx.input.bind_axis("move_z", AxisBinding::Keys {
            negative: esox_engine::esox_input::KeyCode::KeyS,
            positive: esox_engine::esox_input::KeyCode::KeyW,
        });

        // ── Load Knight ──
        let knight_scene = GltfScene::load(Path::new("assets/knight/KnightCharacter.glb"))
            .expect("failed to load KnightCharacter.glb");
        let knight_handles = ctx.renderer.upload_gltf_scene(ctx.gpu, knight_scene);

        // Log what we got for debugging.
        tracing::info!(
            "Knight: {} meshes, {} materials, {} skins, {} animations",
            knight_handles.meshes.len(),
            knight_handles.materials.len(),
            knight_handles.skins.len(),
            knight_handles.animations.len(),
        );
        for (i, clip) in knight_handles.animations.iter().enumerate() {
            tracing::info!("  anim[{i}]: {:?}", clip.name);
        }

        // Find animation clip indices.
        let mut idle_idx = 0;
        let mut run_idx = 0;
        let mut attack_idx = 0;
        for (i, clip) in knight_handles.animations.iter().enumerate() {
            match clip.name.as_deref() {
                Some(n) if n.contains("Idle") && !n.contains("sword") => idle_idx = i,
                Some(n) if n.contains("Run") && !n.contains("sword") && !n.contains("Attack") => run_idx = i,
                Some(n) if n.contains("swordAttackJump") => attack_idx = i,
                _ => {}
            }
        }

        let knight_anim_graph = AnimGraphDef {
            states: vec![
                AnimState {
                    name: "Idle".into(),
                    source: StateSource::Clip { clip_index: idle_idx },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![
                        Transition {
                            target_state: 1,
                            conditions: vec![Condition::FloatGt { param: "speed".into(), threshold: 0.1 }],
                            duration: 0.15,
                            priority: 0,
                        },
                        Transition {
                            target_state: 2,
                            conditions: vec![Condition::BoolTrue { param: "attacking".into() }],
                            duration: 0.1,
                            priority: 1,
                        },
                    ],
                    events: vec![],
                },
                AnimState {
                    name: "Run".into(),
                    source: StateSource::Clip { clip_index: run_idx },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![
                        Transition {
                            target_state: 0,
                            conditions: vec![Condition::FloatLt { param: "speed".into(), threshold: 0.1 }],
                            duration: 0.15,
                            priority: 0,
                        },
                        Transition {
                            target_state: 2,
                            conditions: vec![Condition::BoolTrue { param: "attacking".into() }],
                            duration: 0.1,
                            priority: 1,
                        },
                    ],
                    events: vec![],
                },
                AnimState {
                    name: "Attack".into(),
                    source: StateSource::Clip { clip_index: attack_idx },
                    looping: false,
                    speed: 1.5,
                    transitions: vec![Transition {
                        target_state: 0,
                        conditions: vec![Condition::BoolFalse { param: "attacking".into() }],
                        duration: 0.15,
                        priority: 0,
                    }],
                    events: vec![],
                },
            ],
            default_state: 0,
        };

        let knight_parts = collect_parts(&knight_handles);
        spawn_skinned_entity(
            ctx,
            &knight_parts,
            &knight_handles.skins[0],
            &knight_anim_graph,
            &knight_handles.animations,
            Vec3::ZERO,
            Vec3::splat(1.0),
            |entity, world| {
                let _ = world.insert_one(entity, Player);
            },
        );

        // ── Load Spider ──
        let spider_scene = GltfScene::load(Path::new("assets/enemies/Spider.glb"))
            .expect("failed to load Spider.glb");
        let spider_handles = ctx.renderer.upload_gltf_scene(ctx.gpu, spider_scene);

        tracing::info!(
            "Spider: {} meshes, {} materials, {} skins, {} animations",
            spider_handles.meshes.len(),
            spider_handles.materials.len(),
            spider_handles.skins.len(),
            spider_handles.animations.len(),
        );

        let mut sp_idle_idx = 0;
        let mut sp_walk_idx = 0;
        let mut sp_death_idx = 0;
        for (i, clip) in spider_handles.animations.iter().enumerate() {
            tracing::info!("  anim[{i}]: {:?}", clip.name);
            match clip.name.as_deref() {
                Some(n) if n.contains("Idle") => sp_idle_idx = i,
                Some(n) if n.contains("Walk") => sp_walk_idx = i,
                Some(n) if n.contains("Death") => sp_death_idx = i,
                _ => {}
            }
        }

        self.spider_anim_graph = Some(AnimGraphDef {
            states: vec![
                AnimState {
                    name: "Idle".into(),
                    source: StateSource::Clip { clip_index: sp_idle_idx },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 1,
                        conditions: vec![Condition::FloatGt { param: "speed".into(), threshold: 0.1 }],
                        duration: 0.2,
                        priority: 0,
                    }],
                    events: vec![],
                },
                AnimState {
                    name: "Walk".into(),
                    source: StateSource::Clip { clip_index: sp_walk_idx },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![
                        Transition {
                            target_state: 0,
                            conditions: vec![Condition::FloatLt { param: "speed".into(), threshold: 0.1 }],
                            duration: 0.2,
                            priority: 0,
                        },
                        Transition {
                            target_state: 2,
                            conditions: vec![Condition::BoolTrue { param: "dying".into() }],
                            duration: 0.1,
                            priority: 1,
                        },
                    ],
                    events: vec![],
                },
                AnimState {
                    name: "Death".into(),
                    source: StateSource::Clip { clip_index: sp_death_idx },
                    looping: false,
                    speed: 1.0,
                    transitions: vec![],
                    events: vec![],
                },
            ],
            default_state: 0,
        });

        self.spider_parts = collect_parts(&spider_handles);
        self.spider_clips = spider_handles.animations.clone();
        self.spider_skin = Some(spider_handles.skins[0].clone());

        // Spawn initial enemies.
        for i in 0..4 {
            let angle = (i as f32 / 4.0) * std::f32::consts::TAU;
            let pos = Vec3::new(angle.cos() * 12.0, 0.0, angle.sin() * 12.0);
            self.spawn_enemy(ctx, pos);
        }

        // ── Ground plane ──
        let ground = MeshData::plane(ARENA_RADIUS * 2.5, ARENA_RADIUS * 2.5, 1);
        let ground_mesh = ctx.renderer.upload_mesh(ctx.gpu, &ground);
        let ground_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.25, 0.3, 0.2, 1.0],
            roughness: 0.95,
            metallic: 0.0,
            ..MaterialDescriptor::default()
        });
        ctx.world.spawn((
            Transform3D::default(),
            GlobalTransform::default(),
            MeshRenderer {
                mesh: ground_mesh,
                material: ground_mat,
                tint: [1.0; 4],
                visible: true,
            },
        ));

        // ── Sun ──
        ctx.world.spawn((
            Transform3D {
                rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_3),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            DirectionalLightComponent {
                color: [1.0, 0.95, 0.85],
                intensity: 2.5,
            },
        ));

        // ── Camera ──
        let cam_pos = Vec3::new(0.0, 18.0, -14.0);
        let cam_target = Vec3::ZERO;
        ctx.world.spawn((
            Transform3D {
                position: cam_pos,
                rotation: look_at_quat(cam_pos, cam_target),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            Camera3D {
                active: true,
                fov_y: FRAC_PI_4,
                near: 0.1,
                far: 200.0,
            },
        ));

        // ── Rendering setup ──
        ctx.renderer.generate_procedural_ibl(ctx.gpu);
        ctx.renderer.set_shadow_config(ShadowConfig {
            shadow_distance: 50.0,
            depth_bias: 0.003,
            normal_bias: 0.03,
            ..ShadowConfig::default()
        });
        ctx.renderer.set_postprocess(PostProcess3DConfig {
            bloom_enabled: true,
            bloom_intensity: 0.04,
            bloom_threshold: 3.5,
            bloom_soft_knee: 0.1,
            tone_map_enabled: true,
            ssao_enabled: true,
            fog_enabled: false,
            fog_color: [0.75, 0.82, 0.90],
            fog_start: 50.0,
            fog_end: 200.0,
        });
    }

    fn update(&mut self, ctx: &mut Ctx) {
        let dt = ctx.time.tick_dt;

        if ctx.input.just_pressed("exit") {
            self.exit = true;
            return;
        }

        // ── Player movement ──
        let move_x = ctx.input.axis("move_x");
        let move_z = ctx.input.axis("move_z");
        let attacking = ctx.input.just_pressed("attack");

        if attacking && self.attack_timer <= 0.0 {
            self.attack_timer = ATTACK_DURATION;
        }
        if self.attack_timer > 0.0 {
            self.attack_timer -= dt;
        }

        let mut player_pos = Vec3::ZERO;

        for (_e, (transform, anim, _player)) in ctx
            .world
            .query_mut::<(&mut Transform3D, &mut AnimGraphController, &Player)>()
        {
            let input_dir = Vec3::new(move_x, 0.0, move_z);
            let speed = input_dir.length().min(1.0);

            if speed > 0.01 {
                let dir = input_dir.normalize();
                transform.position += dir * KNIGHT_SPEED * dt;
                transform.rotation = look_at_quat_dir(dir);
            }

            // Clamp to arena.
            let dist = transform.position.xz().length();
            if dist > ARENA_RADIUS {
                let clamped = transform.position.xz().normalize() * ARENA_RADIUS;
                transform.position.x = clamped.x;
                transform.position.z = clamped.y;
            }

            anim.graph.params.set_float("speed", speed);
            anim.graph.params.set_bool("attacking", self.attack_timer > ATTACK_DURATION * 0.5);

            player_pos = transform.position;
        }

        // ── Attack hit detection ──
        if self.attack_timer > ATTACK_DURATION * 0.3 && self.attack_timer < ATTACK_DURATION * 0.7 {
            let mut killed = Vec::new();
            for (e, (transform, enemy, anim)) in ctx
                .world
                .query_mut::<(&Transform3D, &mut Enemy, &mut AnimGraphController)>()
            {
                let dist = (transform.position - player_pos).length();
                if dist < ATTACK_RADIUS && enemy.health > 0.0 {
                    enemy.health = 0.0;
                    anim.graph.params.set_bool("dying", true);
                    anim.graph.params.set_float("speed", 0.0);
                    killed.push(e);
                }
            }
            for e in &killed {
                let _ = ctx.world.insert_one(*e, DeathTimer(1.5));
                self.kills += 1;
            }
        }

        // ── Enemy AI: chase player ──
        for (_e, (transform, enemy, anim)) in ctx
            .world
            .query_mut::<(&mut Transform3D, &Enemy, &mut AnimGraphController)>()
        {
            if enemy.health <= 0.0 {
                continue;
            }
            let dir = player_pos - transform.position;
            let dist = dir.length();
            if dist > 1.5 {
                let dir_n = dir / dist;
                transform.position += dir_n * SPIDER_SPEED * dt;
                transform.rotation = look_at_quat_dir(dir_n);
                anim.graph.params.set_float("speed", 1.0);
            } else {
                anim.graph.params.set_float("speed", 0.0);
            }
        }

        // ── Remove dead enemies after death animation ──
        // Also despawn their child entities.
        let mut to_remove = Vec::new();
        for (e, timer) in ctx.world.query_mut::<&mut DeathTimer>() {
            timer.0 -= dt;
            if timer.0 <= 0.0 {
                to_remove.push(e);
            }
        }
        for e in to_remove {
            // Collect child IDs before mutating.
            let child_ids: Vec<_> = ctx
                .world
                .get::<&Children>(e)
                .map(|c| c.0.clone())
                .unwrap_or_default();
            for c in child_ids {
                let _ = ctx.world.despawn(c);
            }
            let _ = ctx.world.despawn(e);
        }

        // ── Spawn new enemies ──
        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 {
            self.spawn_timer = SPAWN_INTERVAL;
            let enemy_count = ctx.world.query::<&Enemy>().iter().count();
            if enemy_count < MAX_ENEMIES {
                let angle = (self.kills as f32 * 1.618) % std::f32::consts::TAU;
                let pos = Vec3::new(
                    angle.cos() * ARENA_RADIUS * 0.9,
                    0.0,
                    angle.sin() * ARENA_RADIUS * 0.9,
                );
                self.spawn_enemy(ctx, pos);
            }
        }

        // ── Camera follow ──
        for (_e, (transform, _cam)) in ctx
            .world
            .query_mut::<(&mut Transform3D, &Camera3D)>()
        {
            let target_cam = player_pos + Vec3::new(0.0, 18.0, -14.0);
            transform.position = transform.position.lerp(target_cam, 4.0 * dt);
            transform.rotation = look_at_quat(transform.position, player_pos);
        }
    }

    fn ui(&mut self, ui: &mut esox_engine::esox_ui::Ui, _ctx: &Ctx) {
        use esox_gfx::Color;

        ui.padding(16.0, |ui| {
            ui.label_colored("Combat Demo", Color::WHITE);
            ui.add_space(4.0);
            ui.label_colored(
                &format!("Kills: {}  |  WASD move, Space attack, Esc exit", self.kills),
                Color::new(0.8, 0.8, 0.8, 1.0),
            );
        });
    }

    fn should_exit(&self) -> bool {
        self.exit
    }
}

impl CombatDemo {
    fn spawn_enemy(&self, ctx: &mut Ctx, position: Vec3) {
        let skin = match &self.spider_skin {
            Some(s) => s,
            None => return,
        };
        let graph_def = match &self.spider_anim_graph {
            Some(g) => g,
            None => return,
        };
        if self.spider_parts.is_empty() {
            return;
        }

        spawn_skinned_entity(
            ctx,
            &self.spider_parts,
            skin,
            graph_def,
            &self.spider_clips,
            position,
            Vec3::splat(1.0),
            |entity, world| {
                let _ = world.insert_one(entity, Enemy { health: 1.0 });
            },
        );
    }
}

fn look_at_quat(eye: Vec3, target: Vec3) -> Quat {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    Quat::from_mat3(&Mat3::from_cols(right, up, -forward))
}

fn look_at_quat_dir(dir: Vec3) -> Quat {
    let forward = Vec3::new(dir.x, 0.0, dir.z).normalize();
    if forward.length_squared() < 1e-6 {
        return Quat::IDENTITY;
    }
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    Quat::from_mat3(&Mat3::from_cols(right, up, -forward))
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("combat_demo=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "Combat Demo".into(),
                width: Some(1280),
                height: Some(720),
                ..Default::default()
            },
            msaa: 4,
            ..Default::default()
        },
        ..EngineConfig::default()
    };

    let game = CombatDemo {
        exit: false,
        attack_timer: 0.0,
        spawn_timer: SPAWN_INTERVAL,
        kills: 0,
        spider_parts: Vec::new(),
        spider_clips: Vec::new(),
        spider_skin: None,
        spider_anim_graph: None,
    };

    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
