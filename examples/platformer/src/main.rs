//! 3D platformer — engine validation game.
//!
//! One small level, ~12 platforms (3 moving), 6 collectibles, third-person camera,
//! gravity + AABB collision, jump SFX + collect SFX. Win condition: collect all items.
//!
//! Uses trigger volumes for collectible pickup detection and GPU particles for
//! jump dust, collect sparkle, and win fireworks effects.

use esox_engine::*;
use esox_engine::glam::{Quat, Vec3};
use esox_gfx::mesh3d::{
    Aabb, AnimationPlayer, GltfScene, MaterialDescriptor, MaterialType,
    MeshData, ParticlePoolHandle, PostProcess3DConfig, ShadowConfig,
};

use std::f32::consts::{FRAC_PI_4, TAU};
use std::path::Path;

// ── Constants ──

const GRAVITY: f32 = -20.0;
const JUMP_IMPULSE: f32 = 8.0;
const MOVE_SPEED: f32 = 6.0;
const PLAYER_HALF: Vec3 = Vec3::new(0.4, 0.5, 0.4);
const SPAWN_POS: Vec3 = Vec3::new(0.0, 2.0, 0.0);

const CAM_DISTANCE: f32 = 8.0;
const CAM_PITCH: f32 = 0.4;
const CAM_HEIGHT_OFFSET: f32 = 1.5;
const ORBIT_SNAP: f32 = std::f32::consts::FRAC_PI_2; // 90°
const ORBIT_LERP_SPEED: f32 = 12.0;
const JUMP_BUFFER_TICKS: u32 = 6; // ~100ms input grace window
const COYOTE_TICKS: u32 = 5;      // ~83ms after leaving ground

// ── Level data ──

struct PlatformDef {
    center: Vec3,
    half_extents: Vec3,
    mover: Option<MovingDef>,
    collectible: bool,
}

#[derive(Clone, Copy)]
struct MovingDef {
    axis: Vec3,
    amplitude: f32,
    period: f32,
}

fn level_platforms() -> Vec<PlatformDef> {
    vec![
        PlatformDef { center: Vec3::new(0.0, -0.5, 0.0),   half_extents: Vec3::new(12.0, 0.5, 12.0), mover: None, collectible: false },
        PlatformDef { center: Vec3::new(3.0, 1.0, 0.0),     half_extents: Vec3::new(1.5, 0.2, 1.5),   mover: None, collectible: false },
        PlatformDef { center: Vec3::new(6.0, 2.5, 3.0),     half_extents: Vec3::new(1.2, 0.2, 1.2),   mover: None, collectible: true },
        PlatformDef { center: Vec3::new(4.0, 4.0, 7.0),     half_extents: Vec3::new(1.0, 0.2, 1.5),   mover: Some(MovingDef { axis: Vec3::X, amplitude: 2.0, period: 3.0 }), collectible: false },
        PlatformDef { center: Vec3::new(0.0, 5.5, 9.0),     half_extents: Vec3::new(1.5, 0.2, 1.0),   mover: None, collectible: true },
        PlatformDef { center: Vec3::new(-4.0, 7.0, 7.0),    half_extents: Vec3::new(1.2, 0.2, 1.2),   mover: None, collectible: true },
        PlatformDef { center: Vec3::new(-6.0, 8.5, 3.0),    half_extents: Vec3::new(1.0, 0.2, 1.5),   mover: Some(MovingDef { axis: Vec3::Z, amplitude: 1.5, period: 2.5 }), collectible: false },
        PlatformDef { center: Vec3::new(-4.0, 10.0, 0.0),   half_extents: Vec3::new(1.5, 0.2, 1.0),   mover: None, collectible: true },
        PlatformDef { center: Vec3::new(-1.0, 11.5, -3.0),  half_extents: Vec3::new(1.2, 0.2, 1.2),   mover: Some(MovingDef { axis: Vec3::X, amplitude: 1.0, period: 2.0 }), collectible: false },
        PlatformDef { center: Vec3::new(2.0, 13.0, -5.0),   half_extents: Vec3::new(1.5, 0.2, 1.5),   mover: None, collectible: true },
        PlatformDef { center: Vec3::new(0.0, 14.5, -2.0),   half_extents: Vec3::new(2.0, 0.2, 2.0),   mover: None, collectible: true },
        PlatformDef { center: Vec3::new(5.0, 3.5, -4.0),    half_extents: Vec3::new(1.0, 0.2, 1.0),   mover: None, collectible: false },
    ]
}

// ── Runtime state ──

struct PlatformRuntime {
    entity: hecs::Entity,
    base_center: Vec3,
    half_extents: Vec3,
    current_center: Vec3,
    mover: Option<MovingDef>,
}

impl PlatformRuntime {
    fn aabb(&self) -> Aabb {
        Aabb::new(
            self.current_center - self.half_extents,
            self.current_center + self.half_extents,
        )
    }
}

struct CollectibleState {
    entity: hecs::Entity,
    /// Physics body handle for the trigger sensor.
    body_handle: BodyHandle,
    base_pos: Vec3,
    collected: bool,
}

const COLLECTIBLE_HALF: Vec3 = Vec3::new(0.3, 0.3, 0.3);
const COLLISION_SKIN: f32 = 0.002;

// ── Audio ──

struct GameAudio {
    manager: esox_engine::audio::AudioManager,
    jump_sfx: esox_engine::audio::spatial::SoundHandle,
    collect_sfx: esox_engine::audio::spatial::SoundHandle,
}

// ── Particles ──

struct ParticleEffects {
    particle_mat: esox_gfx::mesh3d::MaterialHandle,
    jump_pool: ParticlePoolHandle,
    collect_pool: ParticlePoolHandle,
    win_pool: ParticlePoolHandle,
    /// Entity for the jump dust emitter (burst-only, dormant).
    jump_emitter: hecs::Entity,
    /// Entity for the collect sparkle emitter (burst-only, dormant).
    collect_emitter: hecs::Entity,
    /// Entity for the win fireworks emitter (continuous, dormant).
    win_emitter: hecs::Entity,
}

// ── Game ──

struct Platformer {
    player_entity: Option<hecs::Entity>,
    player_body: Option<BodyHandle>,
    player_pos: Vec3,
    prev_player_pos: Vec3,
    player_vel: Vec3,
    player_facing: Quat,
    grounded: bool,
    jump_buffer: u32,
    coyote_timer: u32,

    camera_entity: Option<hecs::Entity>,
    orbit_angle: f32,
    orbit_target: f32,

    platforms: Vec<PlatformRuntime>,
    collectibles: Vec<CollectibleState>,

    collected: u32,
    total: u32,
    won: bool,
    exit: bool,

    audio: Option<GameAudio>,
    particles: Option<ParticleEffects>,
}

impl Platformer {
    fn new() -> Self {
        Self {
            player_entity: None,
            player_body: None,
            player_pos: SPAWN_POS,
            prev_player_pos: SPAWN_POS,
            player_vel: Vec3::ZERO,
            player_facing: Quat::IDENTITY,
            grounded: false,
            jump_buffer: 0,
            coyote_timer: 0,

            camera_entity: None,
            orbit_angle: 0.0,
            orbit_target: 0.0,

            platforms: Vec::new(),
            collectibles: Vec::new(),

            collected: 0,
            total: 0,
            won: false,
            exit: false,

            audio: None,
            particles: None,
        }
    }

    fn player_aabb(&self) -> Aabb {
        Aabb::new(self.player_pos - PLAYER_HALF, self.player_pos + PLAYER_HALF)
    }

    fn camera_position(&self, target: Vec3) -> Vec3 {
        let (sin_o, cos_o) = self.orbit_angle.sin_cos();
        Vec3::new(
            target.x + CAM_DISTANCE * CAM_PITCH.cos() * sin_o,
            target.y + CAM_HEIGHT_OFFSET + CAM_DISTANCE * CAM_PITCH.sin(),
            target.z + CAM_DISTANCE * CAM_PITCH.cos() * cos_o,
        )
    }
}

impl Game for Platformer {
    fn init(&mut self, ctx: &mut Ctx) {
        // ── Input bindings ──
        use esox_engine::winit::keyboard::KeyCode;

        ctx.input.bind_axis("move_x", AxisBinding::Keys { negative: KeyCode::KeyA, positive: KeyCode::KeyD });
        ctx.input.bind_axis("move_z", AxisBinding::Keys { negative: KeyCode::KeyS, positive: KeyCode::KeyW });
        ctx.input.bind_action("jump", ActionBinding::Key(KeyCode::Space));
        ctx.input.bind_action("orbit_left", ActionBinding::Key(KeyCode::KeyQ));
        ctx.input.bind_action("orbit_right", ActionBinding::Key(KeyCode::KeyE));
        ctx.input.bind_action("exit", ActionBinding::Key(KeyCode::Escape));

        // ── Materials ──
        let ground_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.4, 0.4, 0.35, 1.0],
            roughness: 0.9,
            metallic: 0.0,
            ..MaterialDescriptor::default()
        });

        let platform_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.5, 0.48, 0.42, 1.0],
            roughness: 0.7,
            metallic: 0.1,
            ..MaterialDescriptor::default()
        });

        let moving_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.3, 0.45, 0.7, 1.0],
            roughness: 0.5,
            metallic: 0.3,
            ..MaterialDescriptor::default()
        });

        // player_mat is no longer needed — Fox.glb brings its own materials.

        let collectible_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [1.0, 0.85, 0.2, 1.0],
            roughness: 0.35,
            metallic: 0.8,
            emissive: [0.08, 0.06, 0.01],
            ..MaterialDescriptor::default()
        });

        // ── Meshes ──
        let cube_data = MeshData::cube(1.0);
        let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &cube_data);
        ctx.assets.register_mesh(cube_mesh);

        let torus_data = MeshData::torus(0.3, 0.1, 16, 8);
        let torus_mesh = ctx.renderer.upload_mesh(ctx.gpu, &torus_data);
        ctx.assets.register_mesh(torus_mesh);

        // ── Player kinematic body (for trigger detection) ──
        // Solid collider (not sensor) so Rapier generates intersection events
        // when it overlaps sensor colliders on collectibles.
        let player_handle = ctx.physics.add_body(BodyDesc {
            position: self.player_pos,
            rotation: Quat::IDENTITY,
            body_type: BodyType::Kinematic,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Box { half_extents: PLAYER_HALF },
                is_sensor: false,
                ..ColliderDesc::default()
            }),
        });
        self.player_body = Some(player_handle);

        // ── Spawn platforms ──
        let defs = level_platforms();
        for def in &defs {
            let mat = if def.mover.is_some() {
                moving_mat
            } else if def.half_extents.x > 5.0 {
                ground_mat
            } else {
                platform_mat
            };

            let entity = ctx.world.spawn((
                Transform3D {
                    position: def.center,
                    scale: def.half_extents * 2.0,
                    ..Transform3D::default()
                },
                GlobalTransform::default(),
                MeshRenderer {
                    mesh: cube_mesh,
                    material: mat,
                    tint: [1.0; 4],
                    visible: true,
                },
            ));

            self.platforms.push(PlatformRuntime {
                entity,
                base_center: def.center,
                half_extents: def.half_extents,
                current_center: def.center,
                mover: def.mover,
            });

            if def.collectible {
                let coll_pos = Vec3::new(def.center.x, def.center.y + def.half_extents.y + 0.5, def.center.z);

                // Create a sensor body for trigger-based pickup detection.
                let coll_handle = ctx.physics.add_body(BodyDesc {
                    position: coll_pos,
                    rotation: Quat::IDENTITY,
                    body_type: BodyType::Static,
                    collider: Some(ColliderDesc {
                        shape: ColliderShape::Box { half_extents: COLLECTIBLE_HALF },
                        is_sensor: true,
                        ..ColliderDesc::default()
                    }),
                });

                let coll_entity = ctx.world.spawn((
                    Transform3D {
                        position: coll_pos,
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: torus_mesh,
                        material: collectible_mat,
                        tint: [1.0; 4],
                        visible: true,
                    },
                    TriggerVolume { tag: Some("collectible") },
                    RigidBodyComponent {
                        handle: coll_handle,
                        body_type: BodyType::Static,
                    },
                ));

                // Register in entity map so trigger events can resolve to entities.
                ctx.entity_map.insert(coll_handle, coll_entity);

                self.collectibles.push(CollectibleState {
                    entity: coll_entity,
                    body_handle: coll_handle,
                    base_pos: coll_pos,
                    collected: false,
                });
            }
        }

        // Register player in entity map too.
        if let Some(pe) = self.player_entity {
            ctx.entity_map.insert(player_handle, pe);
        }
        // We need the player entity first — spawn it now.

        self.total = self.collectibles.len() as u32;

        // ── Load Fox model ──
        let fox_scene = GltfScene::load(Path::new("assets/models/Fox.glb"))
            .expect("failed to load Fox.glb");
        let fox_handles = ctx.renderer.upload_gltf_scene(ctx.gpu, fox_scene);

        eprintln!("[platformer] fox: {} meshes, {} materials, {} skins, {} anims",
            fox_handles.meshes.len(), fox_handles.materials.len(),
            fox_handles.skins.len(), fox_handles.animations.len());
        for (i, si) in fox_handles.skinned_mesh_indices.iter().enumerate() {
            eprintln!("[platformer]   mesh[{}] handle={:?} skinned={:?}", i, fox_handles.meshes[i], si);
        }
        for (i, a) in fox_handles.animations.iter().enumerate() {
            eprintln!("[platformer]   anim[{}] name={:?} dur={:.2}s", i, a.name, a.duration);
        }

        // Find clip indices by name. Fox has: Survey (0), Walk (1), Run (2).
        let mut survey_idx = 0;
        let mut walk_idx = 1;
        let mut run_idx = 2;
        for (i, clip) in fox_handles.animations.iter().enumerate() {
            match clip.name.as_deref() {
                Some("Survey") => survey_idx = i,
                Some("Walk") => walk_idx = i,
                Some("Run") => run_idx = i,
                _ => {}
            }
        }

        // Build animation graph: Idle (Survey) <-> Locomotion (Walk/Run blend tree)
        let anim_graph_def = AnimGraphDef {
            states: vec![
                AnimState {
                    name: "Idle".into(),
                    source: StateSource::Clip { clip_index: survey_idx },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 1,
                        conditions: vec![Condition::FloatGt {
                            param: "speed".into(),
                            threshold: 0.5,
                        }],
                        duration: 0.25,
                        priority: 0,
                    }],
                },
                AnimState {
                    name: "Locomotion".into(),
                    source: StateSource::BlendTree1D {
                        param: "speed".into(),
                        entries: vec![
                            BlendEntry { clip_index: walk_idx, threshold: 0.0 },
                            BlendEntry { clip_index: run_idx, threshold: 1.0 },
                        ],
                    },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 0,
                        conditions: vec![Condition::FloatLt {
                            param: "speed".into(),
                            threshold: 0.5,
                        }],
                        duration: 0.3,
                        priority: 0,
                    }],
                },
            ],
            default_state: 0,
        };

        // Find the skinned mesh handle and its index.
        let fox_mesh = fox_handles.meshes[0];
        let fox_material = fox_handles.materials[0];
        let skinned_mesh_index = fox_handles.skinned_mesh_indices[0]
            .expect("Fox mesh should be skinned");

        // Build animation graph runtime.
        let player_anim = AnimationPlayer::new(&fox_handles.skins[0]);
        let anim_graph = AnimGraphRuntime::new(anim_graph_def, player_anim);

        // ── Spawn player (Fox model) ──
        // Fox model is large, scale down. The fox faces +Z by default.
        let player_entity = ctx.world.spawn((
            Transform3D {
                position: self.player_pos,
                scale: Vec3::splat(0.02),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            MeshRenderer {
                mesh: fox_mesh,
                material: fox_material,
                tint: [1.0; 4],
                visible: true,
            },
            RigidBodyComponent {
                handle: player_handle,
                body_type: BodyType::Kinematic,
            },
            AnimGraphController {
                graph: anim_graph,
                clips: fox_handles.animations,
                skinned_mesh_index,
            },
        ));
        self.player_entity = Some(player_entity);
        ctx.entity_map.insert(player_handle, player_entity);

        // ── Lights ──
        ctx.world.spawn((
            Transform3D {
                position: Vec3::ZERO,
                rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_3),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            DirectionalLightComponent {
                color: [1.0, 0.95, 0.85],
                intensity: 2.5,
            },
        ));

        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(0.0, 18.0, 0.0),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            PointLightComponent {
                color: [0.6, 0.7, 1.0],
                intensity: 5.0,
                range: 30.0,
            },
        ));

        // ── Camera ──
        let cam_pos = self.camera_position(self.player_pos);
        let cam_entity = ctx.world.spawn((
            Transform3D {
                position: cam_pos,
                rotation: look_at_quat(cam_pos, self.player_pos + Vec3::Y * 0.5),
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
        self.camera_entity = Some(cam_entity);

        // ── Environment ──
        ctx.renderer.generate_procedural_ibl(ctx.gpu);

        ctx.renderer.set_shadow_config(ShadowConfig {
            enabled: true,
            cascade_count: 3,
            shadow_distance: 40.0,
            depth_bias: 0.005,
            normal_bias: 0.05,
        });

        ctx.renderer.set_postprocess(PostProcess3DConfig {
            bloom_enabled: true,
            bloom_intensity: 0.04,
            bloom_threshold: 3.5,
            bloom_soft_knee: 0.1,
            tone_map_enabled: true,
            ssao_enabled: false,
            motion_blur_enabled: false,
        });

        // ── Particles ──
        let particle_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::Unlit,
            albedo: [1.0, 1.0, 1.0, 1.0],
            ..MaterialDescriptor::default()
        });

        let jump_pool = ctx.renderer.create_particle_pool(ctx.gpu, 256);
        let collect_pool = ctx.renderer.create_particle_pool(ctx.gpu, 512);
        let win_pool = ctx.renderer.create_particle_pool(ctx.gpu, 2048);

        // Jump dust emitter — burst-only, activated on jump.
        let jump_emitter = ctx.world.spawn((
            Transform3D { position: SPAWN_POS, ..Transform3D::default() },
            GlobalTransform::default(),
            ParticleEmitter {
                pool: jump_pool,
                material: particle_mat,
                spawn_rate: 0.0,
                burst_count: 0,
                velocity_min: Vec3::new(-1.5, 0.2, -1.5),
                velocity_max: Vec3::new(1.5, 1.0, 1.5),
                gravity: Vec3::new(0.0, -5.0, 0.0),
                lifetime: [0.2, 0.5],
                size: [0.08, 0.02],
                color_start: [0.8, 0.75, 0.65, 0.8],
                color_end: [0.6, 0.55, 0.5, 0.0],
                active: false,
                ..Default::default()
            },
        ));

        // Collect sparkle emitter — burst-only, activated on pickup.
        let collect_emitter = ctx.world.spawn((
            Transform3D { position: Vec3::ZERO, ..Transform3D::default() },
            GlobalTransform::default(),
            ParticleEmitter {
                pool: collect_pool,
                material: particle_mat,
                spawn_rate: 0.0,
                burst_count: 0,
                velocity_min: Vec3::new(-2.0, 1.0, -2.0),
                velocity_max: Vec3::new(2.0, 4.0, 2.0),
                gravity: Vec3::new(0.0, -3.0, 0.0),
                lifetime: [0.4, 1.0],
                size: [0.1, 0.01],
                color_start: [1.0, 0.9, 0.3, 1.0],
                color_end: [1.0, 0.6, 0.1, 0.0],
                active: false,
                ..Default::default()
            },
        ));

        // Win fireworks emitter — continuous spray, activated on win.
        let win_emitter = ctx.world.spawn((
            Transform3D { position: Vec3::ZERO, ..Transform3D::default() },
            GlobalTransform::default(),
            ParticleEmitter {
                pool: win_pool,
                material: particle_mat,
                spawn_rate: 0.0,
                burst_count: 0,
                velocity_min: Vec3::new(-4.0, 5.0, -4.0),
                velocity_max: Vec3::new(4.0, 12.0, 4.0),
                gravity: Vec3::new(0.0, -8.0, 0.0),
                lifetime: [1.0, 2.5],
                size: [0.12, 0.03],
                color_start: [1.0, 0.85, 0.2, 1.0],
                color_end: [1.0, 0.3, 0.1, 0.0],
                active: false,
                ..Default::default()
            },
        ));

        self.particles = Some(ParticleEffects {
            particle_mat,
            jump_pool,
            collect_pool,
            win_pool,
            jump_emitter,
            collect_emitter,
            win_emitter,
        });

        // ── Audio (graceful fallback if files missing) ──
        if let Some(mut mgr) = esox_engine::audio::AudioManager::new() {
            let jump = mgr.load("assets/jump.ogg");
            let collect = mgr.load("assets/collect.ogg");
            match (jump, collect) {
                (Ok(j), Ok(c)) => {
                    self.audio = Some(GameAudio {
                        manager: mgr,
                        jump_sfx: j,
                        collect_sfx: c,
                    });
                }
                _ => {
                    eprintln!("[platformer] audio files not found, running silent");
                }
            }
        }

        eprintln!("[platformer] init complete — {} platforms, {} collectibles", self.platforms.len(), self.total);
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.input.just_pressed("exit") {
            self.exit = true;
            return;
        }

        let dt = ctx.time.tick_dt;
        let elapsed = ctx.time.elapsed;

        // ── Process trigger events (collectible pickup via physics sensors) ──
        if !self.won {
            let triggers = ctx.physics.drain_triggers();
            for event in &triggers {
                if event.phase != TriggerPhase::Enter {
                    continue;
                }
                // Check if either body in the pair is the player and the other is a collectible.
                if let Some((entity_a, entity_b)) = ctx.entity_map.resolve_trigger(event) {
                    let (player_e, other_e) = if Some(entity_a) == self.player_entity {
                        (entity_a, entity_b)
                    } else if Some(entity_b) == self.player_entity {
                        (entity_b, entity_a)
                    } else {
                        continue;
                    };

                    // Find the matching collectible.
                    let _ = player_e;
                    for coll in &mut self.collectibles {
                        if coll.collected || coll.entity != other_e {
                            continue;
                        }
                        coll.collected = true;
                        if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(coll.entity) {
                            mr.visible = false;
                        }
                        self.collected += 1;
                        eprintln!("[platformer] collected {}/{}", self.collected, self.total);

                        if let Some(audio) = &mut self.audio {
                            audio.manager.play(audio.collect_sfx);
                        }

                        // Burst collect particles at the collectible position.
                        if let Some(particles) = &self.particles {
                            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(particles.collect_emitter) {
                                t.position = coll.base_pos;
                            }
                            if let Ok(mut em) = ctx.world.get::<&mut ParticleEmitter>(particles.collect_emitter) {
                                em.active = true;
                                em.burst_count = 64;
                            }
                        }

                        if self.collected == self.total {
                            self.won = true;
                            eprintln!("[platformer] YOU WIN! All {} items collected!", self.total);

                            // Start win fireworks at the player position.
                            if let Some(particles) = &self.particles {
                                if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(particles.win_emitter) {
                                    t.position = self.player_pos;
                                }
                                if let Ok(mut em) = ctx.world.get::<&mut ParticleEmitter>(particles.win_emitter) {
                                    em.active = true;
                                    em.spawn_rate = 200.0;
                                    em.burst_count = 256;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }

        // ── Moving platforms ──
        for plat in &mut self.platforms {
            if let Some(mover) = plat.mover {
                let offset = mover.axis * mover.amplitude * (elapsed * TAU / mover.period).sin();
                plat.current_center = plat.base_center + offset;

                if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(plat.entity) {
                    t.position = plat.current_center;
                }
            }
        }

        // ── Collectible animation ──
        for coll in &self.collectibles {
            if coll.collected {
                continue;
            }
            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(coll.entity) {
                t.rotation = Quat::from_rotation_y(elapsed * 2.0);
                t.position.y = coll.base_pos.y + 0.15 * (elapsed * 3.0).sin();
            }
            // Sync the sensor body position to match the visual.
            let pos = Vec3::new(coll.base_pos.x, coll.base_pos.y + 0.15 * (elapsed * 3.0).sin(), coll.base_pos.z);
            ctx.physics.set_transform(coll.body_handle, pos, Quat::IDENTITY);
        }

        // ── Player input ──
        let move_x = ctx.input.axis("move_x");
        let move_z = ctx.input.axis("move_z");

        // Camera-relative movement (use target so controls snap immediately)
        let (sin_o, cos_o) = self.orbit_target.sin_cos();
        let dir = Vec3::new(
            move_x * cos_o - move_z * sin_o,
            0.0,
            -move_x * sin_o - move_z * cos_o,
        );

        self.player_vel.x = dir.x * MOVE_SPEED;
        self.player_vel.z = dir.z * MOVE_SPEED;

        // Gravity
        self.player_vel.y += GRAVITY * dt;
        self.player_vel.y = self.player_vel.y.max(-30.0);

        // Buffer jump input
        if ctx.input.just_pressed("jump") {
            self.jump_buffer = JUMP_BUFFER_TICKS;
        }

        // ── Integrate & collide (split-axis: XZ then Y) ──
        self.prev_player_pos = self.player_pos;

        // Horizontal movement + collision.
        self.player_pos.x += self.player_vel.x * dt;
        self.player_pos.z += self.player_vel.z * dt;

        let y_skin = Vec3::new(0.0, COLLISION_SKIN, 0.0);
        for plat in &self.platforms {
            let player_aabb = Aabb::new(
                self.player_pos - PLAYER_HALF + y_skin,
                self.player_pos + PLAYER_HALF - y_skin,
            );
            let plat_aabb = plat.aabb();
            if !player_aabb.intersects(&plat_aabb) {
                continue;
            }

            let pen_pos = player_aabb.max - plat_aabb.min;
            let pen_neg = plat_aabb.max - player_aabb.min;
            let pen_x = pen_pos.x.min(pen_neg.x);
            let pen_z = pen_pos.z.min(pen_neg.z);

            if pen_x <= pen_z {
                if pen_pos.x < pen_neg.x {
                    self.player_pos.x -= pen_pos.x;
                } else {
                    self.player_pos.x += pen_neg.x;
                }
                self.player_vel.x = 0.0;
            } else {
                if pen_pos.z < pen_neg.z {
                    self.player_pos.z -= pen_pos.z;
                } else {
                    self.player_pos.z += pen_neg.z;
                }
                self.player_vel.z = 0.0;
            }
        }

        // Vertical movement + collision.
        self.player_pos.y += self.player_vel.y * dt;
        let mut landed = false;

        let xz_skin = Vec3::new(COLLISION_SKIN, 0.0, COLLISION_SKIN);
        for plat in &self.platforms {
            let player_aabb = Aabb::new(
                self.player_pos - PLAYER_HALF + xz_skin,
                self.player_pos + PLAYER_HALF - xz_skin,
            );
            let plat_aabb = plat.aabb();
            if !player_aabb.intersects(&plat_aabb) {
                continue;
            }

            let pen_above = player_aabb.max.y - plat_aabb.min.y;
            let pen_below = plat_aabb.max.y - player_aabb.min.y;

            if pen_above < pen_below {
                self.player_pos.y -= pen_above;
            } else {
                self.player_pos.y += pen_below;
                landed = true;
            }
            self.player_vel.y = 0.0;
        }

        self.grounded = landed;

        // Coyote time — grace window after leaving ground
        if self.grounded {
            self.coyote_timer = COYOTE_TICKS;
        } else {
            self.coyote_timer = self.coyote_timer.saturating_sub(1);
        }

        // Jump (checked after collision so grounded/coyote state is current)
        if self.jump_buffer > 0 && self.coyote_timer > 0 {
            self.player_vel.y = JUMP_IMPULSE;
            self.jump_buffer = 0;
            self.coyote_timer = 0;
            if let Some(audio) = &mut self.audio {
                audio.manager.play(audio.jump_sfx);
            }

            // Burst jump dust particles at the player's feet.
            if let Some(particles) = &self.particles {
                let foot_pos = self.player_pos - Vec3::Y * PLAYER_HALF.y;
                if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(particles.jump_emitter) {
                    t.position = foot_pos;
                }
                if let Ok(mut em) = ctx.world.get::<&mut ParticleEmitter>(particles.jump_emitter) {
                    em.active = true;
                    em.burst_count = 24;
                }
            }
        }
        self.jump_buffer = self.jump_buffer.saturating_sub(1);

        // ── Kill plane ──
        if self.player_pos.y < -10.0 {
            self.player_pos = SPAWN_POS;
            self.player_vel = Vec3::ZERO;
            self.grounded = false;
        }

        // ── Update player facing and animation params ──
        let horiz_speed = Vec3::new(self.player_vel.x, 0.0, self.player_vel.z).length();

        // Face movement direction (smooth slerp).
        if horiz_speed > 0.1 {
            let move_dir = Vec3::new(self.player_vel.x, 0.0, self.player_vel.z).normalize();
            let target_facing = Quat::from_rotation_arc(Vec3::Z, move_dir);
            self.player_facing = self.player_facing.slerp(target_facing, (10.0 * dt).min(1.0));
        }

        // Drive animation graph: normalize speed to [0, 1] for walk/run blend.
        if let Some(pe) = self.player_entity {
            if let Ok(mut ctrl) = ctx.world.get::<&mut AnimGraphController>(pe) {
                let normalized = (horiz_speed / MOVE_SPEED).clamp(0.0, 1.0);
                ctrl.graph.params.set_float("speed", normalized);
            }
        }

        // ── Update player entity transform ──
        if let Some(pe) = self.player_entity {
            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(pe) {
                t.position = self.player_pos;
                t.rotation = self.player_facing;
            }
        }

        // ── Sync player kinematic sensor to physics world ──
        if let Some(body) = self.player_body {
            ctx.physics.set_transform(body, self.player_pos, self.player_facing);
        }

        // ── Keep win emitter tracking the player ──
        if self.won {
            if let Some(particles) = &self.particles {
                if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(particles.win_emitter) {
                    t.position = self.player_pos + Vec3::Y * 2.0;
                }
            }
        }

        // ── Camera orbit input ──
        if ctx.input.just_pressed("orbit_left") {
            self.orbit_target -= ORBIT_SNAP;
        }
        if ctx.input.just_pressed("orbit_right") {
            self.orbit_target += ORBIT_SNAP;
        }
        // Smooth lerp toward target angle.
        let diff = self.orbit_target - self.orbit_angle;
        self.orbit_angle += diff * (ORBIT_LERP_SPEED * dt).min(1.0);
    }

    fn render(&mut self, ctx: &mut Ctx, alpha: f32) {
        let visual_pos = self.prev_player_pos.lerp(self.player_pos, alpha);
        if let Some(pe) = self.player_entity {
            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(pe) {
                t.position = visual_pos;
                t.rotation = self.player_facing;
            }
        }

        // Camera tracks the interpolated visual position to avoid jitter
        // between fixed-rate physics ticks and variable-rate rendering.
        let cam_pos = self.camera_position(visual_pos);
        if let Some(ce) = self.camera_entity {
            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(ce) {
                t.position = cam_pos;
                t.rotation = look_at_quat(cam_pos, visual_pos + Vec3::Y * 0.5);
            }
        }
    }

    fn ui(&mut self, ui: &mut esox_engine::esox_ui::Ui, ctx: &Ctx) {
        use esox_gfx::Color;

        // ── Collect counter (top-left) ──
        ui.padding(16.0, |ui| {
            ui.label_colored(
                &format!("{} / {}", self.collected, self.total),
                Color::WHITE,
            );
        });

        // ── Win message (centered) ──
        if self.won {
            let vp_h = ctx.viewport.1 as f32;
            let target_y = vp_h * 0.4;
            let current_y = ui.cursor_y();
            if target_y > current_y {
                ui.add_space(target_y - current_y);
            }

            let text = "YOU WIN!";
            // Approximate heading width (~14px per char at heading size).
            let approx_w = text.len() as f32 * 14.0;
            ui.center_horizontal(approx_w, |ui| {
                ui.label_colored(text, Color::new(1.0, 0.85, 0.2, 1.0));
            });
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
    Quat::from_mat3(&esox_engine::glam::Mat3::from_cols(right, up, -forward))
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("platformer=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "esox platformer".into(),
                width: Some(1280),
                height: Some(720),
                ..Default::default()
            },
            msaa: 4,
            ..Default::default()
        },
        physics: Some(Box::new(RapierPhysics::new(Vec3::new(0.0, -20.0, 0.0)))),
        ..EngineConfig::default()
    };

    let game = Platformer::new();

    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
