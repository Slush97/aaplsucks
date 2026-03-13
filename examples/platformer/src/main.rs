//! 3D platformer — engine validation game.
//!
//! One small level, ~12 platforms (3 moving), 6 collectibles, third-person camera,
//! gravity + AABB collision, jump SFX + collect SFX. Win condition: collect all items.

use esox_engine::*;
use esox_engine::glam::{Quat, Vec3};
use esox_gfx::mesh3d::{Aabb, MaterialDescriptor, MaterialType, MeshData, PostProcess3DConfig, ShadowConfig};

use std::f32::consts::{FRAC_PI_4, TAU};

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

// ── Game ──

struct Platformer {
    player_entity: Option<hecs::Entity>,
    player_pos: Vec3,
    prev_player_pos: Vec3,
    player_vel: Vec3,
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
}

impl Platformer {
    fn new() -> Self {
        Self {
            player_entity: None,
            player_pos: SPAWN_POS,
            prev_player_pos: SPAWN_POS,
            player_vel: Vec3::ZERO,
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

        let player_mat = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [1.0, 0.5, 0.1, 1.0],
            roughness: 0.4,
            metallic: 0.2,
            ..MaterialDescriptor::default()
        });

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
                ));
                self.collectibles.push(CollectibleState {
                    entity: coll_entity,
                    base_pos: coll_pos,
                    collected: false,
                });
            }
        }

        self.total = self.collectibles.len() as u32;

        // ── Spawn player ──
        let player_entity = ctx.world.spawn((
            Transform3D {
                position: self.player_pos,
                scale: PLAYER_HALF * 2.0,
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            MeshRenderer {
                mesh: cube_mesh,
                material: player_mat,
                tint: [1.0; 4],
                visible: true,
            },
        ));
        self.player_entity = Some(player_entity);

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
        // Resolving each axis group independently prevents the classic bug
        // where landing on a platform edge is resolved horizontally instead
        // of vertically, and eliminates jitter at corners where two
        // platforms meet.
        self.prev_player_pos = self.player_pos;

        // Horizontal movement + collision.
        // Shrink Y by COLLISION_SKIN so vertical surface contacts (standing
        // on a platform) don't trigger false horizontal resolution.
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
        // Shrink XZ by COLLISION_SKIN so horizontal surface contacts (flush
        // against a wall) don't trigger false vertical resolution.
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
        }
        self.jump_buffer = self.jump_buffer.saturating_sub(1);

        // ── Kill plane ──
        if self.player_pos.y < -10.0 {
            self.player_pos = SPAWN_POS;
            self.player_vel = Vec3::ZERO;
            self.grounded = false;
        }

        // ── Collectible pickup ──
        if !self.won {
            let player_aabb = self.player_aabb();
            for coll in &mut self.collectibles {
                if coll.collected {
                    continue;
                }
                let coll_pos = if let Ok(t) = ctx.world.get::<&Transform3D>(coll.entity) {
                    t.position
                } else {
                    coll.base_pos
                };
                let coll_aabb = Aabb::new(coll_pos - COLLECTIBLE_HALF, coll_pos + COLLECTIBLE_HALF);
                if player_aabb.intersects(&coll_aabb) {
                    coll.collected = true;
                    if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(coll.entity) {
                        mr.visible = false;
                    }
                    self.collected += 1;
                    eprintln!("[platformer] collected {}/{}", self.collected, self.total);

                    if let Some(audio) = &mut self.audio {
                        audio.manager.play(audio.collect_sfx);
                    }

                    if self.collected == self.total {
                        self.won = true;
                        eprintln!("[platformer] YOU WIN! All {} items collected!", self.total);
                    }
                }
            }
        }

        // ── Update player entity transform ──
        if let Some(pe) = self.player_entity {
            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(pe) {
                t.position = self.player_pos;
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
        ..EngineConfig::default()
    };

    let game = Platformer::new();

    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
