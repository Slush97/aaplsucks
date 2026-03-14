//! MOBA Lane Prototype — minions spawn at each end and walk toward the center.

use esox_engine::*;
use esox_engine::glam::{Mat3, Quat, Vec3};
use esox_gfx::mesh3d::{
    AnimationClip, AnimationPlayer, GltfScene, MaterialDescriptor, MaterialType,
    MeshData, PostProcess3DConfig, ShadowConfig,
};

use std::f32::consts::FRAC_PI_4;
use std::path::Path;

// ── Custom components ──

struct Team(u8); // 0 = blue, 1 = red
struct LaneMinion;

struct WaypointFollower {
    waypoints: Vec<Vec3>,
    current: usize,
    speed: f32,
}

// ── Game ──

struct MobaLane;

impl Game for MobaLane {
    fn init(&mut self, ctx: &mut Ctx) {
        // ── Load Fox.glb ──
        let fox_scene = GltfScene::load(Path::new("assets/models/Fox.glb"))
            .expect("failed to load Fox.glb");
        let fox_handles = ctx.renderer.upload_gltf_scene(ctx.gpu, fox_scene);

        // Find clip indices by name.
        let mut survey_idx = 0;
        let mut walk_idx = 1;
        for (i, clip) in fox_handles.animations.iter().enumerate() {
            match clip.name.as_deref() {
                Some("Survey") => survey_idx = i,
                Some("Walk") => walk_idx = i,
                _ => {}
            }
        }

        // Animation graph: Idle (Survey) <-> Walk
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
                            threshold: 0.1,
                        }],
                        duration: 0.2,
                        priority: 0,
                    }],
                    events: vec![],
                },
                AnimState {
                    name: "Walk".into(),
                    source: StateSource::Clip { clip_index: walk_idx },
                    looping: true,
                    speed: 1.0,
                    transitions: vec![Transition {
                        target_state: 0,
                        conditions: vec![Condition::FloatLt {
                            param: "speed".into(),
                            threshold: 0.1,
                        }],
                        duration: 0.2,
                        priority: 0,
                    }],
                    events: vec![],
                },
            ],
            default_state: 0,
        };

        let fox_mesh = fox_handles.meshes[0];
        let fox_material = fox_handles.materials[0];
        let skinned_mesh_index = fox_handles.skinned_mesh_indices[0]
            .expect("Fox mesh should be skinned");

        // ── Spawn minions ──
        let x_offsets = [-2.0_f32, 0.0, 2.0];
        let z_steps: Vec<f32> = (-35..=35).step_by(10).map(|z| z as f32).collect();

        for &x in &x_offsets {
            // Blue team — spawn at z=-35, walk toward z=+35
            let blue_waypoints: Vec<Vec3> = z_steps.iter().map(|&z| Vec3::new(x, 0.0, z)).collect();
            spawn_minion(
                ctx,
                &fox_handles.animations,
                &fox_handles.skins[0],
                &anim_graph_def,
                fox_mesh,
                fox_material,
                skinned_mesh_index,
                Vec3::new(x, 0.0, -35.0),
                Quat::IDENTITY, // faces +Z
                blue_waypoints,
                0, // blue
            );

            // Red team — spawn at z=+35, walk toward z=-35
            let red_waypoints: Vec<Vec3> = z_steps.iter().rev().map(|&z| Vec3::new(x, 0.0, z)).collect();
            spawn_minion(
                ctx,
                &fox_handles.animations,
                &fox_handles.skins[0],
                &anim_graph_def,
                fox_mesh,
                fox_material,
                skinned_mesh_index,
                Vec3::new(x, 0.0, 35.0),
                Quat::from_rotation_y(std::f32::consts::PI), // faces -Z
                red_waypoints,
                1, // red
            );
        }

        // ── Ground plane ──
        let ground_mesh_data = MeshData::plane(12.0, 80.0, 1);
        let ground_mesh = ctx.renderer.upload_mesh(ctx.gpu, &ground_mesh_data);
        let ground_material = ctx.renderer.create_material(ctx.gpu, &MaterialDescriptor {
            material_type: MaterialType::PBR,
            albedo: [0.3, 0.35, 0.25, 1.0],
            roughness: 0.9,
            metallic: 0.0,
            ..MaterialDescriptor::default()
        });

        ctx.world.spawn((
            Transform3D::default(),
            GlobalTransform::default(),
            MeshRenderer {
                mesh: ground_mesh,
                material: ground_material,
                tint: [1.0; 4],
                visible: true,
            },
        ));

        // ── Sun light ──
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

        // ── Camera ──
        let cam_pos = Vec3::new(0.0, 25.0, -20.0);
        let cam_target = Vec3::new(0.0, 0.0, 5.0);
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

        // ── Environment ──
        ctx.renderer.generate_procedural_ibl(ctx.gpu);

        ctx.renderer.set_shadow_config(ShadowConfig {
            shadow_distance: 60.0,
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
            ssao_enabled: false,
            motion_blur_enabled: false,
        });
    }

    fn update(&mut self, ctx: &mut Ctx) {
        let dt = ctx.time.tick_dt;

        for (_entity, (follower, transform, anim)) in ctx
            .world
            .query_mut::<(&mut WaypointFollower, &mut Transform3D, &mut AnimGraphController)>()
        {
            if follower.current < follower.waypoints.len() {
                let target = follower.waypoints[follower.current];
                let dir = target - transform.position;
                let dist = dir.length();

                if dist < 0.5 {
                    follower.current += 1;
                } else {
                    let dir_n = dir / dist;
                    transform.position += dir_n * follower.speed * dt;
                    transform.rotation = look_at_quat_dir(dir_n);
                }

                anim.graph.params.set_float("speed", 1.0);
            } else {
                anim.graph.params.set_float("speed", 0.0);
            }
        }
    }

    fn ui(&mut self, ui: &mut esox_engine::esox_ui::Ui, _ctx: &Ctx) {
        use esox_gfx::Color;

        ui.padding(16.0, |ui| {
            ui.label_colored("MOBA Lane Prototype", Color::WHITE);
            ui.add_space(4.0);
            ui.label_colored("Blue team ->  <- Red team", Color::new(0.7, 0.7, 0.7, 1.0));
        });
    }
}

fn spawn_minion(
    ctx: &mut Ctx,
    clips: &[AnimationClip],
    skin: &esox_gfx::mesh3d::gltf_loader::GltfSkin,
    anim_graph_def: &AnimGraphDef,
    mesh: esox_gfx::mesh3d::MeshHandle,
    material: esox_gfx::mesh3d::MaterialHandle,
    skinned_mesh_index: usize,
    position: Vec3,
    rotation: Quat,
    waypoints: Vec<Vec3>,
    team: u8,
) {
    let player_anim = AnimationPlayer::new(skin);
    let anim_graph = AnimGraphRuntime::new(anim_graph_def.clone(), player_anim);

    let tint = if team == 0 {
        [0.5, 0.7, 1.0, 1.0] // blue tint
    } else {
        [1.0, 0.5, 0.5, 1.0] // red tint
    };

    ctx.world.spawn((
        Transform3D {
            position,
            rotation,
            scale: Vec3::splat(0.02),
            ..Transform3D::default()
        },
        GlobalTransform::default(),
        MeshRenderer {
            mesh,
            material,
            tint,
            visible: true,
        },
        AnimGraphController {
            graph: anim_graph,
            clips: clips.to_vec(),
            skinned_mesh_index,
        },
        Team(team),
        LaneMinion,
        WaypointFollower {
            waypoints,
            current: 0,
            speed: 5.0,
        },
    ));
}

/// Compute a rotation quaternion looking from `eye` toward `target`.
fn look_at_quat(eye: Vec3, target: Vec3) -> Quat {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    Quat::from_mat3(&Mat3::from_cols(right, up, -forward))
}

/// Compute a rotation quaternion facing a direction vector on the XZ plane.
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
                .add_directive("moba_lane=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "MOBA Lane Prototype".into(),
                width: Some(1280),
                height: Some(720),
                ..Default::default()
            },
            msaa: 4,
            ..Default::default()
        },
        ..EngineConfig::default()
    };

    if let Err(e) = esox_engine::run(config, MobaLane) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
