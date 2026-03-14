use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use esox_engine::esox_gfx::mesh3d::{
    InstanceData, MaterialDescriptor, MaterialHandle, MaterialType, MeshData, MeshHandle,
    PostProcess3DConfig, ShadowConfig,
};
use esox_engine::glam::{self, Mat4, Quat, Vec3};
use esox_engine::hecs;
use esox_engine::winit::keyboard::KeyCode;
use esox_engine::esox_ui::ToastKind;
use esox_engine::{
    ActionBinding, Camera3D, Ctx, DirectionalLightComponent, EngineConfig, Game, GlobalTransform,
    MeshRenderer, PointLightComponent, SpotLightComponent, Tag, Transform3D,
};

mod hierarchy;
mod inspector;
mod picking;
mod undo;

use undo::{ComponentSnapshot, EntitySnapshot, UndoAction, UndoStack};

// ── Component kinds for add/remove ──

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ComponentKind {
    PointLight,
    SpotLight,
    DirLight,
    Camera,
    MeshRenderer,
}

// ── Editor camera ──

#[derive(Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    Orbit,
    Fly,
}

struct EditorCamera {
    mode: CameraMode,
    // Orbit
    orbit_target: Vec3,
    orbit_yaw: f32,
    orbit_pitch: f32,
    orbit_distance: f32,
    // Fly
    fly_yaw: f32,
    fly_pitch: f32,
    fly_pos: Vec3,
}

impl EditorCamera {
    fn new() -> Self {
        Self {
            mode: CameraMode::Orbit,
            orbit_target: Vec3::ZERO,
            orbit_yaw: 0.4,
            orbit_pitch: -0.3,
            orbit_distance: 10.0,
            fly_yaw: 0.4,
            fly_pitch: -0.3,
            fly_pos: Vec3::new(-5.0, 3.0, 8.0),
        }
    }

    fn position(&self) -> Vec3 {
        match self.mode {
            CameraMode::Orbit => {
                let x = self.orbit_distance * self.orbit_pitch.cos() * self.orbit_yaw.sin();
                let y = self.orbit_distance * (-self.orbit_pitch).sin();
                let z = self.orbit_distance * self.orbit_pitch.cos() * self.orbit_yaw.cos();
                self.orbit_target + Vec3::new(x, y, z)
            }
            CameraMode::Fly => self.fly_pos,
        }
    }

    fn forward(&self) -> Vec3 {
        match self.mode {
            CameraMode::Orbit => (self.orbit_target - self.position()).normalize_or_zero(),
            CameraMode::Fly => {
                Vec3::new(
                    self.fly_pitch.cos() * self.fly_yaw.sin(),
                    (-self.fly_pitch).sin(),
                    self.fly_pitch.cos() * self.fly_yaw.cos(),
                )
                .normalize_or_zero()
                * -1.0
            }
        }
    }

    fn rotation(&self) -> Quat {
        let pos = self.position();
        let target = match self.mode {
            CameraMode::Orbit => self.orbit_target,
            CameraMode::Fly => pos + self.forward(),
        };
        let dir = (target - pos).normalize_or_zero();
        if dir.length_squared() < 1e-6 {
            return Quat::IDENTITY;
        }
        Quat::from_rotation_arc(-Vec3::Z, dir)
    }

    fn update(&mut self, ctx: &Ctx) {
        let dt = ctx.time.tick_dt;
        let mmb = ctx.input.is_mouse_button_down(1);
        let rmb = ctx.input.is_mouse_button_down(2);
        let shift = ctx.input.is_key_down(KeyCode::ShiftLeft)
            || ctx.input.is_key_down(KeyCode::ShiftRight);
        let (dx, dy) = ctx.input.mouse_delta();
        let scroll = ctx.input.scroll_delta();

        // Toggle to fly mode when RMB held
        if rmb && !mmb {
            if self.mode == CameraMode::Orbit {
                // Transition: copy orbit camera state to fly
                self.fly_pos = self.position();
                self.fly_yaw = self.orbit_yaw;
                self.fly_pitch = self.orbit_pitch;
                self.mode = CameraMode::Fly;
            }
            // Fly mode: mouse look
            let sensitivity = 0.003;
            self.fly_yaw -= dx as f32 * sensitivity;
            self.fly_pitch += dy as f32 * sensitivity;
            self.fly_pitch = self.fly_pitch.clamp(-PI * 0.49, PI * 0.49);

            // WASD movement
            let speed = if shift { 20.0 } else { 5.0 } * dt;
            let forward = self.forward();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let up = Vec3::Y;

            if ctx.input.is_key_down(KeyCode::KeyW) {
                self.fly_pos += forward * speed;
            }
            if ctx.input.is_key_down(KeyCode::KeyS) {
                self.fly_pos -= forward * speed;
            }
            if ctx.input.is_key_down(KeyCode::KeyA) {
                self.fly_pos -= right * speed;
            }
            if ctx.input.is_key_down(KeyCode::KeyD) {
                self.fly_pos += right * speed;
            }
            if ctx.input.is_key_down(KeyCode::KeyE) {
                self.fly_pos += up * speed;
            }
            if ctx.input.is_key_down(KeyCode::KeyQ) {
                self.fly_pos -= up * speed;
            }
        } else {
            // Return to orbit mode when RMB released
            if self.mode == CameraMode::Fly {
                self.orbit_yaw = self.fly_yaw;
                self.orbit_pitch = self.fly_pitch;
                self.orbit_target = self.fly_pos + self.forward() * self.orbit_distance;
                self.mode = CameraMode::Orbit;
            }

            if mmb {
                if shift {
                    // Pan
                    let sensitivity = 0.005 * self.orbit_distance;
                    let pos = self.position();
                    let forward = (self.orbit_target - pos).normalize_or_zero();
                    let right = forward.cross(Vec3::Y).normalize_or_zero();
                    let up = right.cross(forward).normalize_or_zero();
                    let pan = right * (-dx as f32 * sensitivity)
                        + up * (dy as f32 * sensitivity);
                    self.orbit_target += pan;
                } else {
                    // Rotate
                    let sensitivity = 0.005;
                    self.orbit_yaw -= dx as f32 * sensitivity;
                    self.orbit_pitch += dy as f32 * sensitivity;
                    self.orbit_pitch = self.orbit_pitch.clamp(-PI * 0.49, PI * 0.49);
                }
            }

            // Scroll to zoom
            if scroll.abs() > 0.01 {
                self.orbit_distance *= 1.0 - scroll * 0.1;
                self.orbit_distance = self.orbit_distance.clamp(0.5, 500.0);
            }
        }
    }

    fn focus_on(&mut self, pos: Vec3) {
        self.orbit_target = pos;
        self.mode = CameraMode::Orbit;
    }
}

// ── Pending edits (bridge between immutable ui() and mutable update()) ──

enum PendingEdit {
    SetTransform(hecs::Entity, Transform3D),
    SetPointLightIntensity(hecs::Entity, f32),
    SetPointLightRange(hecs::Entity, f32),
    SetPointLightColor(hecs::Entity, [f32; 3]),
    SetSpotLightIntensity(hecs::Entity, f32),
    SetSpotLightRange(hecs::Entity, f32),
    SetSpotLightColor(hecs::Entity, [f32; 3]),
    SetSpotLightInnerCone(hecs::Entity, f32),
    SetSpotLightOuterCone(hecs::Entity, f32),
    SetDirLightIntensity(hecs::Entity, f32),
    SetDirLightColor(hecs::Entity, [f32; 3]),
    SetTag(hecs::Entity, String),
    SetCameraFov(hecs::Entity, f32),
    SetCameraNear(hecs::Entity, f32),
    SetCameraFar(hecs::Entity, f32),
    SetMeshVisible(hecs::Entity, bool),
    SetMeshTint(hecs::Entity, [f32; 4]),
    SetPointLightShadows(hecs::Entity, bool),
    SetSpotLightShadows(hecs::Entity, bool),
    AddComponent(hecs::Entity, ComponentKind),
    RemoveComponent(hecs::Entity, ComponentKind),
    SetMesh(hecs::Entity, MeshHandle),
    SetMaterial(hecs::Entity, MaterialHandle),
    SetMaterialDescriptor(hecs::Entity, MaterialDescriptor),
}

// ── Gizmo mode ──

#[derive(Clone, Copy, PartialEq, Eq)]
enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

// ── Gizmo drag state ──

struct GizmoDrag {
    mode: GizmoMode,
    axis: picking::GizmoAxis,
    entity: hecs::Entity,
    start_axis_t: f32,
    start_entity_pos: Vec3,
    start_entity_rot: Quat,
    start_entity_scale: Vec3,
    start_angle: f32,
}

// ── Editor state ──

struct EditorApp {
    camera: EditorCamera,
    camera_entity: Option<hecs::Entity>,
    selected: Option<hecs::Entity>,
    tree_state: esox_engine::esox_ui::TreeState,
    exit: bool,
    // Gizmo materials
    gizmo_mats: Option<GizmoMaterials>,
    gizmo_meshes: Option<GizmoMeshes>,
    gizmo_mode: GizmoMode,
    // Grid
    grid_mesh: Option<MeshHandle>,
    grid_mat: Option<MaterialHandle>,
    // Scene tracking
    scene_path: Option<String>,
    dirty: bool,
    // Pending mutations from UI
    pending_edits: Vec<PendingEdit>,
    // Pending menu actions
    pending_menu_action: Option<u64>,
    // Gizmo interaction
    gizmo_drag: Option<GizmoDrag>,
    // Undo/redo
    undo_stack: UndoStack,
    // Toast notification
    toast_pending: Option<(ToastKind, String)>,
    // Unsaved changes modal
    unsaved_modal_open: bool,
    unsaved_modal_deferred_action: Option<u64>,
    // Keyboard shortcuts modal
    shortcuts_modal_open: bool,
    // Default mesh+material for adding MeshRenderer components
    default_mesh: Option<MeshHandle>,
    default_material: Option<MaterialHandle>,
    // Render settings
    render_settings_open: bool,
    postprocess_config: PostProcess3DConfig,
    shadow_config: ShadowConfig,
    render_config_dirty: bool,
}

#[allow(dead_code)]
struct GizmoMaterials {
    red: MaterialHandle,
    green: MaterialHandle,
    blue: MaterialHandle,
    white: MaterialHandle,
}

#[allow(dead_code)]
struct GizmoMeshes {
    arrow_mesh: MeshHandle,
    cube_mesh: MeshHandle,
    ring_mesh: MeshHandle,
    wire_mesh: MeshHandle,
}

impl EditorApp {
    fn new() -> Self {
        Self {
            camera: EditorCamera::new(),
            camera_entity: None,
            selected: None,
            tree_state: esox_engine::esox_ui::TreeState::new(),
            exit: false,
            gizmo_mats: None,
            gizmo_meshes: None,
            gizmo_mode: GizmoMode::Translate,
            grid_mesh: None,
            grid_mat: None,
            scene_path: None,
            dirty: false,
            pending_edits: Vec::new(),
            pending_menu_action: None,
            gizmo_drag: None,
            undo_stack: UndoStack::new(),
            toast_pending: None,
            unsaved_modal_open: false,
            unsaved_modal_deferred_action: None,
            shortcuts_modal_open: false,
            default_mesh: None,
            default_material: None,
            render_settings_open: false,
            postprocess_config: PostProcess3DConfig::default(),
            shadow_config: ShadowConfig::default(),
            render_config_dirty: false,
        }
    }

    fn sync_camera_entity(&self, ctx: &mut Ctx) {
        if let Some(cam_entity) = self.camera_entity {
            if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(cam_entity) {
                t.position = self.camera.position();
                t.rotation = self.camera.rotation();
            }
        }
    }
}

impl Game for EditorApp {
    fn init(&mut self, ctx: &mut Ctx) {
        // Bind editor-specific input
        ctx.input
            .bind_action("exit", ActionBinding::Key(KeyCode::Escape));
        ctx.input
            .bind_action("pick", ActionBinding::MouseButton(0));
        ctx.input
            .bind_action("focus", ActionBinding::Key(KeyCode::KeyF));
        ctx.input
            .bind_action("delete", ActionBinding::Key(KeyCode::Delete));

        // Create camera entity
        let cam_entity = ctx.world.spawn((
            Transform3D {
                position: self.camera.position(),
                rotation: self.camera.rotation(),
                ..Default::default()
            },
            GlobalTransform::default(),
            Camera3D {
                fov_y: FRAC_PI_4,
                near: 0.1,
                far: 500.0,
                active: true,
            },
        ));
        self.camera_entity = Some(cam_entity);

        // Create a default directional light
        ctx.world.spawn((
            Transform3D {
                rotation: Quat::from_euler(
                    glam::EulerRot::XYZ,
                    -0.8,
                    0.5,
                    0.0,
                ),
                ..Default::default()
            },
            GlobalTransform::default(),
            DirectionalLightComponent {
                color: [1.0, 0.95, 0.9],
                intensity: 2.0,
            },
            Tag("Sun".to_string()),
        ));

        // Enable post-processing and shadows
        ctx.renderer.enable_postprocess(ctx.gpu);
        ctx.renderer.enable_ssao(ctx.gpu);
        ctx.renderer.enable_motion_blur(ctx.gpu);
        ctx.renderer.enable_shadows(ctx.gpu);
        ctx.renderer.enable_point_shadows(ctx.gpu);
        ctx.renderer.enable_spot_shadows(ctx.gpu);
        ctx.renderer.generate_procedural_ibl(ctx.gpu);

        // Initialize render config from engine defaults
        if let Some(pp) = ctx.renderer.postprocess_config() {
            self.postprocess_config = pp;
        }
        if let Some(sc) = ctx.renderer.shadow_config() {
            self.shadow_config = sc;
        }

        // Create gizmo materials (unlit, bright colors)
        let red = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::Unlit,
                albedo: [1.0, 0.2, 0.2, 1.0],
                ..Default::default()
            },
        );
        let green = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::Unlit,
                albedo: [0.2, 1.0, 0.2, 1.0],
                ..Default::default()
            },
        );
        let blue = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::Unlit,
                albedo: [0.2, 0.2, 1.0, 1.0],
                ..Default::default()
            },
        );
        let white = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::Unlit,
                albedo: [0.6, 0.6, 0.6, 0.8],
                ..Default::default()
            },
        );
        self.gizmo_mats = Some(GizmoMaterials {
            red,
            green,
            blue,
            white,
        });

        // Create gizmo meshes
        let arrow_mesh = ctx
            .renderer
            .upload_mesh(ctx.gpu, &MeshData::cylinder(0.03, 1.0, 8));
        let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::cube(0.08));
        let ring_mesh = ctx
            .renderer
            .upload_mesh(ctx.gpu, &MeshData::torus(1.0, 0.02, 32, 8));
        let wire_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::cube(1.0));
        self.gizmo_meshes = Some(GizmoMeshes {
            arrow_mesh,
            cube_mesh,
            ring_mesh,
            wire_mesh,
        });

        // Create ground grid mesh
        let grid = generate_grid(20, 1.0);
        let grid_mesh = ctx.renderer.upload_mesh(ctx.gpu, &grid);
        let grid_mat = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::Unlit,
                albedo: [0.3, 0.3, 0.3, 0.5],
                ..Default::default()
            },
        );
        self.grid_mesh = Some(grid_mesh);
        self.grid_mat = Some(grid_mat);

        // Default mesh+material for "Add MeshRenderer" component
        let default_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::cube(1.0));
        let default_material = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [0.7, 0.7, 0.7, 1.0],
                roughness: 0.5,
                metallic: 0.0,
                ..Default::default()
            },
        );
        self.default_mesh = Some(default_mesh);
        self.default_material = Some(default_material);

        // Spawn a few default objects so the scene isn't empty
        let floor_mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::cube(1.0));
        let floor_mat = ctx.renderer.create_material(
            ctx.gpu,
            &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: [0.4, 0.4, 0.4, 1.0],
                roughness: 0.8,
                metallic: 0.0,
                ..Default::default()
            },
        );
        ctx.world.spawn((
            Transform3D {
                position: Vec3::new(0.0, -0.5, 0.0),
                scale: Vec3::new(20.0, 1.0, 20.0),
                ..Default::default()
            },
            GlobalTransform::default(),
            MeshRenderer {
                mesh: floor_mesh,
                material: floor_mat,
                tint: [1.0; 4],
                visible: true,
            },
            Tag("Floor".to_string()),
        ));
    }

    fn update(&mut self, ctx: &mut Ctx) {
        // Apply pending edits from UI
        if !self.pending_edits.is_empty() {
            self.dirty = true;
        }
        let elapsed = ctx.time.elapsed;
        for edit in self.pending_edits.drain(..) {
            match edit {
                PendingEdit::SetTransform(entity, t) => {
                    if let Ok(mut tr) = ctx.world.get::<&mut Transform3D>(entity) {
                        let old = *tr;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditTransform { entity, old, new: t },
                            elapsed,
                        );
                        *tr = t;
                    }
                }
                PendingEdit::SetPointLightIntensity(entity, v) => {
                    if let Ok(mut pl) = ctx.world.get::<&mut PointLightComponent>(entity) {
                        let old = pl.intensity;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditPointLightIntensity { entity, old, new: v },
                            elapsed,
                        );
                        pl.intensity = v;
                    }
                }
                PendingEdit::SetPointLightRange(entity, v) => {
                    if let Ok(mut pl) = ctx.world.get::<&mut PointLightComponent>(entity) {
                        let old = pl.range;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditPointLightRange { entity, old, new: v },
                            elapsed,
                        );
                        pl.range = v;
                    }
                }
                PendingEdit::SetSpotLightIntensity(entity, v) => {
                    if let Ok(mut sl) = ctx.world.get::<&mut SpotLightComponent>(entity) {
                        let old = sl.intensity;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditSpotLightIntensity { entity, old, new: v },
                            elapsed,
                        );
                        sl.intensity = v;
                    }
                }
                PendingEdit::SetSpotLightRange(entity, v) => {
                    if let Ok(mut sl) = ctx.world.get::<&mut SpotLightComponent>(entity) {
                        let old = sl.range;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditSpotLightRange { entity, old, new: v },
                            elapsed,
                        );
                        sl.range = v;
                    }
                }
                PendingEdit::SetDirLightIntensity(entity, v) => {
                    if let Ok(mut dl) = ctx.world.get::<&mut DirectionalLightComponent>(entity) {
                        let old = dl.intensity;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditDirLightIntensity { entity, old, new: v },
                            elapsed,
                        );
                        dl.intensity = v;
                    }
                }
                PendingEdit::SetPointLightColor(entity, v) => {
                    if let Ok(mut pl) = ctx.world.get::<&mut PointLightComponent>(entity) {
                        let old = pl.color;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditPointLightColor { entity, old, new: v },
                            elapsed,
                        );
                        pl.color = v;
                    }
                }
                PendingEdit::SetSpotLightColor(entity, v) => {
                    if let Ok(mut sl) = ctx.world.get::<&mut SpotLightComponent>(entity) {
                        let old = sl.color;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditSpotLightColor { entity, old, new: v },
                            elapsed,
                        );
                        sl.color = v;
                    }
                }
                PendingEdit::SetSpotLightInnerCone(entity, v) => {
                    if let Ok(mut sl) = ctx.world.get::<&mut SpotLightComponent>(entity) {
                        let old = sl.inner_cone_angle;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditSpotLightInnerCone { entity, old, new: v },
                            elapsed,
                        );
                        sl.inner_cone_angle = v;
                    }
                }
                PendingEdit::SetSpotLightOuterCone(entity, v) => {
                    if let Ok(mut sl) = ctx.world.get::<&mut SpotLightComponent>(entity) {
                        let old = sl.outer_cone_angle;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditSpotLightOuterCone { entity, old, new: v },
                            elapsed,
                        );
                        sl.outer_cone_angle = v;
                    }
                }
                PendingEdit::SetDirLightColor(entity, v) => {
                    if let Ok(mut dl) = ctx.world.get::<&mut DirectionalLightComponent>(entity) {
                        let old = dl.color;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditDirLightColor { entity, old, new: v },
                            elapsed,
                        );
                        dl.color = v;
                    }
                }
                PendingEdit::SetTag(entity, v) => {
                    if let Ok(mut tag) = ctx.world.get::<&mut Tag>(entity) {
                        let old = tag.0.clone();
                        self.undo_stack.push_or_merge(
                            UndoAction::EditTag { entity, old, new: v.clone() },
                            elapsed,
                        );
                        tag.0 = v;
                    }
                }
                PendingEdit::SetCameraFov(entity, v) => {
                    if let Ok(mut cam) = ctx.world.get::<&mut Camera3D>(entity) {
                        let old = cam.fov_y;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditCameraFov { entity, old, new: v },
                            elapsed,
                        );
                        cam.fov_y = v;
                    }
                }
                PendingEdit::SetCameraNear(entity, v) => {
                    if let Ok(mut cam) = ctx.world.get::<&mut Camera3D>(entity) {
                        let old = cam.near;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditCameraNear { entity, old, new: v },
                            elapsed,
                        );
                        cam.near = v;
                    }
                }
                PendingEdit::SetCameraFar(entity, v) => {
                    if let Ok(mut cam) = ctx.world.get::<&mut Camera3D>(entity) {
                        let old = cam.far;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditCameraFar { entity, old, new: v },
                            elapsed,
                        );
                        cam.far = v;
                    }
                }
                PendingEdit::SetMeshVisible(entity, v) => {
                    if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(entity) {
                        let old = mr.visible;
                        self.undo_stack.push(
                            UndoAction::EditMeshVisible { entity, old, new: v },
                        );
                        mr.visible = v;
                    }
                }
                PendingEdit::SetMeshTint(entity, v) => {
                    if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(entity) {
                        let old = mr.tint;
                        self.undo_stack.push_or_merge(
                            UndoAction::EditMeshTint { entity, old, new: v },
                            elapsed,
                        );
                        mr.tint = v;
                    }
                }
                PendingEdit::SetPointLightShadows(entity, v) => {
                    if let Ok(mut pl) = ctx.world.get::<&mut PointLightComponent>(entity) {
                        let old = pl.cast_shadows;
                        self.undo_stack.push(
                            UndoAction::EditPointLightShadows { entity, old, new: v },
                        );
                        pl.cast_shadows = v;
                    }
                }
                PendingEdit::SetSpotLightShadows(entity, v) => {
                    if let Ok(mut sl) = ctx.world.get::<&mut SpotLightComponent>(entity) {
                        let old = sl.cast_shadows;
                        self.undo_stack.push(
                            UndoAction::EditSpotLightShadows { entity, old, new: v },
                        );
                        sl.cast_shadows = v;
                    }
                }
                PendingEdit::AddComponent(entity, kind) => {
                    if ctx.world.contains(entity) {
                        match kind {
                            ComponentKind::PointLight => {
                                let comp = PointLightComponent {
                                    color: [1.0, 0.9, 0.8], intensity: 10.0,
                                    range: 15.0, cast_shadows: false,
                                };
                                let _ = ctx.world.insert_one(entity, comp);
                                self.undo_stack.push(UndoAction::AddComponent {
                                    entity, kind,
                                    snapshot: ComponentSnapshot::PointLight(comp),
                                });
                            }
                            ComponentKind::SpotLight => {
                                let comp = SpotLightComponent {
                                    color: [1.0, 1.0, 1.0], intensity: 20.0,
                                    range: 20.0, inner_cone_angle: 0.3,
                                    outer_cone_angle: 0.5, cast_shadows: false,
                                };
                                let _ = ctx.world.insert_one(entity, comp);
                                self.undo_stack.push(UndoAction::AddComponent {
                                    entity, kind,
                                    snapshot: ComponentSnapshot::SpotLight(comp),
                                });
                            }
                            ComponentKind::DirLight => {
                                let comp = DirectionalLightComponent {
                                    color: [1.0, 1.0, 1.0], intensity: 2.0,
                                };
                                let _ = ctx.world.insert_one(entity, comp);
                                self.undo_stack.push(UndoAction::AddComponent {
                                    entity, kind,
                                    snapshot: ComponentSnapshot::DirLight(comp),
                                });
                            }
                            ComponentKind::Camera => {
                                let comp = Camera3D::default();
                                let _ = ctx.world.insert_one(entity, comp);
                                self.undo_stack.push(UndoAction::AddComponent {
                                    entity, kind,
                                    snapshot: ComponentSnapshot::Camera(comp),
                                });
                            }
                            ComponentKind::MeshRenderer => {
                                if let (Some(mesh), Some(material)) = (self.default_mesh, self.default_material) {
                                    let comp = MeshRenderer {
                                        mesh, material,
                                        tint: [1.0; 4], visible: true,
                                    };
                                    let _ = ctx.world.insert_one(entity, comp);
                                    self.undo_stack.push(UndoAction::AddComponent {
                                        entity, kind,
                                        snapshot: ComponentSnapshot::MeshRenderer(comp),
                                    });
                                }
                            }
                        }
                    }
                }
                PendingEdit::RemoveComponent(entity, kind) => {
                    if ctx.world.contains(entity) {
                        let snapshot = match kind {
                            ComponentKind::PointLight => ctx.world.get::<&PointLightComponent>(entity).ok().map(|c| ComponentSnapshot::PointLight(*c)),
                            ComponentKind::SpotLight => ctx.world.get::<&SpotLightComponent>(entity).ok().map(|c| ComponentSnapshot::SpotLight(*c)),
                            ComponentKind::DirLight => ctx.world.get::<&DirectionalLightComponent>(entity).ok().map(|c| ComponentSnapshot::DirLight(*c)),
                            ComponentKind::Camera => ctx.world.get::<&Camera3D>(entity).ok().map(|c| ComponentSnapshot::Camera(*c)),
                            ComponentKind::MeshRenderer => ctx.world.get::<&MeshRenderer>(entity).ok().map(|c| ComponentSnapshot::MeshRenderer(*c)),
                        };
                        if let Some(snapshot) = snapshot {
                            match kind {
                                ComponentKind::PointLight => { let _ = ctx.world.remove_one::<PointLightComponent>(entity); }
                                ComponentKind::SpotLight => { let _ = ctx.world.remove_one::<SpotLightComponent>(entity); }
                                ComponentKind::DirLight => { let _ = ctx.world.remove_one::<DirectionalLightComponent>(entity); }
                                ComponentKind::Camera => { let _ = ctx.world.remove_one::<Camera3D>(entity); }
                                ComponentKind::MeshRenderer => { let _ = ctx.world.remove_one::<MeshRenderer>(entity); }
                            }
                            self.undo_stack.push(UndoAction::RemoveComponent { entity, kind, snapshot });
                        }
                    }
                }
                PendingEdit::SetMesh(entity, handle) => {
                    if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(entity) {
                        let old = mr.mesh;
                        self.undo_stack.push(UndoAction::EditMesh { entity, old, new: handle });
                        mr.mesh = handle;
                    }
                }
                PendingEdit::SetMaterial(entity, handle) => {
                    if let Ok(mut mr) = ctx.world.get::<&mut MeshRenderer>(entity) {
                        let old = mr.material;
                        self.undo_stack.push(UndoAction::EditMaterial { entity, old, new: handle });
                        mr.material = handle;
                    }
                }
                PendingEdit::SetMaterialDescriptor(entity, new_desc) => {
                    if let Ok(mr) = ctx.world.get::<&MeshRenderer>(entity) {
                        let handle = mr.material;
                        if let Some(old_desc) = ctx.renderer.material_descriptor(handle) {
                            let old_desc = old_desc.clone();
                            self.undo_stack.push_or_merge(
                                UndoAction::EditMaterialDescriptor {
                                    entity, handle, old: old_desc, new: new_desc.clone(),
                                },
                                elapsed,
                            );
                            ctx.renderer.update_material(ctx.gpu, handle, &new_desc);
                        }
                    }
                }
            }
        }

        // Apply render config changes
        if self.render_config_dirty {
            ctx.renderer.set_postprocess(self.postprocess_config);
            ctx.renderer.set_shadow_config(self.shadow_config);
            self.render_config_dirty = false;
        }

        // Handle pending menu actions
        if let Some(action) = self.pending_menu_action.take() {
            self.handle_menu_action(action, ctx);
        }

        if ctx.input.just_pressed("exit") {
            self.exit = true;
        }

        // Helper: build camera matrices for picking
        let cam_pos = self.camera.position();
        let cam_target = match self.camera.mode {
            CameraMode::Orbit => self.camera.orbit_target,
            CameraMode::Fly => cam_pos + self.camera.forward(),
        };
        let view = Mat4::look_at_rh(cam_pos, cam_target, Vec3::Y);
        let aspect = ctx.viewport.0 as f32 / ctx.viewport.1.max(1) as f32;
        let projection = Mat4::perspective_rh(FRAC_PI_4, aspect, 0.1, 500.0);

        // Gizmo drag: continue or end
        if let Some(ref drag) = self.gizmo_drag {
            if ctx.input.is_mouse_button_down(0) {
                let (mx, my) = ctx.input.mouse_pos();
                let (ray_origin, ray_dir) =
                    picking::screen_to_ray(mx, my, ctx.viewport, view, projection);

                let axis_dir = match drag.axis {
                    picking::GizmoAxis::X => Vec3::X,
                    picking::GizmoAxis::Y => Vec3::Y,
                    picking::GizmoAxis::Z => Vec3::Z,
                };

                match drag.mode {
                    GizmoMode::Translate => {
                        let t = picking::closest_point_on_axis(
                            ray_origin,
                            ray_dir,
                            drag.start_entity_pos,
                            axis_dir,
                        );
                        let delta_t = t - drag.start_axis_t;
                        let mut new_pos = drag.start_entity_pos + axis_dir * delta_t;
                        // Ctrl: snap to 1.0 grid
                        let ctrl_now = ctx.input.is_key_down(KeyCode::ControlLeft)
                            || ctx.input.is_key_down(KeyCode::ControlRight);
                        if ctrl_now {
                            new_pos = Vec3::new(
                                new_pos.x.round(),
                                new_pos.y.round(),
                                new_pos.z.round(),
                            );
                        }
                        if let Ok(mut tr) = ctx.world.get::<&mut Transform3D>(drag.entity) {
                            tr.position = new_pos;
                        }
                    }
                    GizmoMode::Scale => {
                        let cam_dist = (drag.start_entity_pos - cam_pos).length();
                        let gizmo_scale = cam_dist / 8.0;
                        let t = picking::closest_point_on_axis(
                            ray_origin,
                            ray_dir,
                            drag.start_entity_pos,
                            axis_dir,
                        );
                        let delta_t = t - drag.start_axis_t;
                        let factor = 1.0 + delta_t / gizmo_scale;
                        let mut new_scale = drag.start_entity_scale;
                        match drag.axis {
                            picking::GizmoAxis::X => new_scale.x = (drag.start_entity_scale.x * factor).max(0.01),
                            picking::GizmoAxis::Y => new_scale.y = (drag.start_entity_scale.y * factor).max(0.01),
                            picking::GizmoAxis::Z => new_scale.z = (drag.start_entity_scale.z * factor).max(0.01),
                        }
                        // Ctrl: snap to 0.25
                        let ctrl_now = ctx.input.is_key_down(KeyCode::ControlLeft)
                            || ctx.input.is_key_down(KeyCode::ControlRight);
                        if ctrl_now {
                            new_scale = Vec3::new(
                                (new_scale.x / 0.25).round() * 0.25,
                                (new_scale.y / 0.25).round() * 0.25,
                                (new_scale.z / 0.25).round() * 0.25,
                            );
                            new_scale = new_scale.max(Vec3::splat(0.01));
                        }
                        if let Ok(mut tr) = ctx.world.get::<&mut Transform3D>(drag.entity) {
                            tr.scale = new_scale;
                        }
                    }
                    GizmoMode::Rotate => {
                        if let Some(hit) = picking::project_ray_to_plane(
                            ray_origin,
                            ray_dir,
                            drag.start_entity_pos,
                            axis_dir,
                        ) {
                            let current_angle = picking::angle_on_plane(
                                hit,
                                drag.start_entity_pos,
                                axis_dir,
                            );
                            let mut delta = current_angle - drag.start_angle;
                            // Ctrl: snap to 15 degrees
                            let ctrl_now = ctx.input.is_key_down(KeyCode::ControlLeft)
                                || ctx.input.is_key_down(KeyCode::ControlRight);
                            if ctrl_now {
                                let step = PI / 12.0;
                                delta = (delta / step).round() * step;
                            }
                            let new_rot =
                                Quat::from_axis_angle(axis_dir, delta) * drag.start_entity_rot;
                            if let Ok(mut tr) = ctx.world.get::<&mut Transform3D>(drag.entity) {
                                tr.rotation = new_rot;
                            }
                        }
                    }
                }
            } else {
                // LMB released — end drag
                let drag = self.gizmo_drag.take().unwrap();
                match drag.mode {
                    GizmoMode::Translate => {
                        if let Ok(t) = ctx.world.get::<&Transform3D>(drag.entity) {
                            let final_pos = t.position;
                            if final_pos != drag.start_entity_pos {
                                self.undo_stack.push(UndoAction::GizmoDrag {
                                    entity: drag.entity,
                                    old_pos: drag.start_entity_pos,
                                    new_pos: final_pos,
                                });
                            }
                        }
                    }
                    GizmoMode::Scale => {
                        if let Ok(t) = ctx.world.get::<&Transform3D>(drag.entity) {
                            let final_scale = t.scale;
                            if final_scale != drag.start_entity_scale {
                                self.undo_stack.push(UndoAction::GizmoScale {
                                    entity: drag.entity,
                                    old_scale: drag.start_entity_scale,
                                    new_scale: final_scale,
                                });
                            }
                        }
                    }
                    GizmoMode::Rotate => {
                        if let Ok(t) = ctx.world.get::<&Transform3D>(drag.entity) {
                            let final_rot = t.rotation;
                            if final_rot != drag.start_entity_rot {
                                self.undo_stack.push(UndoAction::GizmoRotate {
                                    entity: drag.entity,
                                    old_rot: drag.start_entity_rot,
                                    new_rot: final_rot,
                                });
                            }
                        }
                    }
                }
                self.dirty = true;
            }
        }

        // Left-click picking (only when no MMB/RMB held, not dragging gizmo)
        if self.gizmo_drag.is_none()
            && ctx.input.just_pressed("pick")
            && !ctx.input.is_mouse_button_down(1)
            && !ctx.input.is_mouse_button_down(2)
        {
            let (mx, my) = ctx.input.mouse_pos();
            let (ray_origin, ray_dir) =
                picking::screen_to_ray(mx, my, ctx.viewport, view, projection);

            // Try gizmo picking first
            let mut gizmo_hit = false;
            if let Some(selected) = self.selected {
                if let Ok(gt) = ctx.world.get::<&GlobalTransform>(selected) {
                    let world_pos = Vec3::new(gt.0.col(3).x, gt.0.col(3).y, gt.0.col(3).z);
                    let cam_dist = (world_pos - cam_pos).length();
                    let gizmo_scale = cam_dist / 8.0;

                    if let Some(axis) =
                        picking::pick_gizmo_axis(ray_origin, ray_dir, world_pos, gizmo_scale)
                    {
                        let axis_dir = match axis {
                            picking::GizmoAxis::X => Vec3::X,
                            picking::GizmoAxis::Y => Vec3::Y,
                            picking::GizmoAxis::Z => Vec3::Z,
                        };

                        // Read current transform for all start fields
                        let (start_rot, start_scale) =
                            if let Ok(tr) = ctx.world.get::<&Transform3D>(selected) {
                                (tr.rotation, tr.scale)
                            } else {
                                (Quat::IDENTITY, Vec3::ONE)
                            };

                        let start_t = picking::closest_point_on_axis(
                            ray_origin,
                            ray_dir,
                            world_pos,
                            axis_dir,
                        );

                        let start_angle = if self.gizmo_mode == GizmoMode::Rotate {
                            if let Some(hit) = picking::project_ray_to_plane(
                                ray_origin, ray_dir, world_pos, axis_dir,
                            ) {
                                picking::angle_on_plane(hit, world_pos, axis_dir)
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };

                        self.gizmo_drag = Some(GizmoDrag {
                            mode: self.gizmo_mode,
                            axis,
                            entity: selected,
                            start_axis_t: start_t,
                            start_entity_pos: world_pos,
                            start_entity_rot: start_rot,
                            start_entity_scale: start_scale,
                            start_angle,
                        });
                        gizmo_hit = true;
                    }
                }
            }

            // Fall through to entity picking if no gizmo was hit
            if !gizmo_hit {
                if let Some((entity, _dist)) =
                    picking::pick_entity(ctx, ray_origin, ray_dir, self.camera_entity)
                {
                    self.selected = Some(entity);
                }
            }
        }

        // Focus on selected (F key)
        if ctx.input.just_pressed("focus") {
            if let Some(selected) = self.selected {
                if let Ok(gt) = ctx.world.get::<&GlobalTransform>(selected) {
                    let pos = Vec3::new(gt.0.col(3).x, gt.0.col(3).y, gt.0.col(3).z);
                    self.camera.focus_on(pos);
                }
            }
        }

        // Delete key
        if ctx.input.just_pressed("delete") {
            if let Some(selected) = self.selected.take() {
                if self.camera_entity != Some(selected) {
                    let snapshot = EntitySnapshot::capture(ctx.world, selected);
                    let _ = ctx.world.despawn(selected);
                    self.undo_stack.push(UndoAction::DeleteEntity {
                        stale_entity: selected,
                        snapshot,
                    });
                    self.dirty = true;
                }
            }
        }

        // Undo/Redo (Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y)
        let ctrl = ctx.input.is_key_down(KeyCode::ControlLeft)
            || ctx.input.is_key_down(KeyCode::ControlRight);
        let shift = ctx.input.is_key_down(KeyCode::ShiftLeft)
            || ctx.input.is_key_down(KeyCode::ShiftRight);
        if ctrl {
            if ctx.input.just_pressed_key(KeyCode::KeyZ) {
                if shift {
                    // Redo
                    if let Some(result) = self.undo_stack.redo(ctx.world) {
                        if let Some(sel) = result.select {
                            self.selected = sel;
                        }
                        if let Some((handle, desc)) = result.material_update {
                            ctx.renderer.update_material(ctx.gpu, handle, &desc);
                        }
                        self.dirty = true;
                    }
                } else {
                    // Undo
                    if let Some(result) = self.undo_stack.undo(ctx.world) {
                        if let Some(sel) = result.select {
                            self.selected = sel;
                        }
                        if let Some((handle, desc)) = result.material_update {
                            ctx.renderer.update_material(ctx.gpu, handle, &desc);
                        }
                        self.dirty = true;
                    }
                }
            }
            if ctx.input.just_pressed_key(KeyCode::KeyY) {
                // Redo (alternative)
                if let Some(result) = self.undo_stack.redo(ctx.world) {
                    if let Some(sel) = result.select {
                        self.selected = sel;
                    }
                    self.dirty = true;
                }
            }
            if ctx.input.just_pressed_key(KeyCode::KeyD) {
                // Duplicate
                self.pending_menu_action = Some(13);
            }
        }

        // Gizmo tool switching (W/E/R, only when not dragging and ctrl not held)
        let rmb = ctx.input.is_mouse_button_down(2);
        if self.gizmo_drag.is_none() && !ctrl && !rmb {
            if ctx.input.just_pressed_key(KeyCode::KeyW) {
                self.gizmo_mode = GizmoMode::Translate;
            }
            if ctx.input.just_pressed_key(KeyCode::KeyE) {
                self.gizmo_mode = GizmoMode::Rotate;
            }
            if ctx.input.just_pressed_key(KeyCode::KeyR) {
                self.gizmo_mode = GizmoMode::Scale;
            }
        }

        self.camera.update(ctx);

        // Grab cursor while orbiting (MMB) or flying (RMB).
        let grab = ctx.input.is_mouse_button_down(1) || ctx.input.is_mouse_button_down(2);
        ctx.input.set_cursor_grab(grab);

        self.sync_camera_entity(ctx);
    }

    fn render(&mut self, ctx: &mut Ctx, _alpha: f32) {
        // Draw ground grid
        if let (Some(grid_mesh), Some(grid_mat)) = (self.grid_mesh, self.grid_mat) {
            ctx.renderer.draw_with_material(
                grid_mesh,
                grid_mat,
                &[InstanceData::from_transform(
                    &esox_engine::esox_gfx::mesh3d::Transform::default(),
                )],
            );
        }

        // Draw selection wireframe + gizmo
        if let Some(selected) = self.selected {
            if let (Some(gizmo_mats), Some(gizmo_meshes)) =
                (&self.gizmo_mats, &self.gizmo_meshes)
            {
                // Selection wireframe box
                if let Ok(gt) = ctx.world.get::<&GlobalTransform>(selected) {
                    let (aabb_min, aabb_max) =
                        if let Ok(mr) = ctx.world.get::<&MeshRenderer>(selected) {
                            if let Some(local_aabb) = ctx.renderer.mesh_local_aabb(mr.mesh) {
                                let world_aabb = local_aabb.transformed(&gt.0);
                                (world_aabb.min, world_aabb.max)
                            } else {
                                let pos = Vec3::new(gt.0.col(3).x, gt.0.col(3).y, gt.0.col(3).z);
                                (pos - Vec3::splat(0.3), pos + Vec3::splat(0.3))
                            }
                        } else {
                            let pos = Vec3::new(gt.0.col(3).x, gt.0.col(3).y, gt.0.col(3).z);
                            (pos - Vec3::splat(0.3), pos + Vec3::splat(0.3))
                        };

                    for edge_t in wireframe_aabb_edges(aabb_min, aabb_max) {
                        ctx.renderer.draw_with_material(
                            gizmo_meshes.wire_mesh,
                            gizmo_mats.white,
                            &[InstanceData::from_transform(&edge_t)],
                        );
                    }
                }

                // Draw gizmo at selected entity's position
                if let Ok(gt) = ctx.world.get::<&GlobalTransform>(selected) {
                    let world_pos =
                        Vec3::new(gt.0.col(3).x, gt.0.col(3).y, gt.0.col(3).z);
                    let cam_dist = (world_pos - self.camera.position()).length();
                    let gizmo_scale = cam_dist / 8.0;

                    match self.gizmo_mode {
                        GizmoMode::Translate => {
                            // X axis (red)
                            let x_transform = esox_engine::esox_gfx::mesh3d::Transform {
                                position: world_pos + Vec3::X * gizmo_scale * 0.5,
                                rotation: Quat::from_rotation_z(-FRAC_PI_2),
                                scale: Vec3::splat(gizmo_scale),
                            };
                            ctx.renderer.draw_with_material(
                                gizmo_meshes.arrow_mesh,
                                gizmo_mats.red,
                                &[InstanceData::from_transform(&x_transform)],
                            );

                            // Y axis (green)
                            let y_transform = esox_engine::esox_gfx::mesh3d::Transform {
                                position: world_pos + Vec3::Y * gizmo_scale * 0.5,
                                rotation: Quat::IDENTITY,
                                scale: Vec3::splat(gizmo_scale),
                            };
                            ctx.renderer.draw_with_material(
                                gizmo_meshes.arrow_mesh,
                                gizmo_mats.green,
                                &[InstanceData::from_transform(&y_transform)],
                            );

                            // Z axis (blue)
                            let z_transform = esox_engine::esox_gfx::mesh3d::Transform {
                                position: world_pos + Vec3::Z * gizmo_scale * 0.5,
                                rotation: Quat::from_rotation_x(FRAC_PI_2),
                                scale: Vec3::splat(gizmo_scale),
                            };
                            ctx.renderer.draw_with_material(
                                gizmo_meshes.arrow_mesh,
                                gizmo_mats.blue,
                                &[InstanceData::from_transform(&z_transform)],
                            );
                        }
                        GizmoMode::Scale => {
                            // Scale gizmo: thin shafts + cube endpoints
                            let axes = [
                                (Vec3::X, Quat::from_rotation_z(-FRAC_PI_2), gizmo_mats.red),
                                (Vec3::Y, Quat::IDENTITY, gizmo_mats.green),
                                (Vec3::Z, Quat::from_rotation_x(FRAC_PI_2), gizmo_mats.blue),
                            ];
                            for (dir, rot, mat) in axes {
                                // Shaft
                                let shaft_t = esox_engine::esox_gfx::mesh3d::Transform {
                                    position: world_pos + dir * gizmo_scale * 0.5,
                                    rotation: rot,
                                    scale: Vec3::splat(gizmo_scale),
                                };
                                ctx.renderer.draw_with_material(
                                    gizmo_meshes.arrow_mesh,
                                    mat,
                                    &[InstanceData::from_transform(&shaft_t)],
                                );
                                // Cube endpoint
                                let cube_t = esox_engine::esox_gfx::mesh3d::Transform {
                                    position: world_pos + dir * gizmo_scale,
                                    rotation: Quat::IDENTITY,
                                    scale: Vec3::splat(gizmo_scale),
                                };
                                ctx.renderer.draw_with_material(
                                    gizmo_meshes.cube_mesh,
                                    mat,
                                    &[InstanceData::from_transform(&cube_t)],
                                );
                            }
                        }
                        GizmoMode::Rotate => {
                            // Rotation gizmo: 3 torus rings
                            let rings = [
                                // Y ring: torus naturally around Y
                                (Quat::IDENTITY, gizmo_mats.green),
                                // X ring: rotate Y-axis torus to X
                                (Quat::from_rotation_z(-FRAC_PI_2), gizmo_mats.red),
                                // Z ring: rotate Y-axis torus to Z
                                (Quat::from_rotation_x(FRAC_PI_2), gizmo_mats.blue),
                            ];
                            for (rot, mat) in rings {
                                let ring_t = esox_engine::esox_gfx::mesh3d::Transform {
                                    position: world_pos,
                                    rotation: rot,
                                    scale: Vec3::splat(gizmo_scale),
                                };
                                ctx.renderer.draw_with_material(
                                    gizmo_meshes.ring_mesh,
                                    mat,
                                    &[InstanceData::from_transform(&ring_t)],
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn ui(&mut self, ui: &mut esox_engine::esox_ui::Ui, ctx: &Ctx) {
        use esox_engine::esox_ui::{InputState, Menu, MenuEntry, MenuItem, ModalAction};

        // Fire pending toast
        if let Some((kind, msg)) = self.toast_pending.take() {
            match kind {
                ToastKind::Success => ui.toast_success(&msg),
                ToastKind::Error => ui.toast_error(&msg),
                ToastKind::Info => ui.toast_info(&msg),
                ToastKind::Warning => ui.toast_warning(&msg),
            }
        }

        // Menu bar
        let file_label = if self.dirty { "File *" } else { "File" };
        let menus = &[
            Menu::new(
                file_label,
                vec![
                    MenuEntry::Item(MenuItem::new("New Scene", 1).with_shortcut("Ctrl+N")),
                    MenuEntry::Item(MenuItem::new("Open Scene", 2).with_shortcut("Ctrl+O")),
                    MenuEntry::Separator,
                    MenuEntry::Item(MenuItem::new("Save", 3).with_shortcut("Ctrl+S")),
                    MenuEntry::Item(
                        MenuItem::new("Save As...", 4).with_shortcut("Ctrl+Shift+S"),
                    ),
                    MenuEntry::Separator,
                    MenuEntry::Item(MenuItem::new("Quit", 9).with_shortcut("Esc")),
                ],
            ),
            Menu::new(
                "Edit",
                vec![
                    MenuEntry::Item({
                        let mut item = MenuItem::new("Undo", 10).with_shortcut("Ctrl+Z");
                        if !self.undo_stack.can_undo() { item = item.disabled(); }
                        item
                    }),
                    MenuEntry::Item({
                        let mut item = MenuItem::new("Redo", 11).with_shortcut("Ctrl+Shift+Z");
                        if !self.undo_stack.can_redo() { item = item.disabled(); }
                        item
                    }),
                    MenuEntry::Separator,
                    MenuEntry::Item(MenuItem::new("Delete Entity", 12).with_shortcut("Del")),
                    MenuEntry::Item(MenuItem::new("Duplicate", 13).with_shortcut("Ctrl+D")),
                ],
            ),
            Menu::new(
                "View",
                vec![
                    MenuEntry::Item(MenuItem::new("Reset Camera", 20)),
                    MenuEntry::Item(MenuItem::new("Focus Selected", 21).with_shortcut("F")),
                    MenuEntry::Separator,
                    MenuEntry::Item(MenuItem::new("Keyboard Shortcuts", 22)),
                ],
            ),
            Menu::new(
                "Render",
                vec![
                    MenuEntry::Item(MenuItem::new("Render Settings", 40)),
                ],
            ),
            Menu::new(
                "Entity",
                vec![
                    MenuEntry::Item(MenuItem::new("Add Empty", 30)),
                    MenuEntry::Item(MenuItem::new("Add Cube", 31)),
                    MenuEntry::Item(MenuItem::new("Add Sphere", 32)),
                    MenuEntry::Separator,
                    MenuEntry::Item(MenuItem::new("Add Point Light", 33)),
                    MenuEntry::Item(MenuItem::new("Add Spot Light", 34)),
                    MenuEntry::Item(MenuItem::new("Add Directional Light", 35)),
                    MenuEntry::Separator,
                    MenuEntry::Item(MenuItem::new("Add Camera", 36)),
                ],
            ),
        ];

        if let Some(action) = ui.menu_bar(menus) {
            // For destructive actions (new/open/quit), check unsaved changes
            if self.dirty && matches!(action, 1 | 2 | 9) {
                self.unsaved_modal_deferred_action = Some(action);
                self.unsaved_modal_open = true;
            } else {
                self.pending_menu_action = Some(action);
            }
        }

        // Unsaved changes modal
        if self.unsaved_modal_open {
            match ui.modal_confirm(
                hash("unsaved_modal"),
                &mut self.unsaved_modal_open,
                "Unsaved Changes",
                "You have unsaved changes. Continue without saving?",
            ) {
                ModalAction::Confirm => {
                    if let Some(action) = self.unsaved_modal_deferred_action.take() {
                        self.pending_menu_action = Some(action);
                    }
                }
                ModalAction::Cancel => {
                    self.unsaved_modal_deferred_action = None;
                }
                ModalAction::None => {}
            }
        }

        // Keyboard shortcuts modal
        if self.shortcuts_modal_open {
            ui.modal(
                hash("shortcuts_modal"),
                &mut self.shortcuts_modal_open,
                "Keyboard Shortcuts",
                400.0,
                |ui| {
                    let shortcuts = [
                        ("W", "Translate gizmo"),
                        ("E", "Rotate gizmo"),
                        ("R", "Scale gizmo"),
                        ("F", "Focus selected"),
                        ("Delete", "Delete entity"),
                        ("Ctrl+Z", "Undo"),
                        ("Ctrl+Shift+Z", "Redo"),
                        ("Ctrl+Y", "Redo (alt)"),
                        ("Ctrl+D", "Duplicate"),
                        ("Ctrl+S", "Save"),
                        ("Ctrl+Shift+S", "Save As"),
                        ("Ctrl+N", "New Scene"),
                        ("Ctrl+O", "Open Scene"),
                        ("MMB drag", "Orbit camera"),
                        ("Shift+MMB", "Pan camera"),
                        ("RMB hold", "Fly camera (WASD+QE)"),
                        ("Scroll", "Zoom"),
                        ("Ctrl+drag", "Grid snap"),
                        ("Esc", "Quit"),
                    ];
                    for (key, desc) in &shortcuts {
                        ui.columns(&[0.35, 0.65], |ui, col| match col {
                            0 => { ui.label(key); }
                            1 => { ui.muted_label(desc); }
                            _ => {}
                        });
                    }
                },
            );
        }

        // Render settings panel
        if self.render_settings_open {
            ui.modal(
                hash("render_settings_modal"),
                &mut self.render_settings_open,
                "Render Settings",
                350.0,
                |ui| {
                    let pp = &mut self.postprocess_config;
                    let sc = &mut self.shadow_config;
                    let dirty = &mut self.render_config_dirty;

                    ui.collapsing_header(hash("rs_postprocess"), "Post-Processing", true, |ui| {
                        // Bloom
                        let bloom_label = if pp.bloom_enabled { "Bloom: ON" } else { "Bloom: OFF" };
                        if ui.button(hash("rs_bloom"), bloom_label).clicked {
                            pp.bloom_enabled = !pp.bloom_enabled;
                            *dirty = true;
                        }
                        if pp.bloom_enabled {
                            ui.muted_label("Intensity");
                            let mut input = InputState::new();
                            input.text = format!("{:.2}", pp.bloom_intensity);
                            if ui.slider(hash("rs_bloom_int"), &mut input, 0.0, 2.0).changed {
                                if let Ok(v) = input.text.parse::<f32>() {
                                    pp.bloom_intensity = v.clamp(0.0, 2.0);
                                    *dirty = true;
                                }
                            }
                            ui.muted_label("Threshold");
                            let mut input = InputState::new();
                            input.text = format!("{:.2}", pp.bloom_threshold);
                            if ui.slider(hash("rs_bloom_thr"), &mut input, 0.0, 5.0).changed {
                                if let Ok(v) = input.text.parse::<f32>() {
                                    pp.bloom_threshold = v.clamp(0.0, 5.0);
                                    *dirty = true;
                                }
                            }
                            ui.muted_label("Soft Knee");
                            let mut input = InputState::new();
                            input.text = format!("{:.2}", pp.bloom_soft_knee);
                            if ui.slider(hash("rs_bloom_knee"), &mut input, 0.0, 1.0).changed {
                                if let Ok(v) = input.text.parse::<f32>() {
                                    pp.bloom_soft_knee = v.clamp(0.0, 1.0);
                                    *dirty = true;
                                }
                            }
                        }

                        // Tone mapping
                        let tm_label = if pp.tone_map_enabled { "Tone Mapping: ON" } else { "Tone Mapping: OFF" };
                        if ui.button(hash("rs_tonemap"), tm_label).clicked {
                            pp.tone_map_enabled = !pp.tone_map_enabled;
                            *dirty = true;
                        }

                        // SSAO
                        let ssao_label = if pp.ssao_enabled { "SSAO: ON" } else { "SSAO: OFF" };
                        if ui.button(hash("rs_ssao"), ssao_label).clicked {
                            pp.ssao_enabled = !pp.ssao_enabled;
                            *dirty = true;
                        }

                        // Motion blur
                        let mb_label = if pp.motion_blur_enabled { "Motion Blur: ON" } else { "Motion Blur: OFF" };
                        if ui.button(hash("rs_motionblur"), mb_label).clicked {
                            pp.motion_blur_enabled = !pp.motion_blur_enabled;
                            *dirty = true;
                        }
                    });

                    ui.collapsing_header(hash("rs_shadows"), "Shadows", true, |ui| {
                        let sh_label = if sc.enabled { "Shadows: ON" } else { "Shadows: OFF" };
                        if ui.button(hash("rs_shadow_en"), sh_label).clicked {
                            sc.enabled = !sc.enabled;
                            *dirty = true;
                        }

                        if sc.enabled {
                            ui.muted_label("Cascade Count");
                            let mut cc = sc.cascade_count as f64;
                            if ui.number_input_clamped(hash("rs_cascades"), &mut cc, 1.0, 2.0, 4.0).changed {
                                sc.cascade_count = cc as usize;
                                *dirty = true;
                            }

                            ui.muted_label("Shadow Distance");
                            let mut input = InputState::new();
                            input.text = format!("{:.0}", sc.shadow_distance);
                            if ui.slider(hash("rs_shadow_dist"), &mut input, 10.0, 500.0).changed {
                                if let Ok(v) = input.text.parse::<f32>() {
                                    sc.shadow_distance = v.clamp(10.0, 500.0);
                                    *dirty = true;
                                }
                            }

                            ui.muted_label("Depth Bias");
                            let mut input = InputState::new();
                            input.text = format!("{:.4}", sc.depth_bias);
                            if ui.slider(hash("rs_depth_bias"), &mut input, 0.0, 0.01).changed {
                                if let Ok(v) = input.text.parse::<f32>() {
                                    sc.depth_bias = v.clamp(0.0, 0.01);
                                    *dirty = true;
                                }
                            }

                            ui.muted_label("Normal Bias");
                            let mut input = InputState::new();
                            input.text = format!("{:.4}", sc.normal_bias);
                            if ui.slider(hash("rs_normal_bias"), &mut input, 0.0, 0.1).changed {
                                if let Ok(v) = input.text.parse::<f32>() {
                                    sc.normal_bias = v.clamp(0.0, 0.1);
                                    *dirty = true;
                                }
                            }
                        }
                    });
                },
            );
        }

        // Main layout: hierarchy | viewport | inspector
        let hierarchy_ratio = 0.15;
        let inspector_ratio = 0.75;

        // Snapshot selected for the inspector (hierarchy may update it).
        let inspector_selected = self.selected;
        let camera_entity = self.camera_entity;
        let tree_state = &mut self.tree_state;
        let selected = &mut self.selected;
        let pending_edits = &mut self.pending_edits;

        ui.split_pane_h(hash("main_split"), hierarchy_ratio, |ui| {
            ui.padding(4.0, |ui| {
                ui.heading("Hierarchy");
                ui.spacing(4.0);
                hierarchy::draw_hierarchy(ui, ctx, tree_state, selected, camera_entity);
            });
        }, |ui| {
            ui.split_pane_h(hash("right_split"), inspector_ratio, |_ui| {
                // Center: viewport (3D rendered behind)
            }, |ui| {
                ui.padding(4.0, |ui| {
                    ui.heading("Inspector");
                    ui.spacing(4.0);
                    inspector::draw_inspector(ui, ctx, inspector_selected, pending_edits);
                });
            });
        });

        // Status bar
        let left = match &self.scene_path {
            Some(p) => {
                if self.dirty { format!("{p} *") } else { p.clone() }
            }
            None => {
                if self.dirty { "(untitled) *".to_string() } else { "(untitled)".to_string() }
            }
        };
        let gizmo_str = match self.gizmo_mode {
            GizmoMode::Translate => "Translate (W)",
            GizmoMode::Rotate => "Rotate (E)",
            GizmoMode::Scale => "Scale (R)",
        };
        let right = match self.selected {
            Some(e) => format!("Entity {} | {}", e.to_bits().get(), gizmo_str),
            None => gizmo_str.to_string(),
        };
        ui.status_bar(&left, &right);
    }

    fn should_exit(&self) -> bool {
        self.exit
    }
}

impl EditorApp {
    fn handle_menu_action(&mut self, action: u64, ctx: &mut Ctx) {
        match action {
            // New Scene
            1 => {
                if let Some(cam) = self.camera_entity {
                    esox_engine::scene::clear_scene(
                        ctx.world,
                        ctx.physics,
                        ctx.entity_map,
                        &[cam],
                    );
                }
                self.selected = None;
                self.scene_path = None;
                self.dirty = false;
                self.undo_stack.clear();
                self.toast_pending = Some((ToastKind::Info, "New scene created".into()));
            }
            // Open Scene
            2 => {
                let file = rfd::FileDialog::new()
                    .add_filter("Scene", &["scene.ron"])
                    .pick_file();
                if let Some(path) = file {
                    // Clear current scene
                    if let Some(cam) = self.camera_entity {
                        esox_engine::scene::clear_scene(
                            ctx.world,
                            ctx.physics,
                            ctx.entity_map,
                            &[cam],
                        );
                    }
                    match std::fs::read_to_string(&path) {
                        Ok(ron_str) => match esox_engine::scene::scene_from_ron(&ron_str) {
                            Ok(scene) => {
                                let _id_map = esox_engine::scene::load_scene(
                                    &scene,
                                    ctx.world,
                                    ctx.assets,
                                    Some(ctx.physics),
                                    Some(ctx.entity_map),
                                );
                                self.scene_path = Some(path.to_string_lossy().into_owned());
                                self.dirty = false;
                                self.selected = None;
                                self.undo_stack.clear();
                                self.toast_pending = Some((ToastKind::Success, format!("Opened: {}", path.display())));
                            }
                            Err(e) => {
                                self.toast_pending = Some((ToastKind::Error, format!("Parse error: {e}")));
                            }
                        },
                        Err(e) => {
                            self.toast_pending = Some((ToastKind::Error, format!("Read error: {e}")));
                        }
                    }
                }
            }
            // Save
            3 => {
                if let Some(ref path) = self.scene_path.clone() {
                    self.save_scene_to(ctx, path);
                } else {
                    self.save_scene_as(ctx);
                }
            }
            // Save As
            4 => {
                self.save_scene_as(ctx);
            }
            9 => self.exit = true,
            10 => {
                // Undo (menu)
                if let Some(result) = self.undo_stack.undo(ctx.world) {
                    if let Some(sel) = result.select {
                        self.selected = sel;
                    }
                    self.dirty = true;
                }
            }
            11 => {
                // Redo (menu)
                if let Some(result) = self.undo_stack.redo(ctx.world) {
                    if let Some(sel) = result.select {
                        self.selected = sel;
                    }
                    self.dirty = true;
                }
            }
            20 => {
                self.camera = EditorCamera::new();
            }
            21 => {
                if let Some(selected) = self.selected {
                    if let Ok(gt) = ctx.world.get::<&GlobalTransform>(selected) {
                        let pos = Vec3::new(gt.0.col(3).x, gt.0.col(3).y, gt.0.col(3).z);
                        self.camera.focus_on(pos);
                    }
                }
            }
            22 => {
                self.shortcuts_modal_open = true;
            }
            40 => {
                self.render_settings_open = !self.render_settings_open;
            }
            12 => {
                // Delete selected entity
                if let Some(selected) = self.selected.take() {
                    if self.camera_entity != Some(selected) {
                        let snapshot = EntitySnapshot::capture(ctx.world, selected);
                        let _ = ctx.world.despawn(selected);
                        self.undo_stack.push(UndoAction::DeleteEntity {
                            stale_entity: selected,
                            snapshot,
                        });
                        self.dirty = true;
                    }
                }
            }
            13 => {
                // Duplicate selected entity
                if let Some(selected) = self.selected {
                    if self.camera_entity != Some(selected) {
                        let snapshot = EntitySnapshot::capture(ctx.world, selected);
                        let new_entity = snapshot.respawn(ctx.world);
                        // Offset so the duplicate is visible
                        if let Ok(mut t) = ctx.world.get::<&mut Transform3D>(new_entity) {
                            t.position.x += 1.0;
                        }
                        let spawn_snapshot = EntitySnapshot::capture(ctx.world, new_entity);
                        self.undo_stack.push(UndoAction::SpawnEntity {
                            entity: new_entity,
                            snapshot: spawn_snapshot,
                        });
                        self.selected = Some(new_entity);
                        self.dirty = true;
                    }
                }
            }
            30 => {
                // Add empty entity
                let entity = ctx.world.spawn((
                    Transform3D::default(),
                    GlobalTransform::default(),
                    Tag("Empty".to_string()),
                ));
                let snapshot = EntitySnapshot::capture(ctx.world, entity);
                self.undo_stack.push(UndoAction::SpawnEntity { entity, snapshot });
                self.selected = Some(entity);
                self.dirty = true;
            }
            31 => {
                // Add cube
                let mesh = ctx.renderer.upload_mesh(ctx.gpu, &MeshData::cube(1.0));
                let mat = ctx.renderer.create_material(
                    ctx.gpu,
                    &MaterialDescriptor {
                        material_type: MaterialType::PBR,
                        albedo: [0.7, 0.7, 0.7, 1.0],
                        roughness: 0.5,
                        metallic: 0.0,
                        ..Default::default()
                    },
                );
                let entity = ctx.world.spawn((
                    Transform3D {
                        position: Vec3::new(0.0, 0.5, 0.0),
                        ..Default::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh,
                        material: mat,
                        tint: [1.0; 4],
                        visible: true,
                    },
                    Tag("Cube".to_string()),
                ));
                let snapshot = EntitySnapshot::capture(ctx.world, entity);
                self.undo_stack.push(UndoAction::SpawnEntity { entity, snapshot });
                self.selected = Some(entity);
                self.dirty = true;
            }
            32 => {
                // Add sphere
                let mesh = ctx
                    .renderer
                    .upload_mesh(ctx.gpu, &MeshData::sphere(0.5, 32, 16));
                let mat = ctx.renderer.create_material(
                    ctx.gpu,
                    &MaterialDescriptor {
                        material_type: MaterialType::PBR,
                        albedo: [0.7, 0.7, 0.7, 1.0],
                        roughness: 0.3,
                        metallic: 0.5,
                        ..Default::default()
                    },
                );
                let entity = ctx.world.spawn((
                    Transform3D {
                        position: Vec3::new(0.0, 0.5, 0.0),
                        ..Default::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh,
                        material: mat,
                        tint: [1.0; 4],
                        visible: true,
                    },
                    Tag("Sphere".to_string()),
                ));
                let snapshot = EntitySnapshot::capture(ctx.world, entity);
                self.undo_stack.push(UndoAction::SpawnEntity { entity, snapshot });
                self.selected = Some(entity);
                self.dirty = true;
            }
            33 => {
                // Add point light
                let entity = ctx.world.spawn((
                    Transform3D {
                        position: Vec3::new(0.0, 3.0, 0.0),
                        ..Default::default()
                    },
                    GlobalTransform::default(),
                    PointLightComponent {
                        color: [1.0, 0.9, 0.8],
                        intensity: 10.0,
                        range: 15.0,
                        cast_shadows: false,
                    },
                    Tag("Point Light".to_string()),
                ));
                let snapshot = EntitySnapshot::capture(ctx.world, entity);
                self.undo_stack.push(UndoAction::SpawnEntity { entity, snapshot });
                self.selected = Some(entity);
                self.dirty = true;
            }
            34 => {
                // Add spot light
                let entity = ctx.world.spawn((
                    Transform3D {
                        position: Vec3::new(0.0, 3.0, 0.0),
                        rotation: Quat::from_rotation_x(-FRAC_PI_4),
                        ..Default::default()
                    },
                    GlobalTransform::default(),
                    SpotLightComponent {
                        color: [1.0, 1.0, 1.0],
                        intensity: 20.0,
                        range: 20.0,
                        inner_cone_angle: 0.3,
                        outer_cone_angle: 0.5,
                        cast_shadows: false,
                    },
                    Tag("Spot Light".to_string()),
                ));
                let snapshot = EntitySnapshot::capture(ctx.world, entity);
                self.undo_stack.push(UndoAction::SpawnEntity { entity, snapshot });
                self.selected = Some(entity);
                self.dirty = true;
            }
            35 => {
                // Add directional light
                let entity = ctx.world.spawn((
                    Transform3D {
                        rotation: Quat::from_euler(
                            glam::EulerRot::XYZ,
                            -0.8,
                            0.5,
                            0.0,
                        ),
                        ..Default::default()
                    },
                    GlobalTransform::default(),
                    DirectionalLightComponent {
                        color: [1.0, 1.0, 1.0],
                        intensity: 2.0,
                    },
                    Tag("Dir Light".to_string()),
                ));
                let snapshot = EntitySnapshot::capture(ctx.world, entity);
                self.undo_stack.push(UndoAction::SpawnEntity { entity, snapshot });
                self.selected = Some(entity);
                self.dirty = true;
            }
            36 => {
                // Add camera
                let entity = ctx.world.spawn((
                    Transform3D {
                        position: Vec3::new(0.0, 2.0, 5.0),
                        ..Default::default()
                    },
                    GlobalTransform::default(),
                    Camera3D {
                        fov_y: FRAC_PI_4,
                        near: 0.1,
                        far: 500.0,
                        active: false,
                    },
                    Tag("Camera".to_string()),
                ));
                let snapshot = EntitySnapshot::capture(ctx.world, entity);
                self.undo_stack.push(UndoAction::SpawnEntity { entity, snapshot });
                self.selected = Some(entity);
                self.dirty = true;
            }
            _ => {}
        }
    }

    fn save_scene_to(&mut self, ctx: &Ctx, path: &str) {
        let scene = esox_engine::scene::save_scene(ctx.world, ctx.assets);
        match esox_engine::scene::scene_to_ron(&scene) {
            Ok(ron_str) => match std::fs::write(path, &ron_str) {
                Ok(()) => {
                    self.dirty = false;
                    self.toast_pending = Some((ToastKind::Success, format!("Saved: {path}")));
                }
                Err(e) => {
                    self.toast_pending = Some((ToastKind::Error, format!("Write error: {e}")));
                }
            },
            Err(e) => {
                self.toast_pending = Some((ToastKind::Error, format!("Serialize error: {e}")));
            }
        }
    }

    fn save_scene_as(&mut self, ctx: &Ctx) {
        let file = rfd::FileDialog::new()
            .add_filter("Scene", &["scene.ron"])
            .set_file_name("untitled.scene.ron")
            .save_file();
        if let Some(path) = file {
            let path_str = path.to_string_lossy().into_owned();
            self.save_scene_to(ctx, &path_str);
            self.scene_path = Some(path_str);
        }
    }
}

// ── Grid generator ──

fn generate_grid(half_size: i32, spacing: f32) -> MeshData {
    use esox_engine::esox_gfx::mesh3d::vertex::Vertex3D;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let color = [0.3, 0.3, 0.3, 0.5];

    for i in -half_size..=half_size {
        let offset = i as f32 * spacing;
        let extent = half_size as f32 * spacing;
        let idx = vertices.len() as u32;

        // Line along Z
        vertices.push(Vertex3D {
            position: [offset, 0.0, -extent],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            color,
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        vertices.push(Vertex3D {
            position: [offset, 0.0, extent],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
            color,
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        // Degenerate triangles to render as thin quads
        vertices.push(Vertex3D {
            position: [offset + 0.01, 0.0, extent],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 1.0],
            color,
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        vertices.push(Vertex3D {
            position: [offset + 0.01, 0.0, -extent],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 1.0],
            color,
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        indices.extend_from_slice(&[idx, idx + 1, idx + 2, idx, idx + 2, idx + 3]);

        let idx = vertices.len() as u32;
        // Line along X
        vertices.push(Vertex3D {
            position: [-extent, 0.0, offset],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            color,
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        vertices.push(Vertex3D {
            position: [extent, 0.0, offset],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
            color,
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        vertices.push(Vertex3D {
            position: [extent, 0.0, offset + 0.01],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 1.0],
            color,
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        vertices.push(Vertex3D {
            position: [-extent, 0.0, offset + 0.01],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 1.0],
            color,
            tangent: [1.0, 0.0, 0.0, 1.0],
        });
        indices.extend_from_slice(&[idx, idx + 1, idx + 2, idx, idx + 2, idx + 3]);
    }

    MeshData { vertices, indices }
}

// ── Wireframe helper ──

/// Generate 12 thin-bar transforms that outline an AABB.
fn wireframe_aabb_edges(min: Vec3, max: Vec3) -> Vec<esox_engine::esox_gfx::mesh3d::Transform> {
    let size = max - min;
    let center = (min + max) * 0.5;
    let thickness = 0.02;
    let mut edges = Vec::with_capacity(12);

    // 4 edges along X (at the 4 Y/Z corner combinations)
    for &y in &[min.y, max.y] {
        for &z in &[min.z, max.z] {
            edges.push(esox_engine::esox_gfx::mesh3d::Transform {
                position: Vec3::new(center.x, y, z),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(size.x, thickness, thickness),
            });
        }
    }
    // 4 edges along Y (at the 4 X/Z corner combinations)
    for &x in &[min.x, max.x] {
        for &z in &[min.z, max.z] {
            edges.push(esox_engine::esox_gfx::mesh3d::Transform {
                position: Vec3::new(x, center.y, z),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(thickness, size.y, thickness),
            });
        }
    }
    // 4 edges along Z (at the 4 X/Y corner combinations)
    for &x in &[min.x, max.x] {
        for &y in &[min.y, max.y] {
            edges.push(esox_engine::esox_gfx::mesh3d::Transform {
                position: Vec3::new(x, y, center.z),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(thickness, thickness, size.z),
            });
        }
    }

    edges
}

// ── Utilities ──

/// Simple hash for UI widget IDs.
fn hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,editor=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_engine::esox_platform::config::WindowConfig {
                title: "esox Editor".into(),
                width: Some(1600),
                height: Some(900),
                ..Default::default()
            },
            msaa: 4,
            ..Default::default()
        },
        ..EngineConfig::default()
    };

    let game = EditorApp::new();
    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("editor error: {e}");
        std::process::exit(1);
    }
}
