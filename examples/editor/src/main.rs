use std::f32::consts::{FRAC_PI_4, PI};

use esox_engine::esox_gfx::mesh3d::{
    InstanceData, MaterialDescriptor, MaterialHandle, MaterialType, MeshData, MeshHandle,
};
use esox_engine::glam::{self, Mat4, Quat, Vec3};
use esox_engine::hecs;
use esox_engine::winit::keyboard::KeyCode;
use esox_engine::{
    ActionBinding, Camera3D, Ctx, DirectionalLightComponent, EngineConfig, Game, GlobalTransform,
    MeshRenderer, Tag, Transform3D,
};

mod hierarchy;
mod inspector;
mod picking;

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
    SetSpotLightIntensity(hecs::Entity, f32),
    SetSpotLightRange(hecs::Entity, f32),
    SetDirLightIntensity(hecs::Entity, f32),
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
    // Grid
    grid_mesh: Option<MeshHandle>,
    grid_mat: Option<MaterialHandle>,
    // Scene tracking (used by Save/Load, Steps 6-7)
    #[allow(dead_code)]
    scene_path: Option<String>,
    #[allow(dead_code)]
    dirty: bool,
    // Pending mutations from UI
    pending_edits: Vec<PendingEdit>,
    // Pending menu actions
    pending_menu_action: Option<u64>,
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
            grid_mesh: None,
            grid_mat: None,
            scene_path: None,
            dirty: false,
            pending_edits: Vec::new(),
            pending_menu_action: None,
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
        ctx.renderer.enable_shadows(ctx.gpu);
        ctx.renderer.enable_point_shadows(ctx.gpu);
        ctx.renderer.enable_spot_shadows(ctx.gpu);
        ctx.renderer.generate_procedural_ibl(ctx.gpu);

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
        self.gizmo_meshes = Some(GizmoMeshes {
            arrow_mesh,
            cube_mesh,
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
        for edit in self.pending_edits.drain(..) {
            match edit {
                PendingEdit::SetTransform(entity, t) => {
                    if let Ok(mut tr) = ctx.world.get::<&mut Transform3D>(entity) {
                        *tr = t;
                    }
                }
                PendingEdit::SetPointLightIntensity(entity, v) => {
                    if let Ok(mut pl) = ctx.world.get::<&mut esox_engine::PointLightComponent>(entity) {
                        pl.intensity = v;
                    }
                }
                PendingEdit::SetPointLightRange(entity, v) => {
                    if let Ok(mut pl) = ctx.world.get::<&mut esox_engine::PointLightComponent>(entity) {
                        pl.range = v;
                    }
                }
                PendingEdit::SetSpotLightIntensity(entity, v) => {
                    if let Ok(mut sl) = ctx.world.get::<&mut esox_engine::SpotLightComponent>(entity) {
                        sl.intensity = v;
                    }
                }
                PendingEdit::SetSpotLightRange(entity, v) => {
                    if let Ok(mut sl) = ctx.world.get::<&mut esox_engine::SpotLightComponent>(entity) {
                        sl.range = v;
                    }
                }
                PendingEdit::SetDirLightIntensity(entity, v) => {
                    if let Ok(mut dl) = ctx.world.get::<&mut DirectionalLightComponent>(entity) {
                        dl.intensity = v;
                    }
                }
            }
        }

        // Handle pending menu actions
        if let Some(action) = self.pending_menu_action.take() {
            self.handle_menu_action(action, ctx);
        }

        if ctx.input.just_pressed("exit") {
            self.exit = true;
        }

        // Left-click picking (only when no MMB/RMB held)
        if ctx.input.just_pressed("pick")
            && !ctx.input.is_mouse_button_down(1)
            && !ctx.input.is_mouse_button_down(2)
        {
            let (mx, my) = ctx.input.mouse_pos();

            // Build view and projection matrices from camera entity
            let cam_pos = self.camera.position();
            let cam_target = match self.camera.mode {
                CameraMode::Orbit => self.camera.orbit_target,
                CameraMode::Fly => cam_pos + self.camera.forward(),
            };

            let view = Mat4::look_at_rh(cam_pos, cam_target, Vec3::Y);
            let aspect = ctx.viewport.0 as f32 / ctx.viewport.1.max(1) as f32;
            let projection = Mat4::perspective_rh(FRAC_PI_4, aspect, 0.1, 500.0);

            let (ray_origin, ray_dir) =
                picking::screen_to_ray(mx, my, ctx.viewport, view, projection);

            if let Some((entity, _dist)) =
                picking::pick_entity(ctx, ray_origin, ray_dir, self.camera_entity)
            {
                self.selected = Some(entity);
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
                    let _ = ctx.world.despawn(selected);
                }
            }
        }

        self.camera.update(ctx);
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

        // Draw selection highlight (wireframe cube around selected entity)
        if let Some(selected) = self.selected {
            if let (Some(gizmo_mats), Some(gizmo_meshes)) =
                (&self.gizmo_mats, &self.gizmo_meshes)
            {
                // Draw translate gizmo at selected entity's position
                if let Ok(gt) = ctx.world.get::<&GlobalTransform>(selected) {
                    let world_pos =
                        Vec3::new(gt.0.col(3).x, gt.0.col(3).y, gt.0.col(3).z);
                    let cam_dist = (world_pos - self.camera.position()).length();
                    let gizmo_scale = cam_dist / 8.0;

                    // X axis (red)
                    let x_transform = esox_engine::esox_gfx::mesh3d::Transform {
                        position: world_pos + Vec3::X * gizmo_scale * 0.5,
                        rotation: Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2),
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
                        rotation: Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                        scale: Vec3::splat(gizmo_scale),
                    };
                    ctx.renderer.draw_with_material(
                        gizmo_meshes.arrow_mesh,
                        gizmo_mats.blue,
                        &[InstanceData::from_transform(&z_transform)],
                    );
                }
            }
        }
    }

    fn ui(&mut self, ui: &mut esox_engine::esox_ui::Ui, ctx: &Ctx) {
        use esox_engine::esox_ui::{Menu, MenuEntry, MenuItem};

        // Menu bar
        let menus = &[
            Menu::new(
                "File",
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
                    MenuEntry::Item(
                        MenuItem::new("Undo", 10)
                            .with_shortcut("Ctrl+Z")
                            .disabled(),
                    ),
                    MenuEntry::Item(
                        MenuItem::new("Redo", 11)
                            .with_shortcut("Ctrl+Shift+Z")
                            .disabled(),
                    ),
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
                ],
            ),
            Menu::new(
                "Entity",
                vec![
                    MenuEntry::Item(MenuItem::new("Add Empty", 30)),
                    MenuEntry::Item(MenuItem::new("Add Cube", 31)),
                    MenuEntry::Item(MenuItem::new("Add Sphere", 32)),
                    MenuEntry::Item(MenuItem::new("Add Point Light", 33)),
                    MenuEntry::Item(MenuItem::new("Add Spot Light", 34)),
                ],
            ),
        ];

        if let Some(action) = ui.menu_bar(menus) {
            self.pending_menu_action = Some(action);
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
    }

    fn should_exit(&self) -> bool {
        self.exit
    }
}

impl EditorApp {
    fn handle_menu_action(&mut self, action: u64, ctx: &mut Ctx) {
        match action {
            9 => self.exit = true,
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
            12 => {
                // Delete selected entity
                if let Some(selected) = self.selected.take() {
                    let _ = ctx.world.despawn(selected);
                }
            }
            30 => {
                // Add empty entity
                let entity = ctx.world.spawn((
                    Transform3D::default(),
                    GlobalTransform::default(),
                    Tag("Empty".to_string()),
                ));
                self.selected = Some(entity);
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
                self.selected = Some(entity);
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
                self.selected = Some(entity);
            }
            33 => {
                // Add point light
                let entity = ctx.world.spawn((
                    Transform3D {
                        position: Vec3::new(0.0, 3.0, 0.0),
                        ..Default::default()
                    },
                    GlobalTransform::default(),
                    esox_engine::PointLightComponent {
                        color: [1.0, 0.9, 0.8],
                        intensity: 10.0,
                        range: 15.0,
                        cast_shadows: false,
                    },
                    Tag("Point Light".to_string()),
                ));
                self.selected = Some(entity);
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
                    esox_engine::SpotLightComponent {
                        color: [1.0, 1.0, 1.0],
                        intensity: 20.0,
                        range: 20.0,
                        inner_cone_angle: 0.3,
                        outer_cone_angle: 0.5,
                        cast_shadows: false,
                    },
                    Tag("Spot Light".to_string()),
                ));
                self.selected = Some(entity);
            }
            _ => {}
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
