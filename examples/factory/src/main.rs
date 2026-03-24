//! Factory game — 3D Factorio-style factory builder on the ESOX engine.
//!
//! Phase 2+3: inventory, belts, inserters, machines, mining, power, and fluids.
//!
//! Controls:
//!   WASD / Arrow keys — Pan camera
//!   Q/E — Rotate camera
//!   Scroll — Zoom in/out
//!   1-9,0 — Select building (1=Belt, 2=Inserter, 3=Smelter, 4=Assembler,
//!            5=Miner, 6=Steam Engine, 7=Power Pole, 8=Pipe, 9=Refinery,
//!            0=Underground Belt)
//!   R — Rotate placement
//!   LMB — Place building
//!   Escape — Cancel placement / Exit

mod belt;
mod fluid;
mod hud;
mod inserter;
mod inventory;
mod mining;
mod power;
mod recipe;

use esox_engine::*;
use esox_engine::esox_input::KeyCode;
use esox_engine::glam::{Quat, Vec3};
use esox_gfx::mesh3d::{CameraMode, MaterialDescriptor, MaterialType, MeshData, MeshHandle, MaterialHandle};

use belt::{BeltSegment, Dir4, GridPos, UndergroundBelt, UndergroundBeltMode, MAX_UNDERGROUND_DISTANCE, SLOTS_PER_BELT};
use fluid::{FluidIO, FluidSource, FluidType, Pipe};
use inserter::Inserter;
use inventory::{Inventory, ItemRegistry};
use mining::{Miner, ResourceNode};
use power::{PowerConsumer, PowerPole, PowerSource};
use recipe::{Machine, MachineType, OutputInventory, RecipeRegistry};

/// Grid cell size in world units.
const CELL_SIZE: f32 = 1.0;

/// What the player wants to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuildTool {
    Belt,
    Inserter,
    Smelter,
    Assembler,
    Miner,
    SteamEngine,
    PowerPole,
    Pipe,
    Refinery,
    UndergroundBelt,
}

struct FactoryGame {
    // Registries
    items: Option<ItemRegistry>,
    recipes: Option<RecipeRegistry>,

    // Camera state
    cam_focus: Vec3,
    cam_angle: f32,
    cam_distance: f32,
    cam_pitch: f32,

    // Build mode
    build_tool: Option<BuildTool>,
    build_direction: Dir4,
    cursor_grid: GridPos,

    // Meshes & materials (loaded in init)
    cube_mesh: Option<MeshHandle>,
    plane_mesh: Option<MeshHandle>,
    belt_mat: Option<MaterialHandle>,
    inserter_mat: Option<MaterialHandle>,
    smelter_mat: Option<MaterialHandle>,
    assembler_mat: Option<MaterialHandle>,
    miner_mat: Option<MaterialHandle>,
    ore_node_mat: Option<MaterialHandle>,
    steam_engine_mat: Option<MaterialHandle>,
    power_pole_mat: Option<MaterialHandle>,
    pipe_mat: Option<MaterialHandle>,
    refinery_mat: Option<MaterialHandle>,
    underground_belt_mat: Option<MaterialHandle>,
    oil_well_mat: Option<MaterialHandle>,
    ghost_mat: Option<MaterialHandle>,
    ground_mat: Option<MaterialHandle>,
    item_mat: Option<MaterialHandle>,

    // Ground plane
    ground: Option<esox_engine::ground_plane::GroundPlane>,
    overlay_renderer: esox_engine::ground_overlay::GroundOverlayRenderer,

    // Stats (for UI)
    tick_count: u64,

    exit: bool,
}

impl Default for FactoryGame {
    fn default() -> Self {
        Self {
            items: None,
            recipes: None,
            cam_focus: Vec3::new(8.0, 0.0, 8.0),
            cam_angle: std::f32::consts::FRAC_PI_4,
            cam_distance: 20.0,
            cam_pitch: 1.0, // ~57 degrees
            build_tool: None,
            build_direction: Dir4::East,
            cursor_grid: GridPos::new(0, 0),
            cube_mesh: None,
            plane_mesh: None,
            belt_mat: None,
            inserter_mat: None,
            smelter_mat: None,
            assembler_mat: None,
            miner_mat: None,
            ore_node_mat: None,
            steam_engine_mat: None,
            power_pole_mat: None,
            pipe_mat: None,
            refinery_mat: None,
            underground_belt_mat: None,
            oil_well_mat: None,
            ghost_mat: None,
            ground_mat: None,
            item_mat: None,
            ground: None,
            overlay_renderer: esox_engine::ground_overlay::GroundOverlayRenderer::new(),
            tick_count: 0,
            exit: false,
        }
    }
}

impl FactoryGame {
    fn items(&self) -> &ItemRegistry {
        self.items.as_ref().unwrap()
    }

    fn recipes(&self) -> &RecipeRegistry {
        self.recipes.as_ref().unwrap()
    }
}

/// Compute a quaternion that looks from `eye` toward `target` (Y-up).
fn look_at_quat(eye: Vec3, target: Vec3) -> Quat {
    let forward = (target - eye).normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward))
}

impl Game for FactoryGame {
    fn init(&mut self, ctx: &mut Ctx) {
        // ── Load data ──
        let items_data = include_str!("../assets/items.ron");
        let recipes_data = include_str!("../assets/recipes.ron");
        let items = ItemRegistry::load_from_ron(items_data);
        let recipes = RecipeRegistry::load_from_ron(recipes_data, &items);
        self.items = Some(items);
        self.recipes = Some(recipes);

        // ── Input bindings ──
        ctx.input.bind_action("exit", ActionBinding::Key(KeyCode::Escape));
        ctx.input.bind_action("place", ActionBinding::MouseButton(0));
        ctx.input.bind_action("rotate", ActionBinding::Key(KeyCode::KeyR));
        ctx.input.bind_action("tool_belt", ActionBinding::Key(KeyCode::Digit1));
        ctx.input.bind_action("tool_inserter", ActionBinding::Key(KeyCode::Digit2));
        ctx.input.bind_action("tool_smelter", ActionBinding::Key(KeyCode::Digit3));
        ctx.input.bind_action("tool_assembler", ActionBinding::Key(KeyCode::Digit4));
        ctx.input.bind_action("tool_miner", ActionBinding::Key(KeyCode::Digit5));
        ctx.input.bind_action("tool_steam_engine", ActionBinding::Key(KeyCode::Digit6));
        ctx.input.bind_action("tool_power_pole", ActionBinding::Key(KeyCode::Digit7));
        ctx.input.bind_action("tool_pipe", ActionBinding::Key(KeyCode::Digit8));
        ctx.input.bind_action("tool_refinery", ActionBinding::Key(KeyCode::Digit9));
        ctx.input.bind_action("tool_underground_belt", ActionBinding::Key(KeyCode::Digit0));

        ctx.input.bind_axis("pan_x", AxisBinding::Keys {
            negative: KeyCode::KeyA,
            positive: KeyCode::KeyD,
        });
        ctx.input.bind_axis("pan_z", AxisBinding::Keys {
            negative: KeyCode::KeyW,
            positive: KeyCode::KeyS,
        });
        ctx.input.bind_axis("cam_rotate", AxisBinding::Keys {
            negative: KeyCode::KeyQ,
            positive: KeyCode::KeyE,
        });
        // Zoom handled directly via scroll_delta() below.

        // ── Meshes ──
        let cube = MeshData::cube(0.9);
        let cube_mesh = ctx.renderer.upload_mesh(ctx.gpu, &cube);
        ctx.assets.register_mesh(cube_mesh);
        self.cube_mesh = Some(cube_mesh);

        let plane = MeshData::plane(CELL_SIZE, CELL_SIZE, 1);
        let plane_mesh = ctx.renderer.upload_mesh(ctx.gpu, &plane);
        ctx.assets.register_mesh(plane_mesh);
        self.plane_mesh = Some(plane_mesh);

        // ── Materials ──
        let make_mat = |ctx_renderer: &mut esox_gfx::mesh3d::Renderer3D,
                        gpu: &esox_gfx::GpuContext,
                        color: [f32; 4],
                        roughness: f32,
                        metallic: f32| {
            ctx_renderer.create_material(gpu, &MaterialDescriptor {
                material_type: MaterialType::PBR,
                albedo: color,
                roughness,
                metallic,
                ..MaterialDescriptor::default()
            })
        };

        self.belt_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.6, 0.6, 0.55, 1.0], 0.8, 0.1));
        self.inserter_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.9, 0.7, 0.2, 1.0], 0.4, 0.6));
        self.smelter_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.8, 0.3, 0.15, 1.0], 0.5, 0.5));
        self.assembler_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.2, 0.5, 0.8, 1.0], 0.4, 0.5));
        self.miner_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.7, 0.5, 0.2, 1.0], 0.6, 0.4));
        self.ore_node_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.4, 0.25, 0.15, 1.0], 0.9, 0.0));
        self.steam_engine_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.5, 0.2, 0.15, 1.0], 0.5, 0.6));
        self.power_pole_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.55, 0.35, 0.15, 1.0], 0.8, 0.1));
        self.pipe_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.6, 0.6, 0.65, 1.0], 0.3, 0.8));
        self.refinery_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.45, 0.42, 0.4, 1.0], 0.4, 0.7));
        self.underground_belt_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.4, 0.4, 0.35, 1.0], 0.8, 0.15));
        self.oil_well_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.15, 0.12, 0.1, 1.0], 0.9, 0.2));
        self.ghost_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.5, 1.0, 0.5, 0.4], 0.5, 0.0));
        self.ground_mat = Some(make_mat(ctx.renderer, ctx.gpu, [0.28, 0.32, 0.22, 1.0], 0.95, 0.0));
        self.item_mat = Some(make_mat(ctx.renderer, ctx.gpu, [1.0, 0.9, 0.3, 1.0], 0.3, 0.7));

        // ── Ground plane ──
        self.ground = Some(esox_engine::ground_plane::GroundPlane::new(
            ctx.gpu,
            ctx.renderer,
            esox_engine::ground_plane::GroundPlaneConfig {
                tile_size: CELL_SIZE * 4.0,
                render_radius: 12,
                material: self.ground_mat.unwrap(),
            },
        ));

        self.overlay_renderer.ensure_mesh(ctx.gpu, ctx.renderer);

        // ── Lighting ──
        ctx.world.spawn((
            Transform3D {
                position: Vec3::ZERO,
                rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_3)
                    * Quat::from_rotation_y(0.3),
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
                position: Vec3::new(8.0, 8.0, 8.0),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            PointLightComponent {
                color: [0.6, 0.8, 1.0],
                intensity: 15.0,
                range: 30.0,
                cast_shadows: false,
            },
        ));

        // ── Camera (orthographic, isometric angle) ──
        let cam_pos = self.camera_position();
        ctx.world.spawn((
            Transform3D {
                position: cam_pos,
                rotation: look_at_quat(cam_pos, self.cam_focus),
                ..Transform3D::default()
            },
            GlobalTransform::default(),
            Camera3D {
                fov_y: std::f32::consts::FRAC_PI_4,
                near: 0.1,
                far: 500.0,
                active: true,
                mode: CameraMode::Orthographic { ortho_size: self.cam_distance },
            },
        ));

        // ── Demo scene: spawn some ore patches ──
        let items = self.items();
        let iron_ore = items.id_of("iron-ore").unwrap();
        let copper_ore = items.id_of("copper-ore").unwrap();

        // Iron ore patch.
        for row in 0..3 {
            for col in 0..3 {
                let pos = GridPos::new(2 + col, 2 + row);
                let world_pos = pos.to_world(CELL_SIZE);
                ctx.world.spawn((
                    ResourceNode::new(iron_ore, 5000, pos),
                    Transform3D {
                        position: world_pos,
                        scale: Vec3::new(0.9, 0.3, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: cube_mesh,
                        material: self.ore_node_mat.unwrap(),
                        tint: [0.6, 0.4, 0.3, 1.0],
                        visible: true,
                    },
                ));
            }
        }

        // Copper ore patch.
        for row in 0..3 {
            for col in 0..3 {
                let pos = GridPos::new(12 + col, 2 + row);
                let world_pos = pos.to_world(CELL_SIZE);
                ctx.world.spawn((
                    ResourceNode::new(copper_ore, 5000, pos),
                    Transform3D {
                        position: world_pos,
                        scale: Vec3::new(0.9, 0.3, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: cube_mesh,
                        material: self.ore_node_mat.unwrap(),
                        tint: [0.3, 0.5, 0.4, 1.0],
                        visible: true,
                    },
                ));
            }
        }

        // Crude oil wells.
        for row in 0..2 {
            for col in 0..2 {
                let pos = GridPos::new(7 + col, 7 + row);
                let world_pos = pos.to_world(CELL_SIZE);
                ctx.world.spawn((
                    FluidSource::new(FluidType::CrudeOil, fluid::CRUDE_OIL_RATE),
                    pos,
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.1, world_pos.z),
                        scale: Vec3::new(0.9, 0.2, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: cube_mesh,
                        material: self.oil_well_mat.unwrap(),
                        tint: [0.15, 0.12, 0.1, 1.0],
                        visible: true,
                    },
                ));
            }
        }
    }

    fn update(&mut self, ctx: &mut Ctx) {
        if ctx.input.just_pressed("exit") {
            if self.build_tool.is_some() {
                self.build_tool = None;
            } else {
                self.exit = true;
            }
            return;
        }

        let dt = ctx.time.tick_dt;

        // ── Camera panning ──
        let pan_x = ctx.input.axis("pan_x");
        let pan_z = ctx.input.axis("pan_z");
        let cam_rot = ctx.input.axis("cam_rotate");
        let scroll = ctx.input.scroll_delta();

        let forward = Vec3::new(-self.cam_angle.sin(), 0.0, -self.cam_angle.cos());
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let pan_speed = 15.0;
        self.cam_focus += (right * pan_x + forward * pan_z) * pan_speed * dt;
        self.cam_angle += cam_rot * 2.0 * dt;
        self.cam_distance = (self.cam_distance - scroll * 2.0).clamp(5.0, 60.0);

        // Update camera entity.
        let cam_pos = self.camera_position();
        for (_e, (t, cam)) in ctx.world.query_mut::<(&mut Transform3D, &mut Camera3D)>() {
            if cam.active {
                t.position = cam_pos;
                t.rotation = look_at_quat(cam_pos, self.cam_focus);
                cam.mode = CameraMode::Orthographic { ortho_size: self.cam_distance };
            }
        }

        // ── Tool selection ──
        if ctx.input.just_pressed("tool_belt") {
            self.build_tool = Some(BuildTool::Belt);
        }
        if ctx.input.just_pressed("tool_inserter") {
            self.build_tool = Some(BuildTool::Inserter);
        }
        if ctx.input.just_pressed("tool_smelter") {
            self.build_tool = Some(BuildTool::Smelter);
        }
        if ctx.input.just_pressed("tool_assembler") {
            self.build_tool = Some(BuildTool::Assembler);
        }
        if ctx.input.just_pressed("tool_miner") {
            self.build_tool = Some(BuildTool::Miner);
        }
        if ctx.input.just_pressed("tool_steam_engine") {
            self.build_tool = Some(BuildTool::SteamEngine);
        }
        if ctx.input.just_pressed("tool_power_pole") {
            self.build_tool = Some(BuildTool::PowerPole);
        }
        if ctx.input.just_pressed("tool_pipe") {
            self.build_tool = Some(BuildTool::Pipe);
        }
        if ctx.input.just_pressed("tool_refinery") {
            self.build_tool = Some(BuildTool::Refinery);
        }
        if ctx.input.just_pressed("tool_underground_belt") {
            self.build_tool = Some(BuildTool::UndergroundBelt);
        }

        // ── Rotation ──
        if ctx.input.just_pressed("rotate") {
            self.build_direction = self.build_direction.rotate_cw();
        }

        // ── Cursor position from mouse ray ──
        {
            let (mx, my) = ctx.input.mouse_pos();
            let (view, proj) = self.get_camera_matrices(ctx);
            let (ray_o, ray_d) = picking::screen_to_ray(mx, my, ctx.viewport, view, proj);
            if let Some(hit) = picking::ray_ground_plane(ray_o, ray_d) {
                self.cursor_grid = GridPos::from_world(hit, CELL_SIZE);
            }
        }

        // ── Placement ──
        if ctx.input.just_pressed("place") {
            if let Some(tool) = self.build_tool {
                self.place_building(tool, ctx);
            }
        }

        // ── Game systems ──
        power::power_tick_system(ctx.world);
        fluid::fluid_tick_system(ctx.world);
        belt::belt_tick_system(ctx.world);
        inserter::inserter_tick_system(ctx.world, self.items());
        recipe::machine_tick_system(ctx.world, self.items(), self.recipes());
        mining::mining_tick_system(ctx.world, self.items());

        self.tick_count = ctx.time.total_ticks;
    }

    fn render(&mut self, ctx: &mut Ctx, _alpha: f32) {
        // Draw ground plane.
        if let Some(ground) = &self.ground {
            ground.draw(ctx.renderer, self.cam_focus);
        }

        // Draw placement ghost.
        if let Some(tool) = self.build_tool {
            let world_pos = self.cursor_grid.to_world(CELL_SIZE);
            let (scale, y_offset) = match tool {
                BuildTool::Belt => (Vec3::new(0.9, 0.15, 0.9), 0.075),
                BuildTool::Inserter => (Vec3::new(0.3, 0.6, 0.3), 0.3),
                BuildTool::Smelter => (Vec3::new(0.9, 0.8, 0.9), 0.4),
                BuildTool::Assembler => (Vec3::new(0.9, 0.7, 0.9), 0.35),
                BuildTool::Miner => (Vec3::new(0.85, 0.5, 0.85), 0.25),
                BuildTool::SteamEngine => (Vec3::new(0.9, 0.9, 0.9), 0.45),
                BuildTool::PowerPole => (Vec3::new(0.2, 1.4, 0.2), 0.7),
                BuildTool::Pipe => (Vec3::new(0.5, 0.3, 0.5), 0.15),
                BuildTool::Refinery => (Vec3::new(0.9, 1.0, 0.9), 0.5),
                BuildTool::UndergroundBelt => (Vec3::new(0.9, 0.12, 0.9), 0.06),
            };
            let pos = Vec3::new(world_pos.x, y_offset, world_pos.z);
            let rot = Quat::from_rotation_y(self.build_direction.angle_y());
            let instance = esox_gfx::mesh3d::InstanceData::with_color(
                &esox_gfx::mesh3d::Transform {
                    position: pos,
                    rotation: rot,
                    scale,
                },
                [0.3, 1.0, 0.3, 0.4],
            );
            ctx.renderer.draw_with_material(
                self.cube_mesh.unwrap(),
                self.ghost_mat.unwrap(),
                &[instance],
            );

            // For inserter, also show pickup/dropoff indicators.
            if tool == BuildTool::Inserter {
                let pickup = self.cursor_grid.neighbor(self.build_direction.opposite());
                let dropoff = self.cursor_grid.neighbor(self.build_direction);
                self.draw_grid_highlight(ctx, pickup, [0.2, 0.8, 0.2, 0.3]);
                self.draw_grid_highlight(ctx, dropoff, [0.8, 0.2, 0.2, 0.3]);
            }
        }

        // Draw items on belts.
        self.draw_belt_items(ctx);

        // Draw overlays.
        self.overlay_renderer.draw(ctx.renderer);
    }

    fn ui(&mut self, ui: &mut esox_ui::Ui, ctx: &Ctx) {
        hud::draw_hud(
            ui,
            ctx.world,
            self.items.as_ref().unwrap(),
            self.recipes.as_ref().unwrap(),
            self.build_tool,
            self.build_direction,
            self.cursor_grid,
            self.tick_count,
            ctx.viewport,
        );
    }

    fn should_exit(&self) -> bool {
        self.exit
    }
}

impl FactoryGame {
    fn camera_position(&self) -> Vec3 {
        let horizontal = self.cam_distance * self.cam_pitch.cos();
        let vertical = self.cam_distance * self.cam_pitch.sin();
        self.cam_focus
            + Vec3::new(
                self.cam_angle.cos() * horizontal,
                vertical,
                self.cam_angle.sin() * horizontal,
            )
    }

    fn get_camera_matrices(&self, ctx: &Ctx) -> (glam::Mat4, glam::Mat4) {
        for (_e, (t, cam)) in ctx.world.query::<(&Transform3D, &Camera3D)>().iter() {
            if cam.active {
                let view = t.matrix().inverse();
                let aspect = ctx.viewport.0 as f32 / ctx.viewport.1 as f32;
                let proj = match cam.mode {
                    CameraMode::Orthographic { ortho_size } => {
                        glam::Mat4::orthographic_rh(
                            -ortho_size * aspect,
                            ortho_size * aspect,
                            -ortho_size,
                            ortho_size,
                            cam.near,
                            cam.far,
                        )
                    }
                    CameraMode::Perspective => {
                        glam::Mat4::perspective_rh(cam.fov_y, aspect, cam.near, cam.far)
                    }
                };
                return (view, proj);
            }
        }
        (glam::Mat4::IDENTITY, glam::Mat4::IDENTITY)
    }

    fn place_building(&mut self, tool: BuildTool, ctx: &mut Ctx) {
        let pos = self.cursor_grid;
        let dir = self.build_direction;
        let world_pos = pos.to_world(CELL_SIZE);

        // Check if something already exists at this grid position.
        let occupied = self.is_grid_occupied(ctx, pos);
        if occupied {
            return;
        }

        match tool {
            BuildTool::Belt => {
                ctx.world.spawn((
                    BeltSegment::new(pos, dir),
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.075, world_pos.z),
                        rotation: Quat::from_rotation_y(dir.angle_y()),
                        scale: Vec3::new(0.9, 0.15, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.belt_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                ));
            }
            BuildTool::Inserter => {
                let pickup = pos.neighbor(dir.opposite());
                let dropoff = pos.neighbor(dir);
                ctx.world.spawn((
                    Inserter::new(pickup, dropoff),
                    PowerConsumer::new(power::INSERTER_WATTS),
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.3, world_pos.z),
                        rotation: Quat::from_rotation_y(dir.angle_y()),
                        scale: Vec3::new(0.3, 0.6, 0.3),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.inserter_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                    pos,
                ));
            }
            BuildTool::Smelter => {
                let recipe = self.recipes().id_of("smelt-iron");
                ctx.world.spawn((
                    Machine::with_recipe(MachineType::Smelter, recipe.unwrap_or(0)),
                    Inventory::new(4),
                    OutputInventory(Inventory::new(4)),
                    PowerConsumer::new(power::SMELTER_WATTS),
                    pos,
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.4, world_pos.z),
                        rotation: Quat::from_rotation_y(dir.angle_y()),
                        scale: Vec3::new(0.9, 0.8, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.smelter_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                ));
            }
            BuildTool::Assembler => {
                let recipe = self.recipes().id_of("craft-gear");
                ctx.world.spawn((
                    Machine::with_recipe(MachineType::Assembler, recipe.unwrap_or(0)),
                    Inventory::new(8),
                    OutputInventory(Inventory::new(4)),
                    PowerConsumer::new(power::ASSEMBLER_WATTS),
                    pos,
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.35, world_pos.z),
                        rotation: Quat::from_rotation_y(dir.angle_y()),
                        scale: Vec3::new(0.9, 0.7, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.assembler_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                ));
            }
            BuildTool::Miner => {
                ctx.world.spawn((
                    Miner::new(pos),
                    OutputInventory(Inventory::new(4)),
                    PowerConsumer::new(power::MINER_WATTS),
                    pos,
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.25, world_pos.z),
                        rotation: Quat::from_rotation_y(dir.angle_y()),
                        scale: Vec3::new(0.85, 0.5, 0.85),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.miner_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                ));
            }
            BuildTool::SteamEngine => {
                ctx.world.spawn((
                    PowerSource::new(power::STEAM_ENGINE_WATTS),
                    pos,
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.45, world_pos.z),
                        rotation: Quat::from_rotation_y(dir.angle_y()),
                        scale: Vec3::new(0.9, 0.9, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.steam_engine_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                ));
            }
            BuildTool::PowerPole => {
                ctx.world.spawn((
                    PowerPole::new(power::POLE_REACH),
                    pos,
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.7, world_pos.z),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::new(0.2, 1.4, 0.2),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.power_pole_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                ));
            }
            BuildTool::Pipe => {
                ctx.world.spawn((
                    Pipe::new(pos),
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.15, world_pos.z),
                        rotation: Quat::IDENTITY,
                        scale: Vec3::new(0.5, 0.3, 0.5),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.pipe_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                ));
            }
            BuildTool::Refinery => {
                let (recipe_id, fio) = {
                    let recipes = self.recipes();
                    let rid = recipes.id_of("refine-petroleum").unwrap();
                    let recipe = recipes.get(rid);
                    (rid, FluidIO::from_recipe(&recipe.fluid_inputs, &recipe.fluid_outputs))
                };
                ctx.world.spawn((
                    Machine::with_recipe(MachineType::Refinery, recipe_id),
                    Inventory::new(4),
                    OutputInventory(Inventory::new(4)),
                    PowerConsumer::new(power::REFINERY_WATTS),
                    fio,
                    pos,
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.5, world_pos.z),
                        rotation: Quat::from_rotation_y(dir.angle_y()),
                        scale: Vec3::new(0.9, 1.0, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.refinery_mat.unwrap(),
                        tint: [1.0; 4],
                        visible: true,
                    },
                ));
            }
            BuildTool::UndergroundBelt => {
                // Auto-detect: if there's an unpaired Entry within range facing the same
                // direction, place an Exit and pair. Otherwise place an Entry.
                let mut found_entry: Option<hecs::Entity> = None;
                {
                    let mut scan_pos = pos;
                    for _ in 0..MAX_UNDERGROUND_DISTANCE {
                        scan_pos = scan_pos.neighbor(dir.opposite());
                        for (e, ub) in ctx.world.query::<&UndergroundBelt>().iter() {
                            if ub.grid_pos == scan_pos && ub.direction == dir
                                && ub.mode == UndergroundBeltMode::Entry && ub.pair.is_none()
                            {
                                found_entry = Some(e);
                                break;
                            }
                        }
                        if found_entry.is_some() {
                            break;
                        }
                    }
                }

                let (mode, tint) = if found_entry.is_some() {
                    (UndergroundBeltMode::Exit, [0.7, 0.8, 0.7, 1.0])
                } else {
                    (UndergroundBeltMode::Entry, [0.8, 0.7, 0.7, 1.0])
                };

                let entity = ctx.world.spawn((
                    UndergroundBelt::new(pos, dir, mode),
                    Transform3D {
                        position: Vec3::new(world_pos.x, 0.06, world_pos.z),
                        rotation: Quat::from_rotation_y(dir.angle_y()),
                        scale: Vec3::new(0.9, 0.12, 0.9),
                        ..Transform3D::default()
                    },
                    GlobalTransform::default(),
                    MeshRenderer {
                        mesh: self.cube_mesh.unwrap(),
                        material: self.underground_belt_mat.unwrap(),
                        tint,
                        visible: true,
                    },
                ));

                // Pair with found entry.
                if let Some(entry_entity) = found_entry {
                    ctx.world.get::<&mut UndergroundBelt>(entity).unwrap().pair = Some(entry_entity);
                    ctx.world.get::<&mut UndergroundBelt>(entry_entity).unwrap().pair = Some(entity);
                }
            }
        }
    }

    /// Check if any belt, machine, miner, or inserter occupies this grid cell.
    fn is_grid_occupied(&self, ctx: &Ctx, pos: GridPos) -> bool {
        for (_, belt) in ctx.world.query::<&BeltSegment>().iter() {
            if belt.grid_pos == pos {
                return true;
            }
        }
        for (_, ub) in ctx.world.query::<&UndergroundBelt>().iter() {
            if ub.grid_pos == pos {
                return true;
            }
        }
        for (_, pipe) in ctx.world.query::<&Pipe>().iter() {
            if pipe.grid_pos == pos {
                return true;
            }
        }
        for (_, grid) in ctx.world.query::<&GridPos>().iter() {
            if *grid == pos {
                return true;
            }
        }
        false
    }

    /// Draw small cubes on belt segments to represent items being transported.
    fn draw_belt_items(&self, ctx: &mut Ctx) {
        let item_mesh = self.cube_mesh.unwrap();
        let item_mat = self.item_mat.unwrap();
        let mut instances = Vec::new();

        for (_e, (belt, t)) in ctx.world.query::<(&BeltSegment, &Transform3D)>().iter() {
            let base = Vec3::new(t.position.x, 0.2, t.position.z);
            let dir_vec = match belt.direction {
                Dir4::North => Vec3::new(0.0, 0.0, -1.0),
                Dir4::East => Vec3::new(1.0, 0.0, 0.0),
                Dir4::South => Vec3::new(0.0, 0.0, 1.0),
                Dir4::West => Vec3::new(-1.0, 0.0, 0.0),
            };

            for (slot_idx, item) in belt.items.iter().enumerate() {
                if item.is_some() {
                    // Position item along the belt from back (-0.35) to front (+0.35).
                    let t_along = (slot_idx as f32 / (SLOTS_PER_BELT - 1) as f32) - 0.5;
                    let item_pos = base + dir_vec * (t_along * 0.7);
                    instances.push(esox_gfx::mesh3d::InstanceData::from_transform(
                        &esox_gfx::mesh3d::Transform {
                            position: item_pos,
                            scale: Vec3::splat(0.15),
                            ..esox_gfx::mesh3d::Transform::default()
                        },
                    ));
                }
            }
        }

        if !instances.is_empty() {
            ctx.renderer.draw_with_material(item_mesh, item_mat, &instances);
        }
    }

    fn draw_grid_highlight(&mut self, ctx: &mut Ctx, pos: GridPos, color: [f32; 4]) {
        let world_pos = pos.to_world(CELL_SIZE);
        self.overlay_renderer.add(esox_engine::ground_overlay::GroundOverlay {
            position: world_pos,
            size: (CELL_SIZE * 0.9, CELL_SIZE * 0.9),
            rotation: 0.0,
            color,
            material: self.ghost_mat.unwrap(),
        });
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("factory=info".parse().unwrap())
                .add_directive("esox_engine=info".parse().unwrap())
                .add_directive("esox_gfx=info".parse().unwrap())
                .add_directive("esox_platform=info".parse().unwrap()),
        )
        .init();

    let config = EngineConfig {
        platform: esox_engine::esox_platform::config::PlatformConfig {
            window: esox_platform::config::WindowConfig {
                title: "factory".into(),
                width: Some(1600),
                height: Some(900),
                ..Default::default()
            },
            ..Default::default()
        },
        postprocess: true,
        shadows: true,
        ..EngineConfig::default()
    };

    let game = FactoryGame::default();

    if let Err(e) = esox_engine::run(config, game) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
