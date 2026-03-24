//! Factory HUD — overlay panels for build tools, entity info, and global stats.

use esox_engine::hecs;
use esox_engine::esox_gfx::Color;
use esox_engine::esox_ui::fnv1a_runtime as hash;
use esox_engine::esox_ui::Ui;

use crate::belt::{BeltSegment, Dir4, GridPos, UndergroundBelt, UndergroundBeltMode, SLOTS_PER_BELT};
use crate::fluid::{FluidIO, FluidIOMode, FluidSource, FluidType, Pipe};
use crate::inserter::{Inserter, InserterState};
use crate::inventory::{Inventory, ItemId, ItemRegistry};
use crate::mining::{Miner, ResourceNode};
use crate::power::{PowerConsumer, PowerPole, PowerSource};
use crate::recipe::{Machine, MachineType, OutputInventory, RecipeRegistry};
use crate::BuildTool;

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

const HUD_BG: Color = Color::new(0.059, 0.059, 0.067, 0.85);
const HUD_BG_ITEM: Color = Color::new(0.122, 0.122, 0.133, 0.90);
const HUD_BORDER: Color = Color::new(0.25, 0.25, 0.28, 0.40);
const HUD_SELECTED: Color = Color::new(0.306, 0.533, 0.957, 0.25);

const COLOR_GREEN: Color = Color::new(0.243, 0.812, 0.416, 1.0);
const COLOR_AMBER: Color = Color::new(0.961, 0.737, 0.133, 1.0);
const COLOR_RED: Color = Color::new(0.941, 0.376, 0.376, 1.0);
const COLOR_ACCENT: Color = Color::new(0.306, 0.533, 0.957, 1.0);
const COLOR_DIM: Color = Color::new(0.360, 0.360, 0.360, 1.0);

// ---------------------------------------------------------------------------
// CursorEntity — pre-extracted display data for the entity at cursor
// ---------------------------------------------------------------------------

enum CursorEntity {
    Belt {
        direction: Dir4,
        items: [Option<ItemId>; SLOTS_PER_BELT],
        fill_count: usize,
    },
    UndergroundBelt {
        direction: Dir4,
        mode: UndergroundBeltMode,
        held_item: Option<ItemId>,
        paired: bool,
    },
    Inserter {
        state: InserterState,
        held_item: Option<ItemId>,
        progress_norm: f32,
        power_satisfaction: f32,
    },
    Machine {
        machine_type: MachineType,
        recipe_name: Option<String>,
        progress_norm: f32,
        active: bool,
        input_items: Vec<(String, u32)>,
        output_items: Vec<(String, u32)>,
        power_satisfaction: f32,
        fluid_ports: Vec<FluidPortInfo>,
    },
    Miner {
        resource_name: String,
        remaining: u32,
        progress_norm: f32,
        active: bool,
        power_satisfaction: f32,
    },
    ResourceNode {
        item_name: String,
        remaining: u32,
    },
    PowerSource {
        watts: f32,
    },
    PowerPole {
        reach: i32,
    },
    Pipe {
        fluid_name: Option<&'static str>,
        amount: f32,
        capacity: f32,
    },
    FluidSource {
        fluid_name: &'static str,
        rate: f32,
    },
}

struct FluidPortInfo {
    fluid_name: &'static str,
    mode: FluidIOMode,
    amount: f32,
    capacity: f32,
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn fluid_display_name(ft: FluidType) -> &'static str {
    match ft {
        FluidType::Water => "Water",
        FluidType::CrudeOil => "Crude Oil",
        FluidType::Petroleum => "Petroleum",
        FluidType::SulfuricAcid => "Sulfuric Acid",
    }
}

fn machine_type_name(mt: MachineType) -> &'static str {
    match mt {
        MachineType::Smelter => "Smelter",
        MachineType::Assembler => "Assembler",
        MachineType::Refinery => "Refinery",
        MachineType::ChemicalPlant => "Chem Plant",
    }
}

fn dir_label(d: Dir4) -> &'static str {
    match d {
        Dir4::North => "North ^",
        Dir4::East => "East >",
        Dir4::South => "South v",
        Dir4::West => "West <",
    }
}

fn power_color(satisfaction: f32) -> Color {
    if satisfaction >= 1.0 {
        COLOR_GREEN
    } else if satisfaction > 0.0 {
        COLOR_AMBER
    } else {
        COLOR_RED
    }
}

// ---------------------------------------------------------------------------
// ECS queries
// ---------------------------------------------------------------------------

fn query_power_totals(world: &hecs::World) -> (f32, f32) {
    let mut supply = 0.0f32;
    let mut demand = 0.0f32;
    for (_, src) in world.query::<&PowerSource>().iter() {
        supply += src.watts;
    }
    for (_, con) in world.query::<&PowerConsumer>().iter() {
        demand += con.watts_required;
    }
    (supply, demand)
}

fn query_entity_at_cursor(
    world: &hecs::World,
    pos: GridPos,
    items: &ItemRegistry,
    recipes: &RecipeRegistry,
) -> Option<CursorEntity> {
    // Belts
    for (_, belt) in world.query::<&BeltSegment>().iter() {
        if belt.grid_pos == pos {
            let fill = belt.items.iter().filter(|s| s.is_some()).count();
            return Some(CursorEntity::Belt {
                direction: belt.direction,
                items: belt.items,
                fill_count: fill,
            });
        }
    }

    // Underground belts
    for (_, ub) in world.query::<&UndergroundBelt>().iter() {
        if ub.grid_pos == pos {
            return Some(CursorEntity::UndergroundBelt {
                direction: ub.direction,
                mode: ub.mode,
                held_item: ub.held_item,
                paired: ub.pair.is_some(),
            });
        }
    }

    // Inserters (position = midpoint of pickup and dropoff)
    for (entity, ins) in world.query::<&Inserter>().iter() {
        let mid = GridPos(
            (ins.pickup_pos.0 + ins.dropoff_pos.0) / 2,
        );
        if mid == pos {
            let progress = if ins.cycle_ticks > 0 {
                ins.progress as f32 / ins.cycle_ticks as f32
            } else {
                0.0
            };
            let power_sat = world
                .get::<&PowerConsumer>(entity)
                .map(|c| c.satisfaction)
                .unwrap_or(1.0);
            return Some(CursorEntity::Inserter {
                state: ins.state,
                held_item: ins.held_item,
                progress_norm: progress,
                power_satisfaction: power_sat,
            });
        }
    }

    // Machines (have GridPos component)
    for (entity, (machine, grid)) in world.query::<(&Machine, &GridPos)>().iter() {
        if *grid == pos {
            let input_items = world
                .get::<&Inventory>(entity)
                .ok()
                .map(|inv| {
                    inv.slots
                        .iter()
                        .filter_map(|s| s.as_ref())
                        .map(|s| (items.name(s.item).to_string(), s.count))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let output_items = world
                .get::<&OutputInventory>(entity)
                .ok()
                .map(|out| {
                    out.0
                        .slots
                        .iter()
                        .filter_map(|s| s.as_ref())
                        .map(|s| (items.name(s.item).to_string(), s.count))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let recipe_name = machine
                .recipe
                .map(|r| recipes.get(r).name.clone());
            let progress = machine
                .recipe
                .map(|r| {
                    let dur = recipes.get(r).duration_ticks;
                    if dur > 0 {
                        machine.progress as f32 / dur as f32
                    } else {
                        0.0
                    }
                })
                .unwrap_or(0.0);
            let power_sat = world
                .get::<&PowerConsumer>(entity)
                .map(|c| c.satisfaction)
                .unwrap_or(1.0);

            let fluid_ports = world
                .get::<&FluidIO>(entity)
                .ok()
                .map(|fio| {
                    fio.ports
                        .iter()
                        .map(|p| FluidPortInfo {
                            fluid_name: fluid_display_name(p.fluid_type),
                            mode: p.mode,
                            amount: p.amount,
                            capacity: p.capacity,
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            return Some(CursorEntity::Machine {
                machine_type: machine.machine_type,
                recipe_name,
                progress_norm: progress,
                active: machine.active,
                input_items,
                output_items,
                power_satisfaction: power_sat,
                fluid_ports,
            });
        }
    }

    // Miners
    for (entity, miner) in world.query::<&Miner>().iter() {
        if miner.grid_pos == pos {
            let (res_name, remaining) = world
                .query::<&ResourceNode>()
                .iter()
                .find(|(_, n)| n.grid_pos == pos)
                .map(|(_, n)| (items.name(n.item).to_string(), n.remaining))
                .unwrap_or(("Unknown".into(), 0));
            let progress = if miner.cycle_ticks > 0 {
                miner.progress as f32 / miner.cycle_ticks as f32
            } else {
                0.0
            };
            let power_sat = world
                .get::<&PowerConsumer>(entity)
                .map(|c| c.satisfaction)
                .unwrap_or(1.0);
            return Some(CursorEntity::Miner {
                resource_name: res_name,
                remaining,
                progress_norm: progress,
                active: miner.active,
                power_satisfaction: power_sat,
            });
        }
    }

    // Resource nodes (without a miner on top)
    for (_, node) in world.query::<&ResourceNode>().iter() {
        if node.grid_pos == pos {
            return Some(CursorEntity::ResourceNode {
                item_name: items.name(node.item).to_string(),
                remaining: node.remaining,
            });
        }
    }

    // Power sources
    for (_, (src, grid)) in world.query::<(&PowerSource, &GridPos)>().iter() {
        if *grid == pos {
            return Some(CursorEntity::PowerSource { watts: src.watts });
        }
    }

    // Power poles
    for (_, (pole, grid)) in world.query::<(&PowerPole, &GridPos)>().iter() {
        if *grid == pos {
            return Some(CursorEntity::PowerPole { reach: pole.reach });
        }
    }

    // Pipes
    for (_, pipe) in world.query::<&Pipe>().iter() {
        if pipe.grid_pos == pos {
            return Some(CursorEntity::Pipe {
                fluid_name: pipe.fluid.map(fluid_display_name),
                amount: pipe.amount,
                capacity: pipe.capacity,
            });
        }
    }

    // Fluid sources
    for (_, (src, grid)) in world.query::<(&FluidSource, &GridPos)>().iter() {
        if *grid == pos {
            return Some(CursorEntity::FluidSource {
                fluid_name: fluid_display_name(src.fluid_type),
                rate: src.rate,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn draw_hud(
    ui: &mut Ui,
    world: &hecs::World,
    items: &ItemRegistry,
    recipes: &RecipeRegistry,
    build_tool: Option<BuildTool>,
    build_direction: Dir4,
    cursor_grid: GridPos,
    tick_count: u64,
    viewport: (u32, u32),
) {
    let vw = viewport.0 as f32;
    let vh = viewport.1 as f32;

    // -- Top-right: Global Stats --
    ui.padding(0.0, |ui| {
        let panel_w = 220.0;
        let margin = 10.0;
        ui.add_space(margin);
        ui.indent(vw - panel_w - margin, panel_w);
        draw_global_stats(ui, world, tick_count);
    });

    // -- Right side: Entity Info (only if entity at cursor) --
    let cursor_entity = query_entity_at_cursor(world, cursor_grid, items, recipes);
    if let Some(ref entity) = cursor_entity {
        ui.padding(0.0, |ui| {
            let panel_w = 280.0;
            let margin = 10.0;
            ui.add_space(4.0);
            ui.indent(vw - panel_w - margin, panel_w);
            draw_entity_info(ui, entity, items);
        });
    }

    // -- Bottom-center: Build Toolbar --
    ui.padding(0.0, |ui| {
        let toolbar_h = 80.0;
        let margin = 12.0;
        ui.add_space(vh - toolbar_h - margin);
        draw_build_toolbar(ui, build_tool, build_direction, vw);
    });
}

// ---------------------------------------------------------------------------
// Global Stats (top-right)
// ---------------------------------------------------------------------------

fn draw_global_stats(ui: &mut Ui, world: &hecs::World, tick_count: u64) {
    ui.box_container()
        .bg(HUD_BG)
        .border(HUD_BORDER, 1.0)
        .radius(6.0)
        .padding(10.0)
        .show(|ui| {
            ui.with_spacing(4.0, |ui| {
                // Power
                let (supply, demand) = query_power_totals(world);
                let pcolor = if supply >= demand {
                    COLOR_GREEN
                } else if supply >= demand * 0.5 {
                    COLOR_AMBER
                } else {
                    COLOR_RED
                };
                ui.row(|ui| {
                    ui.muted_label("Power");
                    ui.fill_space(130.0);
                    ui.label_colored(
                        &format!("{}W / {}W", supply as i32, demand as i32),
                        pcolor,
                    );
                });

                // Time
                let total_secs = tick_count / 60;
                let mins = total_secs / 60;
                let secs = total_secs % 60;
                ui.row(|ui| {
                    ui.muted_label("Time");
                    ui.fill_space(60.0);
                    ui.label(&format!("{:02}:{:02}", mins, secs));
                });
            });
        });
}

// ---------------------------------------------------------------------------
// Entity Info (right panel)
// ---------------------------------------------------------------------------

fn draw_entity_info(ui: &mut Ui, entity: &CursorEntity, items: &ItemRegistry) {
    ui.box_container()
        .bg(HUD_BG)
        .border(HUD_BORDER, 1.0)
        .radius(6.0)
        .padding(12.0)
        .show(|ui| {
            ui.with_spacing(4.0, |ui| {
                match entity {
                    CursorEntity::Belt { direction, items: belt_items, fill_count } => {
                        draw_belt_info(ui, *direction, belt_items, *fill_count, items);
                    }
                    CursorEntity::UndergroundBelt { direction, mode, held_item, paired } => {
                        draw_underground_belt_info(ui, *direction, *mode, *held_item, *paired, items);
                    }
                    CursorEntity::Inserter { state, held_item, progress_norm, power_satisfaction } => {
                        draw_inserter_info(ui, *state, *held_item, *progress_norm, *power_satisfaction, items);
                    }
                    CursorEntity::Machine {
                        machine_type, recipe_name, progress_norm, active,
                        input_items, output_items, power_satisfaction, fluid_ports,
                    } => {
                        draw_machine_info(
                            ui, *machine_type, recipe_name.as_deref(), *progress_norm,
                            *active, input_items, output_items, *power_satisfaction, fluid_ports,
                        );
                    }
                    CursorEntity::Miner { resource_name, remaining, progress_norm, active, power_satisfaction } => {
                        draw_miner_info(ui, resource_name, *remaining, *progress_norm, *active, *power_satisfaction);
                    }
                    CursorEntity::ResourceNode { item_name, remaining } => {
                        draw_resource_node_info(ui, item_name, *remaining);
                    }
                    CursorEntity::PowerSource { watts } => {
                        draw_power_source_info(ui, *watts);
                    }
                    CursorEntity::PowerPole { reach } => {
                        draw_power_pole_info(ui, *reach);
                    }
                    CursorEntity::Pipe { fluid_name, amount, capacity } => {
                        draw_pipe_info(ui, *fluid_name, *amount, *capacity);
                    }
                    CursorEntity::FluidSource { fluid_name, rate } => {
                        draw_fluid_source_info(ui, fluid_name, *rate);
                    }
                }
            });
        });
}

// -- Per-entity renderers --

fn draw_belt_info(
    ui: &mut Ui,
    direction: Dir4,
    belt_items: &[Option<ItemId>; SLOTS_PER_BELT],
    fill_count: usize,
    items: &ItemRegistry,
) {
    ui.row(|ui| {
        ui.label_colored("Belt", COLOR_ACCENT);
        ui.fill_space(80.0);
        ui.label(dir_label(direction));
    });
    ui.separator();

    ui.row(|ui| {
        ui.muted_label("Items");
        ui.fill_space(60.0);
        ui.label(&format!("{}/{}", fill_count, SLOTS_PER_BELT));
    });

    for (i, slot) in belt_items.iter().enumerate() {
        match slot {
            Some(id) => ui.label(&format!("  Slot {}: {}", i, items.name(*id))),
            None => ui.label_colored(&format!("  Slot {}: --", i), COLOR_DIM),
        };
    }

    ui.add_space(4.0);
    let fill_ratio = fill_count as f32 / SLOTS_PER_BELT as f32;
    let bar_color = if fill_count == SLOTS_PER_BELT { COLOR_AMBER } else { COLOR_ACCENT };
    ui.progress_bar_colored(fill_ratio, bar_color);
}

fn draw_underground_belt_info(
    ui: &mut Ui,
    direction: Dir4,
    mode: UndergroundBeltMode,
    held_item: Option<ItemId>,
    paired: bool,
    items: &ItemRegistry,
) {
    let mode_str = match mode {
        UndergroundBeltMode::Entry => "Entry",
        UndergroundBeltMode::Exit => "Exit",
    };
    ui.row(|ui| {
        ui.label_colored("Underground Belt", COLOR_ACCENT);
        ui.fill_space(50.0);
        ui.label(mode_str);
    });
    ui.separator();

    ui.row(|ui| {
        ui.muted_label("Direction");
        ui.fill_space(80.0);
        ui.label(dir_label(direction));
    });
    ui.row(|ui| {
        ui.muted_label("Paired");
        ui.fill_space(40.0);
        if paired {
            ui.label_colored("Yes", COLOR_GREEN);
        } else {
            ui.label_colored("No", COLOR_RED);
        }
    });
    ui.row(|ui| {
        ui.muted_label("Buffer");
        ui.fill_space(100.0);
        match held_item {
            Some(id) => ui.label(items.name(id)),
            None => ui.label_colored("Empty", COLOR_DIM),
        };
    });
}

fn draw_inserter_info(
    ui: &mut Ui,
    state: InserterState,
    held_item: Option<ItemId>,
    progress_norm: f32,
    power_satisfaction: f32,
    items: &ItemRegistry,
) {
    let state_str = match state {
        InserterState::Idle => "Idle",
        InserterState::Delivering => "Delivering",
        InserterState::Returning => "Returning",
    };
    let state_color = match state {
        InserterState::Idle => COLOR_DIM,
        InserterState::Delivering => COLOR_GREEN,
        InserterState::Returning => COLOR_AMBER,
    };
    ui.row(|ui| {
        ui.label_colored("Inserter", COLOR_ACCENT);
        ui.fill_space(80.0);
        ui.label_colored(state_str, state_color);
    });
    ui.separator();

    ui.row(|ui| {
        ui.muted_label("Holding");
        ui.fill_space(120.0);
        match held_item {
            Some(id) => ui.label(items.name(id)),
            None => ui.label_colored("Nothing", COLOR_DIM),
        };
    });

    if state != InserterState::Idle {
        ui.muted_label("Cycle");
        ui.progress_bar_colored(progress_norm, COLOR_GREEN);
    }

    draw_power_row(ui, power_satisfaction);
}

fn draw_machine_info(
    ui: &mut Ui,
    machine_type: MachineType,
    recipe_name: Option<&str>,
    progress_norm: f32,
    active: bool,
    input_items: &[(String, u32)],
    output_items: &[(String, u32)],
    power_satisfaction: f32,
    fluid_ports: &[FluidPortInfo],
) {
    ui.row(|ui| {
        ui.label_colored(machine_type_name(machine_type), COLOR_ACCENT);
        ui.fill_space(60.0);
        if active {
            ui.label_colored("Active", COLOR_GREEN);
        } else {
            ui.label_colored("Idle", COLOR_DIM);
        }
    });
    ui.separator();

    // Recipe
    ui.row(|ui| {
        ui.muted_label("Recipe");
        ui.fill_space(160.0);
        match recipe_name {
            Some(name) => ui.label(name),
            None => ui.label_colored("None", COLOR_DIM),
        };
    });

    // Progress
    if progress_norm > 0.0 {
        ui.muted_label("Progress");
        ui.progress_bar_colored(progress_norm, COLOR_GREEN);
    }

    // Power
    draw_power_row(ui, power_satisfaction);

    // Input inventory
    if !input_items.is_empty() {
        ui.add_space(4.0);
        ui.header_label("INPUT");
        for (name, count) in input_items {
            ui.row(|ui| {
                ui.label(name);
                ui.fill_space(50.0);
                ui.muted_label(&format!("x{}", count));
            });
        }
    }

    // Output inventory
    if !output_items.is_empty() {
        ui.add_space(4.0);
        ui.header_label("OUTPUT");
        for (name, count) in output_items {
            ui.row(|ui| {
                ui.label(name);
                ui.fill_space(50.0);
                ui.muted_label(&format!("x{}", count));
            });
        }
    }

    // Fluid ports
    if !fluid_ports.is_empty() {
        ui.add_space(4.0);
        ui.header_label("FLUIDS");
        for port in fluid_ports {
            let mode_str = match port.mode {
                FluidIOMode::Input => "In",
                FluidIOMode::Output => "Out",
            };
            ui.row(|ui| {
                ui.label(&format!("{} ({})", port.fluid_name, mode_str));
                ui.fill_space(80.0);
                ui.label(&format!("{:.0}/{:.0}", port.amount, port.capacity));
            });
        }
    }
}

fn draw_miner_info(
    ui: &mut Ui,
    resource_name: &str,
    remaining: u32,
    progress_norm: f32,
    active: bool,
    power_satisfaction: f32,
) {
    ui.row(|ui| {
        ui.label_colored("Miner", COLOR_ACCENT);
        ui.fill_space(60.0);
        if active {
            ui.label_colored("Mining", COLOR_GREEN);
        } else {
            ui.label_colored("Idle", COLOR_DIM);
        }
    });
    ui.separator();

    ui.row(|ui| {
        ui.muted_label("Resource");
        ui.fill_space(120.0);
        ui.label(resource_name);
    });

    let rem_color = if remaining > 1000 {
        Color::new(0.9, 0.9, 0.9, 1.0)
    } else if remaining > 100 {
        COLOR_AMBER
    } else {
        COLOR_RED
    };
    ui.row(|ui| {
        ui.muted_label("Remaining");
        ui.fill_space(80.0);
        ui.label_colored(&format!("{}", remaining), rem_color);
    });

    ui.muted_label("Progress");
    ui.progress_bar_colored(progress_norm, COLOR_GREEN);

    draw_power_row(ui, power_satisfaction);
}

fn draw_resource_node_info(ui: &mut Ui, item_name: &str, remaining: u32) {
    ui.label_colored("Resource Node", COLOR_ACCENT);
    ui.separator();
    ui.row(|ui| {
        ui.muted_label("Type");
        ui.fill_space(120.0);
        ui.label(item_name);
    });
    ui.row(|ui| {
        ui.muted_label("Remaining");
        ui.fill_space(80.0);
        ui.label(&format!("{}", remaining));
    });
}

fn draw_power_source_info(ui: &mut Ui, watts: f32) {
    ui.label_colored("Steam Engine", COLOR_ACCENT);
    ui.separator();
    ui.row(|ui| {
        ui.muted_label("Output");
        ui.fill_space(80.0);
        ui.label_colored(&format!("{}W", watts as i32), COLOR_GREEN);
    });
}

fn draw_power_pole_info(ui: &mut Ui, reach: i32) {
    ui.label_colored("Power Pole", COLOR_ACCENT);
    ui.separator();
    ui.row(|ui| {
        ui.muted_label("Reach");
        ui.fill_space(40.0);
        ui.label(&format!("{} tiles", reach));
    });
}

fn draw_pipe_info(ui: &mut Ui, fluid_name: Option<&str>, amount: f32, capacity: f32) {
    ui.label_colored("Pipe", COLOR_ACCENT);
    ui.separator();

    ui.row(|ui| {
        ui.muted_label("Fluid");
        ui.fill_space(100.0);
        match fluid_name {
            Some(name) => ui.label(name),
            None => ui.label_colored("Empty", COLOR_DIM),
        };
    });

    ui.row(|ui| {
        ui.muted_label("Fill");
        ui.fill_space(100.0);
        ui.label(&format!("{:.0} / {:.0}", amount, capacity));
    });

    let fill = if capacity > 0.0 { amount / capacity } else { 0.0 };
    let fill_color = if fill > 0.9 {
        COLOR_AMBER
    } else if fill > 0.0 {
        COLOR_ACCENT
    } else {
        COLOR_DIM
    };
    ui.progress_bar_colored(fill, fill_color);
}

fn draw_fluid_source_info(ui: &mut Ui, fluid_name: &str, rate: f32) {
    ui.label_colored("Fluid Source", COLOR_ACCENT);
    ui.separator();
    ui.row(|ui| {
        ui.muted_label("Fluid");
        ui.fill_space(100.0);
        ui.label(fluid_name);
    });
    ui.row(|ui| {
        ui.muted_label("Rate");
        ui.fill_space(100.0);
        ui.label(&format!("{:.1}/tick", rate));
    });
}

/// Reusable power satisfaction row.
fn draw_power_row(ui: &mut Ui, satisfaction: f32) {
    ui.row(|ui| {
        ui.muted_label("Power");
        ui.fill_space(60.0);
        ui.label_colored(
            &format!("{:.0}%", satisfaction * 100.0),
            power_color(satisfaction),
        );
    });
}

// ---------------------------------------------------------------------------
// Build Toolbar (bottom-center)
// ---------------------------------------------------------------------------

const TOOLS: [(BuildTool, &str, &str); 10] = [
    (BuildTool::Belt, "Belt", "1"),
    (BuildTool::Inserter, "Inserter", "2"),
    (BuildTool::Smelter, "Smelter", "3"),
    (BuildTool::Assembler, "Assembler", "4"),
    (BuildTool::Miner, "Miner", "5"),
    (BuildTool::SteamEngine, "Engine", "6"),
    (BuildTool::PowerPole, "Pole", "7"),
    (BuildTool::Pipe, "Pipe", "8"),
    (BuildTool::Refinery, "Refinery", "9"),
    (BuildTool::UndergroundBelt, "UG Belt", "0"),
];

fn draw_build_toolbar(
    ui: &mut Ui,
    build_tool: Option<BuildTool>,
    build_direction: Dir4,
    _viewport_w: f32,
) {
    let toolbar_w = 560.0;
    ui.center_horizontal(toolbar_w, |ui| {
        ui.box_container()
            .bg(HUD_BG)
            .border(HUD_BORDER, 1.0)
            .radius(8.0)
            .padding(8.0)
            .show(|ui| {
                // Tool buttons
                ui.row_spaced(4.0, |ui| {
                    for &(tool, name, hotkey) in &TOOLS {
                        let is_selected = build_tool == Some(tool);
                        let bg = if is_selected { HUD_SELECTED } else { HUD_BG_ITEM };
                        let label = format!("{} {}", hotkey, name);
                        let id = hash(&label);
                        ui.small_button(id, &label, bg);
                    }
                });

                // Direction indicator
                ui.add_space(4.0);
                ui.center_horizontal(120.0, |ui| {
                    ui.muted_label(&format!("Dir: {}  [R]", dir_label(build_direction)));
                });
            });
    });
}
