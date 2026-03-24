//! Machine/crafting engine — data-driven recipes and production machines.

use std::collections::HashMap;

use serde::Deserialize;

use esox_engine::hecs;

use crate::fluid::{FluidIO, FluidType};
use crate::inventory::{Inventory, ItemId, ItemRegistry};
use crate::power::PowerConsumer;

/// Type of machine that can run a recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum MachineType {
    Smelter,
    Assembler,
    Refinery,
    ChemicalPlant,
}

/// A recipe definition loaded from data.
#[derive(Debug, Clone, Deserialize)]
pub struct RecipeDef {
    pub id: String,
    pub name: String,
    pub machine_type: MachineType,
    /// Duration in ticks (at 60Hz).
    pub duration_ticks: u32,
    /// (item_string_id, count) pairs for inputs.
    pub inputs: Vec<(String, u32)>,
    /// (item_string_id, count) pairs for outputs.
    pub outputs: Vec<(String, u32)>,
    /// (fluid_type_name, amount) pairs for fluid inputs.
    #[serde(default)]
    pub fluid_inputs: Vec<(String, f32)>,
    /// (fluid_type_name, amount) pairs for fluid outputs.
    #[serde(default)]
    pub fluid_outputs: Vec<(String, f32)>,
}

/// Unique recipe identifier (index into the registry).
pub type RecipeId = u16;

/// A resolved recipe with numeric item ids.
#[derive(Debug, Clone)]
pub struct Recipe {
    pub id: String,
    pub name: String,
    pub machine_type: MachineType,
    pub duration_ticks: u32,
    pub inputs: Vec<(ItemId, u32)>,
    pub outputs: Vec<(ItemId, u32)>,
    pub fluid_inputs: Vec<(FluidType, f32)>,
    pub fluid_outputs: Vec<(FluidType, f32)>,
}

/// Registry of all recipes.
pub struct RecipeRegistry {
    recipes: Vec<Recipe>,
    name_to_id: HashMap<String, RecipeId>,
    /// Recipes grouped by machine type for quick lookup.
    by_machine: HashMap<MachineType, Vec<RecipeId>>,
}

impl RecipeRegistry {
    /// Load recipe definitions from RON and resolve item ids.
    pub fn load_from_ron(data: &str, items: &ItemRegistry) -> Self {
        let defs: Vec<RecipeDef> = ron::from_str(data).expect("failed to parse recipes.ron");
        let mut recipes = Vec::new();
        let mut name_to_id = HashMap::new();
        let mut by_machine: HashMap<MachineType, Vec<RecipeId>> = HashMap::new();

        for (i, def) in defs.into_iter().enumerate() {
            let inputs: Vec<(ItemId, u32)> = def
                .inputs
                .iter()
                .map(|(name, count)| {
                    let item_id = items
                        .id_of(name)
                        .unwrap_or_else(|| panic!("unknown item '{name}' in recipe '{}'", def.id));
                    (item_id, *count)
                })
                .collect();
            let outputs: Vec<(ItemId, u32)> = def
                .outputs
                .iter()
                .map(|(name, count)| {
                    let item_id = items
                        .id_of(name)
                        .unwrap_or_else(|| panic!("unknown item '{name}' in recipe '{}'", def.id));
                    (item_id, *count)
                })
                .collect();

            let fluid_inputs: Vec<(FluidType, f32)> = def
                .fluid_inputs
                .iter()
                .map(|(name, amount)| {
                    let ft = FluidType::from_name(name)
                        .unwrap_or_else(|| panic!("unknown fluid '{name}' in recipe '{}'", def.id));
                    (ft, *amount)
                })
                .collect();
            let fluid_outputs: Vec<(FluidType, f32)> = def
                .fluid_outputs
                .iter()
                .map(|(name, amount)| {
                    let ft = FluidType::from_name(name)
                        .unwrap_or_else(|| panic!("unknown fluid '{name}' in recipe '{}'", def.id));
                    (ft, *amount)
                })
                .collect();

            let id = i as RecipeId;
            name_to_id.insert(def.id.clone(), id);
            by_machine.entry(def.machine_type).or_default().push(id);

            recipes.push(Recipe {
                id: def.id,
                name: def.name,
                machine_type: def.machine_type,
                duration_ticks: def.duration_ticks,
                inputs,
                outputs,
                fluid_inputs,
                fluid_outputs,
            });
        }

        Self {
            recipes,
            name_to_id,
            by_machine,
        }
    }

    pub fn get(&self, id: RecipeId) -> &Recipe {
        &self.recipes[id as usize]
    }

    pub fn id_of(&self, name: &str) -> Option<RecipeId> {
        self.name_to_id.get(name).copied()
    }

    pub fn recipes_for(&self, machine_type: MachineType) -> &[RecipeId] {
        self.by_machine.get(&machine_type).map_or(&[], |v| v.as_slice())
    }
}

/// Machine component — a production building that crafts items.
pub struct Machine {
    pub machine_type: MachineType,
    /// Currently selected recipe (None = idle).
    pub recipe: Option<RecipeId>,
    /// Crafting progress in ticks (0 = not started).
    pub progress: u32,
    /// Whether the machine is actively crafting this tick.
    pub active: bool,
}

impl Machine {
    pub fn new(machine_type: MachineType) -> Self {
        Self {
            machine_type,
            recipe: None,
            progress: 0,
            active: false,
        }
    }

    pub fn with_recipe(machine_type: MachineType, recipe: RecipeId) -> Self {
        Self {
            machine_type,
            recipe: Some(recipe),
            progress: 0,
            active: false,
        }
    }
}

/// Run the machine crafting system for one tick.
///
/// Machines with a recipe check their input inventory for ingredients.
/// When all inputs are available, they consume them and start crafting.
/// When crafting completes, outputs go into the output inventory.
///
/// Each machine entity must have: `Machine`, `Inventory` (used as combined I/O),
/// plus an `OutputInventory` for output items.
pub struct OutputInventory(pub Inventory);

pub fn machine_tick_system(
    world: &mut hecs::World,
    items: &ItemRegistry,
    recipes: &RecipeRegistry,
) {
    // Collect machines to avoid borrow conflicts.
    let machine_entities: Vec<(hecs::Entity, MachineType, Option<RecipeId>, u32)> = world
        .query_mut::<&Machine>()
        .into_iter()
        .map(|(e, m)| (e, m.machine_type, m.recipe, m.progress))
        .collect();

    for (entity, _machine_type, recipe_id, progress) in machine_entities {
        let Some(recipe_id) = recipe_id else {
            // No recipe selected.
            if let Ok(mut machine) = world.get::<&mut Machine>(entity) {
                machine.active = false;
            }
            continue;
        };

        // Check power. Entities without a PowerConsumer are always powered (e.g. in tests).
        let powered = world
            .get::<&PowerConsumer>(entity)
            .map(|c| c.is_powered())
            .unwrap_or(true);
        if !powered {
            if let Ok(mut machine) = world.get::<&mut Machine>(entity) {
                machine.active = false;
            }
            continue;
        }

        let recipe = recipes.get(recipe_id);

        if progress > 0 {
            // Currently crafting — advance progress.
            let mut machine = world.get::<&mut Machine>(entity).unwrap();
            machine.progress += 1;
            machine.active = true;

            if machine.progress >= recipe.duration_ticks {
                // Crafting complete — try to output.
                machine.progress = 0;
                machine.active = false;
                drop(machine);

                // Add outputs to output inventory.
                if let Ok(mut output) = world.get::<&mut OutputInventory>(entity) {
                    for &(item, count) in &recipe.outputs {
                        output.0.insert(item, count, items);
                    }
                }
                // Add fluid outputs to FluidIO buffers.
                if !recipe.fluid_outputs.is_empty() {
                    if let Ok(mut fio) = world.get::<&mut FluidIO>(entity) {
                        for &(ft, amt) in &recipe.fluid_outputs {
                            fio.produce_fluid(ft, amt);
                        }
                    }
                }
            }
        } else {
            // Not crafting — check if inputs are available.
            let can_craft_items = {
                if let Ok(inv) = world.get::<&Inventory>(entity) {
                    recipe.inputs.iter().all(|&(item, count)| Inventory::has(&inv, item, count))
                } else {
                    false
                }
            };
            let can_craft_fluids = if recipe.fluid_inputs.is_empty() {
                true
            } else {
                world
                    .get::<&FluidIO>(entity)
                    .map(|fio| {
                        recipe
                            .fluid_inputs
                            .iter()
                            .all(|&(ft, amt)| fio.has_fluid_input(ft, amt))
                    })
                    .unwrap_or(false)
            };
            let can_craft = can_craft_items && can_craft_fluids;

            if can_craft {
                // Consume item inputs.
                {
                    let mut inv = world.get::<&mut Inventory>(entity).unwrap();
                    for &(item, count) in &recipe.inputs {
                        Inventory::remove(&mut inv, item, count);
                    }
                }
                // Consume fluid inputs.
                if !recipe.fluid_inputs.is_empty() {
                    if let Ok(mut fio) = world.get::<&mut FluidIO>(entity) {
                        for &(ft, amt) in &recipe.fluid_inputs {
                            fio.consume_fluid(ft, amt);
                        }
                    }
                }
                // Start crafting.
                let mut machine = world.get::<&mut Machine>(entity).unwrap();
                machine.progress = 1;
                machine.active = true;
            } else {
                if let Ok(mut machine) = world.get::<&mut Machine>(entity) {
                    machine.active = false;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (ItemRegistry, RecipeRegistry) {
        let items = ItemRegistry::load_from_ron(
            r#"[
                (id: "iron-ore", name: "Iron Ore", stack_size: 50),
                (id: "iron-plate", name: "Iron Plate", stack_size: 100),
            ]"#,
        );
        let recipes = RecipeRegistry::load_from_ron(
            r#"[(
                id: "smelt-iron",
                name: "Smelt Iron",
                machine_type: Smelter,
                duration_ticks: 10,
                inputs: [("iron-ore", 1)],
                outputs: [("iron-plate", 1)],
            )]"#,
            &items,
        );
        (items, recipes)
    }

    #[test]
    fn machine_crafts_when_inputs_available() {
        let (items, recipes) = setup();
        let ore = items.id_of("iron-ore").unwrap();
        let plate = items.id_of("iron-plate").unwrap();
        let recipe_id = recipes.id_of("smelt-iron").unwrap();

        let mut world = hecs::World::new();

        let mut input_inv = Inventory::new(4);
        input_inv.insert(ore, 5, &items);

        let entity = world.spawn((
            Machine::with_recipe(MachineType::Smelter, recipe_id),
            input_inv,
            OutputInventory(Inventory::new(4)),
        ));

        // Tick until recipe completes (duration = 10).
        for _ in 0..10 {
            machine_tick_system(&mut world, &items, &recipes);
        }

        let output = world.get::<&OutputInventory>(entity).unwrap();
        assert_eq!(output.0.count_item(plate), 1);

        // Input should have consumed 1 ore.
        let inv = world.get::<&Inventory>(entity).unwrap();
        assert_eq!(inv.count_item(ore), 4);
    }

    #[test]
    fn machine_idles_without_inputs() {
        let (items, recipes) = setup();
        let recipe_id = recipes.id_of("smelt-iron").unwrap();

        let mut world = hecs::World::new();
        let entity = world.spawn((
            Machine::with_recipe(MachineType::Smelter, recipe_id),
            Inventory::new(4), // empty
            OutputInventory(Inventory::new(4)),
        ));

        machine_tick_system(&mut world, &items, &recipes);

        let machine = world.get::<&Machine>(entity).unwrap();
        assert_eq!(machine.progress, 0);
        assert!(!machine.active);
    }

    #[test]
    fn machine_crafts_with_fluid_io() {
        use crate::fluid::{FluidIO, FluidIOMode, FluidPort, FluidType};

        let items = ItemRegistry::load_from_ron(r#"[]"#);
        let recipes = RecipeRegistry::load_from_ron(
            r#"[(
                id: "refine",
                name: "Refine",
                machine_type: Refinery,
                duration_ticks: 10,
                inputs: [],
                outputs: [],
                fluid_inputs: [("crude-oil", 10.0)],
                fluid_outputs: [("petroleum", 5.0)],
            )]"#,
            &items,
        );
        let recipe_id = recipes.id_of("refine").unwrap();

        let mut world = hecs::World::new();

        let fio = FluidIO::new(vec![
            FluidPort {
                fluid_type: FluidType::CrudeOil,
                rate: 10.0,
                mode: FluidIOMode::Input,
                amount: 50.0,
                capacity: 100.0,
            },
            FluidPort {
                fluid_type: FluidType::Petroleum,
                rate: 10.0,
                mode: FluidIOMode::Output,
                amount: 0.0,
                capacity: 100.0,
            },
        ]);

        let entity = world.spawn((
            Machine::with_recipe(MachineType::Refinery, recipe_id),
            Inventory::new(4),
            OutputInventory(Inventory::new(4)),
            fio,
        ));

        // Run 10 ticks to complete one recipe cycle.
        for _ in 0..10 {
            machine_tick_system(&mut world, &items, &recipes);
        }

        let fio = world.get::<&FluidIO>(entity).unwrap();
        // Should have consumed 10 crude oil (50 - 10 = 40).
        assert!((fio.ports[0].amount - 40.0).abs() < f32::EPSILON);
        // Should have produced 5 petroleum.
        assert!((fio.ports[1].amount - 5.0).abs() < f32::EPSILON);
    }
}
