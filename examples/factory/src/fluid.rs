//! Fluid/pipe network — simplified bucket model for fluid transport.
//!
//! Pipes carry fluids between buildings. Each tick the system equalizes
//! fill levels between adjacent pipes of the same fluid type. Machines
//! with [`FluidIO`] ports consume/produce fluid from/to adjacent pipes.
//! [`FluidSource`] entities (oil wells) continuously produce fluid.

use std::collections::HashMap;

use esox_engine::glam::IVec2;
use esox_engine::hecs;
use serde::Deserialize;

use crate::belt::GridPos;

/// Maximum fluid a single pipe segment can hold.
pub const PIPE_CAPACITY: f32 = 100.0;

/// Fraction of fill-level difference equalized between adjacent pipes per tick.
const EQUALIZE_RATE: f32 = 0.4;

/// Default crude oil production rate (units per tick).
pub const CRUDE_OIL_RATE: f32 = 0.5;

/// The four cardinal direction offsets for neighbor lookup.
const CARDINAL_OFFSETS: [IVec2; 4] = [
    IVec2::new(1, 0),
    IVec2::new(-1, 0),
    IVec2::new(0, 1),
    IVec2::new(0, -1),
];

// ---------------------------------------------------------------------------
// FluidType
// ---------------------------------------------------------------------------

/// Types of fluid that can flow through pipes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum FluidType {
    Water,
    CrudeOil,
    Petroleum,
    SulfuricAcid,
}

impl FluidType {
    /// Parse from a string identifier used in recipe definitions.
    pub fn from_name(s: &str) -> Option<FluidType> {
        match s {
            "water" => Some(FluidType::Water),
            "crude-oil" => Some(FluidType::CrudeOil),
            "petroleum" => Some(FluidType::Petroleum),
            "sulfuric-acid" => Some(FluidType::SulfuricAcid),
            _ => None,
        }
    }

    /// String identifier for display / serialization.
    pub fn name(self) -> &'static str {
        match self {
            FluidType::Water => "water",
            FluidType::CrudeOil => "crude-oil",
            FluidType::Petroleum => "petroleum",
            FluidType::SulfuricAcid => "sulfuric-acid",
        }
    }
}

// ---------------------------------------------------------------------------
// Pipe
// ---------------------------------------------------------------------------

/// A pipe segment occupying one grid cell. Carries a single fluid type.
pub struct Pipe {
    pub grid_pos: GridPos,
    /// What fluid is in this pipe (`None` if empty).
    pub fluid: Option<FluidType>,
    /// Current fluid amount (0.0 .. capacity).
    pub amount: f32,
    /// Maximum fluid this pipe can hold.
    pub capacity: f32,
}

impl Pipe {
    pub fn new(grid_pos: GridPos) -> Self {
        Self {
            grid_pos,
            fluid: None,
            amount: 0.0,
            capacity: PIPE_CAPACITY,
        }
    }
}

// ---------------------------------------------------------------------------
// FluidIO — machine fluid ports
// ---------------------------------------------------------------------------

/// Whether a fluid port consumes or produces fluid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidIOMode {
    Input,
    Output,
}

/// A single fluid connection point on a machine.
#[derive(Debug, Clone)]
pub struct FluidPort {
    pub fluid_type: FluidType,
    /// Maximum transfer rate per tick between port buffer and adjacent pipe.
    pub rate: f32,
    pub mode: FluidIOMode,
    /// Internal fluid buffer amount.
    pub amount: f32,
    /// Internal buffer capacity.
    pub capacity: f32,
}

/// Fluid I/O component for machines that process fluids.
pub struct FluidIO {
    pub ports: Vec<FluidPort>,
}

impl FluidIO {
    pub fn new(ports: Vec<FluidPort>) -> Self {
        Self { ports }
    }

    /// Create ports matching a recipe's fluid requirements.
    pub fn from_recipe(
        fluid_inputs: &[(FluidType, f32)],
        fluid_outputs: &[(FluidType, f32)],
    ) -> Self {
        let mut ports = Vec::new();
        for &(fluid_type, _) in fluid_inputs {
            ports.push(FluidPort {
                fluid_type,
                rate: 2.0,
                mode: FluidIOMode::Input,
                amount: 0.0,
                capacity: PIPE_CAPACITY,
            });
        }
        for &(fluid_type, _) in fluid_outputs {
            ports.push(FluidPort {
                fluid_type,
                rate: 2.0,
                mode: FluidIOMode::Output,
                amount: 0.0,
                capacity: PIPE_CAPACITY,
            });
        }
        Self { ports }
    }

    /// Check if input buffers have at least `amount` of `fluid_type`.
    pub fn has_fluid_input(&self, fluid_type: FluidType, amount: f32) -> bool {
        let total: f32 = self
            .ports
            .iter()
            .filter(|p| p.mode == FluidIOMode::Input && p.fluid_type == fluid_type)
            .map(|p| p.amount)
            .sum();
        total >= amount
    }

    /// Consume fluid from input port buffers.
    pub fn consume_fluid(&mut self, fluid_type: FluidType, mut amount: f32) {
        for port in &mut self.ports {
            if port.mode == FluidIOMode::Input && port.fluid_type == fluid_type {
                let take = amount.min(port.amount);
                port.amount -= take;
                amount -= take;
                if amount <= 0.0 {
                    break;
                }
            }
        }
    }

    /// Produce fluid into output port buffers.
    pub fn produce_fluid(&mut self, fluid_type: FluidType, mut amount: f32) {
        for port in &mut self.ports {
            if port.mode == FluidIOMode::Output && port.fluid_type == fluid_type {
                let space = port.capacity - port.amount;
                let add = amount.min(space);
                port.amount += add;
                amount -= add;
                if amount <= 0.0 {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FluidSource
// ---------------------------------------------------------------------------

/// A building that continuously produces fluid into adjacent pipes (e.g. oil well).
pub struct FluidSource {
    pub fluid_type: FluidType,
    /// Units of fluid produced per tick.
    pub rate: f32,
}

impl FluidSource {
    pub fn new(fluid_type: FluidType, rate: f32) -> Self {
        Self { fluid_type, rate }
    }
}

// ---------------------------------------------------------------------------
// fluid_tick_system
// ---------------------------------------------------------------------------

/// Run the fluid network system for one tick.
///
/// 1. Fluid sources produce into adjacent pipes.
/// 2. Machine output ports push fluid to adjacent pipes.
/// 3. Pipes equalize fill levels with adjacent same-fluid pipes.
/// 4. Machine input ports pull fluid from adjacent pipes.
pub fn fluid_tick_system(world: &mut hecs::World) {
    // ── Collect pipe data into a working buffer ──
    let mut pipes: Vec<(hecs::Entity, GridPos, Option<FluidType>, f32, f32)> = world
        .query_mut::<&Pipe>()
        .into_iter()
        .map(|(e, p)| (e, p.grid_pos, p.fluid, p.amount, p.capacity))
        .collect();

    if pipes.is_empty() {
        return;
    }

    let mut grid_lookup: HashMap<GridPos, usize> = HashMap::with_capacity(pipes.len());
    for (i, &(_, pos, ..)) in pipes.iter().enumerate() {
        grid_lookup.insert(pos, i);
    }

    // ── Phase 1: Fluid sources → adjacent pipes ──
    let sources: Vec<(GridPos, FluidType, f32)> = world
        .query_mut::<(&FluidSource, &GridPos)>()
        .into_iter()
        .map(|(_, (s, g))| (*g, s.fluid_type, s.rate))
        .collect();

    for (src_pos, fluid_type, rate) in &sources {
        let mut remaining = *rate;
        for &offset in &CARDINAL_OFFSETS {
            if remaining <= 0.0 {
                break;
            }
            let neighbor = GridPos(src_pos.0 + offset);
            if let Some(&idx) = grid_lookup.get(&neighbor) {
                let (_, _, ref mut fluid, ref mut amount, capacity) = pipes[idx];
                if fluid.is_none() || *fluid == Some(*fluid_type) {
                    let space = capacity - *amount;
                    let add = remaining.min(space);
                    if add > 0.0 {
                        *amount += add;
                        *fluid = Some(*fluid_type);
                        remaining -= add;
                    }
                }
            }
        }
    }

    // ── Phase 2: Machine output ports → adjacent pipes ──
    let machine_ios: Vec<(
        hecs::Entity,
        GridPos,
        Vec<(FluidType, f32, FluidIOMode, f32, f32)>,
    )> = world
        .query_mut::<(&FluidIO, &GridPos)>()
        .into_iter()
        .map(|(e, (fio, g))| {
            let ports: Vec<_> = fio
                .ports
                .iter()
                .map(|p| (p.fluid_type, p.rate, p.mode, p.amount, p.capacity))
                .collect();
            (e, *g, ports)
        })
        .collect();

    let mut fio_changes: Vec<(hecs::Entity, usize, f32)> = Vec::new();

    for (entity, machine_pos, ports) in &machine_ios {
        for (port_idx, &(fluid_type, rate, mode, port_amount, _)) in ports.iter().enumerate() {
            if mode != FluidIOMode::Output {
                continue;
            }
            let mut to_push = port_amount.min(rate);
            let mut new_port_amount = port_amount;
            for &offset in &CARDINAL_OFFSETS {
                if to_push <= 0.0 {
                    break;
                }
                let neighbor = GridPos(machine_pos.0 + offset);
                if let Some(&idx) = grid_lookup.get(&neighbor) {
                    let (_, _, ref mut fluid, ref mut amount, capacity) = pipes[idx];
                    if fluid.is_none() || *fluid == Some(fluid_type) {
                        let space = capacity - *amount;
                        let push = to_push.min(space);
                        if push > 0.0 {
                            *amount += push;
                            *fluid = Some(fluid_type);
                            new_port_amount -= push;
                            to_push -= push;
                        }
                    }
                }
            }
            if (new_port_amount - port_amount).abs() > f32::EPSILON {
                fio_changes.push((*entity, port_idx, new_port_amount));
            }
        }
    }

    // ── Phase 3: Pipe equalization ──
    let right = IVec2::new(1, 0);
    let down = IVec2::new(0, 1);

    for i in 0..pipes.len() {
        for &offset in &[right, down] {
            let pos = pipes[i].1;
            let neighbor_pos = GridPos(pos.0 + offset);
            let Some(&j) = grid_lookup.get(&neighbor_pos) else {
                continue;
            };

            let amount_a = pipes[i].3;
            let fluid_a = pipes[i].2;
            let amount_b = pipes[j].3;
            let fluid_b = pipes[j].2;

            let compatible = match (fluid_a, fluid_b) {
                (Some(a), Some(b)) => a == b,
                (Some(_), None) if amount_a > 0.0 => true,
                (None, Some(_)) if amount_b > 0.0 => true,
                _ => false,
            };
            if !compatible {
                continue;
            }

            let diff = amount_a - amount_b;
            let transfer = diff * EQUALIZE_RATE;

            let cap_a = pipes[i].4;
            let cap_b = pipes[j].4;
            let max_give = amount_a.min(cap_b - amount_b);
            let max_take = amount_b.min(cap_a - amount_a);
            let transfer = transfer.clamp(-max_take, max_give);

            pipes[i].3 -= transfer;
            pipes[j].3 += transfer;

            // Propagate fluid type to empty neighbors.
            if pipes[j].2.is_none() && pipes[j].3 > 0.001 {
                pipes[j].2 = pipes[i].2;
            }
            if pipes[i].2.is_none() && pipes[i].3 > 0.001 {
                pipes[i].2 = pipes[j].2;
            }
        }
    }

    // ── Phase 4: Adjacent pipes → machine input ports ──
    for (entity, machine_pos, ports) in &machine_ios {
        for (port_idx, &(fluid_type, rate, mode, port_amount, port_capacity)) in
            ports.iter().enumerate()
        {
            if mode != FluidIOMode::Input {
                continue;
            }
            let space = port_capacity - port_amount;
            let mut to_pull = space.min(rate);
            let mut new_port_amount = port_amount;

            for &offset in &CARDINAL_OFFSETS {
                if to_pull <= 0.0 {
                    break;
                }
                let neighbor = GridPos(machine_pos.0 + offset);
                if let Some(&idx) = grid_lookup.get(&neighbor) {
                    let (_, _, ref mut fluid, ref mut amount, _) = pipes[idx];
                    if *fluid == Some(fluid_type) && *amount > 0.0 {
                        let pull = to_pull.min(*amount);
                        *amount -= pull;
                        new_port_amount += pull;
                        to_pull -= pull;
                        if *amount <= 0.001 {
                            *amount = 0.0;
                            *fluid = None;
                        }
                    }
                }
            }
            if (new_port_amount - port_amount).abs() > f32::EPSILON {
                fio_changes.push((*entity, port_idx, new_port_amount));
            }
        }
    }

    // ── Phase 5: Write back pipe data ──
    for &(entity, _, fluid, amount, _) in &pipes {
        if let Ok(mut pipe) = world.get::<&mut Pipe>(entity) {
            if amount < 0.001 {
                pipe.amount = 0.0;
                pipe.fluid = None;
            } else {
                pipe.amount = amount;
                pipe.fluid = fluid;
            }
        }
    }

    // ── Phase 6: Write back FluidIO changes ──
    for (entity, port_idx, new_amount) in fio_changes {
        if let Ok(mut fio) = world.get::<&mut FluidIO>(entity) {
            if port_idx < fio.ports.len() {
                fio.ports[port_idx].amount = new_amount;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipes_equalize() {
        let mut world = hecs::World::new();

        world.spawn((Pipe {
            grid_pos: GridPos::new(0, 0),
            fluid: Some(FluidType::Water),
            amount: 80.0,
            capacity: 100.0,
        },));
        world.spawn((Pipe {
            grid_pos: GridPos::new(1, 0),
            fluid: None,
            amount: 0.0,
            capacity: 100.0,
        },));

        for _ in 0..10 {
            fluid_tick_system(&mut world);
        }

        let amounts: Vec<f32> = world
            .query_mut::<&Pipe>()
            .into_iter()
            .map(|(_, p)| p.amount)
            .collect();
        assert!(
            (amounts[0] - amounts[1]).abs() < 1.0,
            "pipes should equalize: {:?}",
            amounts
        );
        // Total fluid is conserved.
        let total: f32 = amounts.iter().sum();
        assert!((total - 80.0).abs() < 0.01);
    }

    #[test]
    fn different_fluids_dont_mix() {
        let mut world = hecs::World::new();

        world.spawn((Pipe {
            grid_pos: GridPos::new(0, 0),
            fluid: Some(FluidType::Water),
            amount: 80.0,
            capacity: 100.0,
        },));
        world.spawn((Pipe {
            grid_pos: GridPos::new(1, 0),
            fluid: Some(FluidType::CrudeOil),
            amount: 20.0,
            capacity: 100.0,
        },));

        fluid_tick_system(&mut world);

        for (_, pipe) in world.query_mut::<&Pipe>() {
            if pipe.grid_pos == GridPos::new(0, 0) {
                assert_eq!(pipe.fluid, Some(FluidType::Water));
                assert!((pipe.amount - 80.0).abs() < f32::EPSILON);
            }
            if pipe.grid_pos == GridPos::new(1, 0) {
                assert_eq!(pipe.fluid, Some(FluidType::CrudeOil));
                assert!((pipe.amount - 20.0).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn fluid_source_fills_adjacent_pipe() {
        let mut world = hecs::World::new();

        world.spawn((
            FluidSource::new(FluidType::CrudeOil, 5.0),
            GridPos::new(0, 0),
        ));
        world.spawn((Pipe::new(GridPos::new(1, 0)),));

        fluid_tick_system(&mut world);

        for (_, pipe) in world.query_mut::<&Pipe>() {
            assert_eq!(pipe.fluid, Some(FluidType::CrudeOil));
            assert!((pipe.amount - 5.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn machine_output_pushes_to_pipe() {
        let mut world = hecs::World::new();

        let fio = FluidIO::new(vec![FluidPort {
            fluid_type: FluidType::Petroleum,
            rate: 10.0,
            mode: FluidIOMode::Output,
            amount: 30.0,
            capacity: 100.0,
        }]);
        world.spawn((fio, GridPos::new(0, 0)));
        world.spawn((Pipe::new(GridPos::new(1, 0)),));

        fluid_tick_system(&mut world);

        for (_, pipe) in world.query_mut::<&Pipe>() {
            assert_eq!(pipe.fluid, Some(FluidType::Petroleum));
            assert!((pipe.amount - 10.0).abs() < f32::EPSILON);
        }
        for (_, fio) in world.query_mut::<&FluidIO>() {
            assert!((fio.ports[0].amount - 20.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn machine_input_pulls_from_pipe() {
        let mut world = hecs::World::new();

        let fio = FluidIO::new(vec![FluidPort {
            fluid_type: FluidType::CrudeOil,
            rate: 5.0,
            mode: FluidIOMode::Input,
            amount: 0.0,
            capacity: 100.0,
        }]);
        world.spawn((fio, GridPos::new(0, 0)));

        let mut pipe = Pipe::new(GridPos::new(1, 0));
        pipe.fluid = Some(FluidType::CrudeOil);
        pipe.amount = 50.0;
        world.spawn((pipe,));

        fluid_tick_system(&mut world);

        for (_, fio) in world.query_mut::<&FluidIO>() {
            assert!((fio.ports[0].amount - 5.0).abs() < f32::EPSILON);
        }
        for (_, pipe) in world.query_mut::<&Pipe>() {
            assert!((pipe.amount - 45.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn fluid_type_from_name_roundtrip() {
        for ft in [
            FluidType::Water,
            FluidType::CrudeOil,
            FluidType::Petroleum,
            FluidType::SulfuricAcid,
        ] {
            assert_eq!(FluidType::from_name(ft.name()), Some(ft));
        }
        assert_eq!(FluidType::from_name("invalid"), None);
    }

    #[test]
    fn empty_pipe_cleared() {
        let mut world = hecs::World::new();

        world.spawn((Pipe {
            grid_pos: GridPos::new(0, 0),
            fluid: Some(FluidType::Water),
            amount: 0.0005,
            capacity: 100.0,
        },));

        fluid_tick_system(&mut world);

        for (_, pipe) in world.query_mut::<&Pipe>() {
            assert_eq!(pipe.fluid, None);
            assert_eq!(pipe.amount, 0.0);
        }
    }

    #[test]
    fn source_doesnt_overflow_pipe() {
        let mut world = hecs::World::new();

        world.spawn((
            FluidSource::new(FluidType::Water, 200.0),
            GridPos::new(0, 0),
        ));
        world.spawn((Pipe::new(GridPos::new(1, 0)),));

        fluid_tick_system(&mut world);

        for (_, pipe) in world.query_mut::<&Pipe>() {
            assert!((pipe.amount - PIPE_CAPACITY).abs() < f32::EPSILON);
        }
    }
}
