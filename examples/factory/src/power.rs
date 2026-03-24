//! Power network — generation, distribution, and consumption.
//!
//! Power poles form a graph connecting buildings into networks.
//! Each tick the system computes supply/demand per network and sets
//! a satisfaction ratio on every consumer. Machines, miners, and
//! inserters check this ratio before operating.

use std::collections::HashMap;

use esox_engine::hecs;

use crate::belt::GridPos;

/// A power-generating building (e.g. steam engine).
pub struct PowerSource {
    /// Power output in watts.
    pub watts: f32,
}

/// A building that requires power to operate.
pub struct PowerConsumer {
    /// Power demand in watts.
    pub watts_required: f32,
    /// Satisfaction ratio (0.0–1.0), written each tick by the power system.
    pub satisfaction: f32,
}

/// A power pole that connects nearby buildings into a network.
pub struct PowerPole {
    /// Maximum Chebyshev distance to connect to another pole.
    pub reach: i32,
}

/// Default power pole reach in grid cells.
pub const POLE_REACH: i32 = 7;

/// Power output of a steam engine (watts).
pub const STEAM_ENGINE_WATTS: f32 = 100.0;

/// Power consumption values (watts).
pub const SMELTER_WATTS: f32 = 30.0;
pub const ASSEMBLER_WATTS: f32 = 40.0;
pub const MINER_WATTS: f32 = 25.0;
pub const INSERTER_WATTS: f32 = 10.0;
pub const REFINERY_WATTS: f32 = 50.0;
pub const CHEMICAL_PLANT_WATTS: f32 = 45.0;

impl PowerSource {
    pub fn new(watts: f32) -> Self {
        Self { watts }
    }
}

impl PowerConsumer {
    pub fn new(watts_required: f32) -> Self {
        Self {
            watts_required,
            satisfaction: 0.0,
        }
    }

    /// Whether this consumer has enough power to operate.
    pub fn is_powered(&self) -> bool {
        self.satisfaction >= 1.0
    }
}

impl PowerPole {
    pub fn new(reach: i32) -> Self {
        Self { reach }
    }
}

/// Simple union-find for grouping entities into power networks.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

/// Chebyshev (chessboard) distance between two grid positions.
fn chebyshev(a: &GridPos, b: &GridPos) -> i32 {
    (a.0.x - b.0.x).abs().max((a.0.y - b.0.y).abs())
}

/// Run the power network system for one tick.
///
/// 1. Collect all power poles and build connected components via union-find.
/// 2. Attach sources and consumers to the nearest pole within 1 tile.
/// 3. Sum supply/demand per network and compute satisfaction ratio.
/// 4. Write satisfaction back to each `PowerConsumer`.
pub fn power_tick_system(world: &mut hecs::World) {
    // Collect poles.
    let poles: Vec<(hecs::Entity, GridPos, i32)> = world
        .query_mut::<(&PowerPole, &GridPos)>()
        .into_iter()
        .map(|(e, (p, g))| (e, *g, p.reach))
        .collect();

    if poles.is_empty() {
        // No poles — all consumers get 0 satisfaction.
        let consumers: Vec<hecs::Entity> = world
            .query_mut::<&PowerConsumer>()
            .into_iter()
            .map(|(e, _)| e)
            .collect();
        for e in consumers {
            if let Ok(mut c) = world.get::<&mut PowerConsumer>(e) {
                c.satisfaction = 0.0;
            }
        }
        return;
    }

    let n = poles.len();
    let mut uf = UnionFind::new(n);

    // Index poles for lookup.
    let mut pole_positions: Vec<(GridPos, usize)> = Vec::with_capacity(n);
    for (i, &(_, pos, _)) in poles.iter().enumerate() {
        pole_positions.push((pos, i));
    }

    // Connect poles within mutual reach.
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = chebyshev(&poles[i].1, &poles[j].1);
            let max_reach = poles[i].2.min(poles[j].2);
            if dist <= max_reach {
                uf.union(i, j);
            }
        }
    }

    // Find the nearest pole within 1 tile of a grid position.
    let find_nearest_pole = |pos: &GridPos| -> Option<usize> {
        let mut best: Option<(i32, usize)> = None;
        for &(pole_pos, idx) in &pole_positions {
            let dist = chebyshev(pos, &pole_pos);
            if dist <= 1 {
                if best.map_or(true, |(d, _)| dist < d) {
                    best = Some((dist, idx));
                }
            }
        }
        best.map(|(_, idx)| idx)
    };

    // Collect sources and consumers.
    let sources: Vec<(GridPos, f32)> = world
        .query_mut::<(&PowerSource, &GridPos)>()
        .into_iter()
        .map(|(_, (s, g))| (*g, s.watts))
        .collect();

    let consumers: Vec<(hecs::Entity, GridPos, f32)> = world
        .query_mut::<(&PowerConsumer, &GridPos)>()
        .into_iter()
        .map(|(e, (c, g))| (e, *g, c.watts_required))
        .collect();

    // Sum supply per network.
    let mut network_supply: HashMap<usize, f32> = HashMap::new();
    let mut network_demand: HashMap<usize, f32> = HashMap::new();

    for (pos, watts) in &sources {
        if let Some(pole_idx) = find_nearest_pole(pos) {
            let root = uf.find(pole_idx);
            *network_supply.entry(root).or_insert(0.0) += watts;
        }
    }

    // Map each consumer to its network and sum demand.
    let mut consumer_networks: Vec<(hecs::Entity, Option<usize>)> = Vec::with_capacity(consumers.len());
    for (entity, pos, watts) in &consumers {
        if let Some(pole_idx) = find_nearest_pole(pos) {
            let root = uf.find(pole_idx);
            *network_demand.entry(root).or_insert(0.0) += watts;
            consumer_networks.push((*entity, Some(root)));
        } else {
            consumer_networks.push((*entity, None));
        }
    }

    // Compute satisfaction per network.
    let mut network_satisfaction: HashMap<usize, f32> = HashMap::new();
    for (&root, &demand) in &network_demand {
        let supply = network_supply.get(&root).copied().unwrap_or(0.0);
        let sat = if demand <= 0.0 {
            1.0
        } else {
            (supply / demand).min(1.0)
        };
        network_satisfaction.insert(root, sat);
    }

    // Write satisfaction to consumers.
    for (entity, network) in consumer_networks {
        let sat = match network {
            Some(root) => network_satisfaction.get(&root).copied().unwrap_or(0.0),
            None => 0.0,
        };
        if let Ok(mut c) = world.get::<&mut PowerConsumer>(entity) {
            c.satisfaction = sat;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumer_unpowered_without_source() {
        let mut world = hecs::World::new();
        world.spawn((PowerPole::new(7), GridPos::new(0, 0)));
        let consumer = world.spawn((PowerConsumer::new(30.0), GridPos::new(1, 0)));

        power_tick_system(&mut world);

        let c = world.get::<&PowerConsumer>(consumer).unwrap();
        assert_eq!(c.satisfaction, 0.0);
        assert!(!c.is_powered());
    }

    #[test]
    fn consumer_powered_with_sufficient_source() {
        let mut world = hecs::World::new();
        world.spawn((PowerPole::new(7), GridPos::new(0, 0)));
        world.spawn((PowerSource::new(100.0), GridPos::new(0, 1)));
        let consumer = world.spawn((PowerConsumer::new(30.0), GridPos::new(1, 0)));

        power_tick_system(&mut world);

        let c = world.get::<&PowerConsumer>(consumer).unwrap();
        assert!((c.satisfaction - 1.0).abs() < f32::EPSILON);
        assert!(c.is_powered());
    }

    #[test]
    fn satisfaction_partial_when_overloaded() {
        let mut world = hecs::World::new();
        world.spawn((PowerPole::new(7), GridPos::new(0, 0)));
        world.spawn((PowerSource::new(50.0), GridPos::new(0, 0)));
        let c1 = world.spawn((PowerConsumer::new(60.0), GridPos::new(1, 0)));
        let c2 = world.spawn((PowerConsumer::new(40.0), GridPos::new(0, 1)));

        power_tick_system(&mut world);

        let sat1 = world.get::<&PowerConsumer>(c1).unwrap().satisfaction;
        let sat2 = world.get::<&PowerConsumer>(c2).unwrap().satisfaction;
        assert!((sat1 - 0.5).abs() < f32::EPSILON);
        assert!((sat2 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn separate_networks_independent() {
        let mut world = hecs::World::new();

        // Network A: powered.
        world.spawn((PowerPole::new(3), GridPos::new(0, 0)));
        world.spawn((PowerSource::new(100.0), GridPos::new(0, 1)));
        let ca = world.spawn((PowerConsumer::new(50.0), GridPos::new(1, 0)));

        // Network B: unpowered (no source).
        world.spawn((PowerPole::new(3), GridPos::new(20, 20)));
        let cb = world.spawn((PowerConsumer::new(50.0), GridPos::new(21, 20)));

        power_tick_system(&mut world);

        let sat_a = world.get::<&PowerConsumer>(ca).unwrap().satisfaction;
        let sat_b = world.get::<&PowerConsumer>(cb).unwrap().satisfaction;
        assert!((sat_a - 1.0).abs() < f32::EPSILON);
        assert_eq!(sat_b, 0.0);
    }

    #[test]
    fn poles_bridge_networks() {
        let mut world = hecs::World::new();

        // Two poles within reach connect into one network.
        world.spawn((PowerPole::new(5), GridPos::new(0, 0)));
        world.spawn((PowerPole::new(5), GridPos::new(4, 0)));

        // Source near first pole, consumer near second.
        world.spawn((PowerSource::new(100.0), GridPos::new(0, 1)));
        let consumer = world.spawn((PowerConsumer::new(50.0), GridPos::new(5, 0)));

        power_tick_system(&mut world);

        let sat = world.get::<&PowerConsumer>(consumer).unwrap().satisfaction;
        assert!((sat - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn consumer_far_from_pole_stays_unpowered() {
        let mut world = hecs::World::new();
        world.spawn((PowerPole::new(3), GridPos::new(0, 0)));
        world.spawn((PowerSource::new(100.0), GridPos::new(0, 0)));
        let consumer = world.spawn((PowerConsumer::new(30.0), GridPos::new(10, 10)));

        power_tick_system(&mut world);

        let sat = world.get::<&PowerConsumer>(consumer).unwrap().satisfaction;
        assert_eq!(sat, 0.0);
    }

    #[test]
    fn no_poles_means_no_power() {
        let mut world = hecs::World::new();
        world.spawn((PowerSource::new(100.0), GridPos::new(0, 0)));
        let consumer = world.spawn((PowerConsumer::new(30.0), GridPos::new(0, 0)));

        power_tick_system(&mut world);

        let sat = world.get::<&PowerConsumer>(consumer).unwrap().satisfaction;
        assert_eq!(sat, 0.0);
    }
}
