//! Chunked world — spatial partitioning with load/unload radius.

use std::collections::{HashMap, HashSet};

use glam::{IVec2, Vec3};
use hecs::Entity;

/// Configuration for the chunk system.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// World-space size of one chunk (square, XZ plane).
    pub chunk_size: f32,
    /// Load radius in chunks around the camera.
    pub load_radius: u32,
    /// Unload radius in chunks (must be > load_radius to avoid thrashing).
    pub unload_radius: u32,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 32.0,
            load_radius: 4,
            unload_radius: 6,
        }
    }
}

/// Integer coordinate of a chunk in chunk-space.
pub type ChunkCoord = IVec2;

/// ECS component marking which chunk an entity belongs to.
#[derive(Debug, Clone, Copy)]
pub struct ChunkMembership(pub ChunkCoord);

/// State of a single chunk.
#[derive(Debug)]
pub struct Chunk {
    pub coord: ChunkCoord,
    pub entities: Vec<Entity>,
    pub dirty: bool,
}

impl Chunk {
    fn new(coord: ChunkCoord) -> Self {
        Self {
            coord,
            entities: Vec::new(),
            dirty: false,
        }
    }
}

/// Result of a chunk manager update: which chunks entered/left the active set.
#[derive(Debug, Default)]
pub struct ChunkTransitions {
    /// Chunks that just entered the active set (need loading).
    pub activated: Vec<ChunkCoord>,
    /// Chunks that just left the active set (need unloading).
    pub deactivated: Vec<ChunkCoord>,
}

/// Manages spatial partitioning of the world into chunks.
pub struct ChunkManager {
    config: ChunkConfig,
    chunks: HashMap<ChunkCoord, Chunk>,
    active_chunks: HashSet<ChunkCoord>,
    camera_chunk: ChunkCoord,
}

impl ChunkManager {
    pub fn new(config: ChunkConfig) -> Self {
        assert!(
            config.unload_radius > config.load_radius,
            "unload_radius must be > load_radius to avoid thrashing"
        );
        Self {
            config,
            chunks: HashMap::new(),
            active_chunks: HashSet::new(),
            camera_chunk: IVec2::new(i32::MAX, i32::MAX), // force initial update
        }
    }

    /// Convert a world-space XZ position to a chunk coordinate.
    pub fn world_to_chunk(&self, pos: Vec3) -> ChunkCoord {
        IVec2::new(
            (pos.x / self.config.chunk_size).floor() as i32,
            (pos.z / self.config.chunk_size).floor() as i32,
        )
    }

    /// Update active chunks based on camera position.
    ///
    /// Returns which chunks were activated and deactivated.
    pub fn update(&mut self, camera_pos: Vec3) -> ChunkTransitions {
        let new_chunk = self.world_to_chunk(camera_pos);
        if new_chunk == self.camera_chunk {
            return ChunkTransitions::default();
        }
        self.camera_chunk = new_chunk;

        let load_r = self.config.load_radius as i32;
        let unload_r = self.config.unload_radius as i32;

        // Determine which chunks should be active.
        let mut desired: HashSet<ChunkCoord> = HashSet::new();
        for dx in -load_r..=load_r {
            for dz in -load_r..=load_r {
                desired.insert(new_chunk + IVec2::new(dx, dz));
            }
        }

        // Chunks to activate (in desired but not currently active).
        let activated: Vec<ChunkCoord> = desired
            .iter()
            .copied()
            .filter(|c| !self.active_chunks.contains(c))
            .collect();

        // Chunks to deactivate (currently active but outside unload radius).
        let deactivated: Vec<ChunkCoord> = self
            .active_chunks
            .iter()
            .copied()
            .filter(|c| {
                let diff = *c - new_chunk;
                diff.x.abs() > unload_r || diff.y.abs() > unload_r
            })
            .collect();

        // Apply transitions.
        for &coord in &activated {
            self.active_chunks.insert(coord);
            self.chunks.entry(coord).or_insert_with(|| Chunk::new(coord));
        }
        for &coord in &deactivated {
            self.active_chunks.remove(&coord);
        }

        ChunkTransitions {
            activated,
            deactivated,
        }
    }

    /// Register an entity in a chunk.
    pub fn register_entity(&mut self, entity: Entity, coord: ChunkCoord) {
        let chunk = self.chunks.entry(coord).or_insert_with(|| Chunk::new(coord));
        chunk.entities.push(entity);
        chunk.dirty = true;
    }

    /// Remove an entity from its chunk.
    pub fn unregister_entity(&mut self, entity: Entity, coord: ChunkCoord) {
        if let Some(chunk) = self.chunks.get_mut(&coord) {
            chunk.entities.retain(|&e| e != entity);
            chunk.dirty = true;
        }
    }

    /// Check if a chunk coordinate is currently active.
    pub fn is_active(&self, coord: ChunkCoord) -> bool {
        self.active_chunks.contains(&coord)
    }

    /// Get the set of currently active chunk coordinates.
    pub fn active_chunks(&self) -> &HashSet<ChunkCoord> {
        &self.active_chunks
    }

    /// Get a reference to a chunk by coordinate.
    pub fn chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    /// Get a mutable reference to a chunk by coordinate.
    pub fn chunk_mut(&mut self, coord: ChunkCoord) -> Option<&mut Chunk> {
        self.chunks.get_mut(&coord)
    }

    /// The chunk coordinate the camera is currently in.
    pub fn camera_chunk(&self) -> ChunkCoord {
        self.camera_chunk
    }

    /// The chunk configuration.
    pub fn config(&self) -> &ChunkConfig {
        &self.config
    }

    /// Iterate entities in all active chunks.
    pub fn active_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.active_chunks
            .iter()
            .filter_map(|coord| self.chunks.get(coord))
            .flat_map(|chunk| chunk.entities.iter().copied())
    }

    /// Remove all tracked data for a chunk (e.g. after unloading its entities).
    pub fn remove_chunk(&mut self, coord: ChunkCoord) {
        self.chunks.remove(&coord);
        self.active_chunks.remove(&coord);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_chunk_basic() {
        let mgr = ChunkManager::new(ChunkConfig {
            chunk_size: 32.0,
            ..Default::default()
        });
        assert_eq!(mgr.world_to_chunk(Vec3::new(0.0, 0.0, 0.0)), IVec2::ZERO);
        assert_eq!(mgr.world_to_chunk(Vec3::new(31.9, 0.0, 31.9)), IVec2::ZERO);
        assert_eq!(mgr.world_to_chunk(Vec3::new(32.0, 0.0, 0.0)), IVec2::new(1, 0));
        assert_eq!(mgr.world_to_chunk(Vec3::new(-1.0, 0.0, -1.0)), IVec2::new(-1, -1));
    }

    #[test]
    fn initial_update_activates_chunks() {
        let mut mgr = ChunkManager::new(ChunkConfig {
            chunk_size: 32.0,
            load_radius: 1,
            unload_radius: 2,
        });
        let transitions = mgr.update(Vec3::ZERO);
        // 3x3 = 9 chunks should activate
        assert_eq!(transitions.activated.len(), 9);
        assert!(transitions.deactivated.is_empty());
        assert_eq!(mgr.active_chunks().len(), 9);
    }

    #[test]
    fn moving_activates_and_deactivates() {
        let mut mgr = ChunkManager::new(ChunkConfig {
            chunk_size: 32.0,
            load_radius: 1,
            unload_radius: 2,
        });
        mgr.update(Vec3::ZERO);

        // Move far enough to trigger deactivation
        let transitions = mgr.update(Vec3::new(32.0 * 4.0, 0.0, 0.0));
        assert!(!transitions.activated.is_empty());
        assert!(!transitions.deactivated.is_empty());
    }

    #[test]
    fn no_transition_when_stationary() {
        let mut mgr = ChunkManager::new(ChunkConfig::default());
        mgr.update(Vec3::ZERO);
        let transitions = mgr.update(Vec3::new(1.0, 0.0, 1.0)); // same chunk
        assert!(transitions.activated.is_empty());
        assert!(transitions.deactivated.is_empty());
    }

    #[test]
    fn register_and_unregister_entity() {
        let mut mgr = ChunkManager::new(ChunkConfig::default());
        // Create a real entity in a temporary world to get a valid Entity handle.
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        let coord = IVec2::ZERO;

        mgr.register_entity(entity, coord);
        assert_eq!(mgr.chunk(coord).unwrap().entities.len(), 1);

        mgr.unregister_entity(entity, coord);
        assert!(mgr.chunk(coord).unwrap().entities.is_empty());
    }
}
