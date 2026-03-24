//! Per-chunk save/load utilities.
//!
//! Provides helpers to serialize/deserialize individual chunks as [`SceneFile`]s.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hecs::{Entity, World};

use crate::assets::AssetManager;
use crate::chunk::{ChunkCoord, ChunkManager, ChunkMembership};
use crate::ecs::components::Transform3D;
use crate::physics::PhysicsBackend;
use crate::physics::entity_map::PhysicsEntityMap;
use crate::scene::{SceneFile, SceneEntity, save_scene, load_scene};

/// Save a single chunk's entities to a [`SceneFile`].
///
/// Only entities with `Transform3D` and `ChunkMembership` matching `coord` are included.
pub fn save_chunk(
    world: &World,
    assets: &AssetManager,
    chunk_manager: &ChunkManager,
    coord: ChunkCoord,
) -> Option<SceneFile> {
    let chunk = chunk_manager.chunk(coord)?;
    if chunk.entities.is_empty() {
        return None;
    }

    // Delegate to the full save_scene, then filter to only entities in this chunk.
    let full_scene = save_scene(world, assets);
    let chunk_entity_set: std::collections::HashSet<Entity> =
        chunk.entities.iter().copied().collect();

    // Build a mapping of full-scene entity IDs that belong to this chunk.
    let mut scene_ids_in_chunk = std::collections::HashSet::new();
    for (i, (_entity, (_t, membership))) in world
        .query::<(&Transform3D, &ChunkMembership)>()
        .iter()
        .enumerate()
    {
        if membership.0 == coord && chunk_entity_set.contains(&_entity) {
            scene_ids_in_chunk.insert(i as u32);
        }
    }

    let entities: Vec<SceneEntity> = full_scene
        .entities
        .into_iter()
        .filter(|se| scene_ids_in_chunk.contains(&se.id))
        .collect();

    if entities.is_empty() {
        return None;
    }

    Some(SceneFile {
        version: full_scene.version,
        entities,
    })
}

/// Load a chunk from a [`SceneFile`] into the world.
///
/// Spawned entities are registered in the chunk manager with [`ChunkMembership`].
/// Returns a mapping from scene-local IDs to live [`Entity`] handles.
pub fn load_chunk(
    scene: &SceneFile,
    world: &mut World,
    assets: &AssetManager,
    physics: Option<&mut dyn PhysicsBackend>,
    entity_map: Option<&mut PhysicsEntityMap>,
    chunk_manager: &mut ChunkManager,
    coord: ChunkCoord,
) -> HashMap<u32, Entity> {
    let id_map = load_scene(scene, world, assets, physics, entity_map);

    // Tag each spawned entity with ChunkMembership and register in the chunk.
    for (&_scene_id, &entity) in &id_map {
        let _ = world.insert_one(entity, ChunkMembership(coord));
        chunk_manager.register_entity(entity, coord);
    }

    id_map
}

/// File path convention for chunk files.
pub fn chunk_file_path(base_dir: &Path, coord: ChunkCoord) -> PathBuf {
    base_dir.join(format!("chunk_{}_{}.scene.ron", coord.x, coord.y))
}
