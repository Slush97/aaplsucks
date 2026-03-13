//! Bidirectional mapping between physics body handles and ECS entities.

use std::collections::HashMap;

use super::{BodyHandle, ContactEvent, TriggerEvent};

/// Bidirectional map between physics `BodyHandle`s and `hecs::Entity` values.
pub struct PhysicsEntityMap {
    handle_to_entity: HashMap<BodyHandle, hecs::Entity>,
    entity_to_handle: HashMap<hecs::Entity, BodyHandle>,
}

impl PhysicsEntityMap {
    pub fn new() -> Self {
        Self {
            handle_to_entity: HashMap::new(),
            entity_to_handle: HashMap::new(),
        }
    }

    pub fn insert(&mut self, handle: BodyHandle, entity: hecs::Entity) {
        self.handle_to_entity.insert(handle, entity);
        self.entity_to_handle.insert(entity, handle);
    }

    pub fn remove_by_handle(&mut self, handle: BodyHandle) -> Option<hecs::Entity> {
        if let Some(entity) = self.handle_to_entity.remove(&handle) {
            self.entity_to_handle.remove(&entity);
            Some(entity)
        } else {
            None
        }
    }

    pub fn remove_by_entity(&mut self, entity: hecs::Entity) -> Option<BodyHandle> {
        if let Some(handle) = self.entity_to_handle.remove(&entity) {
            self.handle_to_entity.remove(&handle);
            Some(handle)
        } else {
            None
        }
    }

    pub fn get_entity(&self, handle: BodyHandle) -> Option<hecs::Entity> {
        self.handle_to_entity.get(&handle).copied()
    }

    pub fn get_handle(&self, entity: hecs::Entity) -> Option<BodyHandle> {
        self.entity_to_handle.get(&entity).copied()
    }

    /// Resolve a contact event to the pair of ECS entities involved.
    pub fn resolve_contact(&self, event: &ContactEvent) -> Option<(hecs::Entity, hecs::Entity)> {
        let a = self.get_entity(event.body_a)?;
        let b = self.get_entity(event.body_b)?;
        Some((a, b))
    }

    /// Resolve a trigger event to the pair of ECS entities involved.
    pub fn resolve_trigger(&self, event: &TriggerEvent) -> Option<(hecs::Entity, hecs::Entity)> {
        let a = self.get_entity(event.body_a)?;
        let b = self.get_entity(event.body_b)?;
        Some((a, b))
    }
}
