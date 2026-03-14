use esox_engine::glam::Vec3;
use esox_engine::hecs::{self, Entity};
use esox_engine::{
    DirectionalLightComponent, GlobalTransform, MeshRenderer, PointLightComponent,
    SpotLightComponent, Tag, Transform3D,
};

const MAX_UNDO: usize = 100;
const MERGE_WINDOW: f32 = 0.5;

/// Snapshot of all editor-relevant components on an entity.
pub struct EntitySnapshot {
    pub transform: Transform3D,
    pub tag: Option<String>,
    pub mesh_renderer: Option<MeshRenderer>,
    pub point_light: Option<PointLightComponent>,
    pub spot_light: Option<SpotLightComponent>,
    pub dir_light: Option<DirectionalLightComponent>,
}

impl EntitySnapshot {
    pub fn capture(world: &hecs::World, entity: Entity) -> Self {
        Self {
            transform: world
                .get::<&Transform3D>(entity)
                .map(|t| *t)
                .unwrap_or_default(),
            tag: world.get::<&Tag>(entity).ok().map(|t| t.0.clone()),
            mesh_renderer: world.get::<&MeshRenderer>(entity).ok().map(|m| *m),
            point_light: world
                .get::<&PointLightComponent>(entity)
                .ok()
                .map(|p| *p),
            spot_light: world
                .get::<&SpotLightComponent>(entity)
                .ok()
                .map(|s| *s),
            dir_light: world
                .get::<&DirectionalLightComponent>(entity)
                .ok()
                .map(|d| *d),
        }
    }

    pub fn respawn(self, world: &mut hecs::World) -> Entity {
        // Build entity with transform + global transform (always present)
        let entity = world.reserve_entity();
        world
            .insert(entity, (self.transform, GlobalTransform::default()))
            .unwrap();

        if let Some(tag) = self.tag {
            world.insert_one(entity, Tag(tag)).unwrap();
        }
        if let Some(mr) = self.mesh_renderer {
            world.insert_one(entity, mr).unwrap();
        }
        if let Some(pl) = self.point_light {
            world.insert_one(entity, pl).unwrap();
        }
        if let Some(sl) = self.spot_light {
            world.insert_one(entity, sl).unwrap();
        }
        if let Some(dl) = self.dir_light {
            world.insert_one(entity, dl).unwrap();
        }
        entity
    }
}

pub enum UndoAction {
    EditTransform {
        entity: Entity,
        old: Transform3D,
        new: Transform3D,
    },
    EditPointLightIntensity {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditPointLightRange {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditSpotLightIntensity {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditSpotLightRange {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditDirLightIntensity {
        entity: Entity,
        old: f32,
        new: f32,
    },
    SpawnEntity {
        entity: Entity,
        snapshot: EntitySnapshot,
    },
    DeleteEntity {
        stale_entity: Entity,
        snapshot: EntitySnapshot,
    },
    GizmoDrag {
        entity: Entity,
        old_pos: Vec3,
        new_pos: Vec3,
    },
}

pub struct UndoResult {
    pub select: Option<Option<Entity>>,
}

pub struct UndoStack {
    undo: Vec<(UndoAction, f32)>, // (action, timestamp)
    redo: Vec<UndoAction>,
}

impl UndoStack {
    pub fn new() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    pub fn push(&mut self, action: UndoAction) {
        self.redo.clear();
        self.undo.push((action, 0.0));
        if self.undo.len() > MAX_UNDO {
            self.undo.remove(0);
        }
    }

    pub fn push_or_merge(&mut self, action: UndoAction, time: f32) {
        // Try to merge with top of undo stack if same type+entity and within merge window
        if let Some((top, top_time)) = self.undo.last_mut() {
            if (time - *top_time) < MERGE_WINDOW {
                if try_merge(top, &action) {
                    *top_time = time;
                    self.redo.clear();
                    return;
                }
            }
        }
        self.redo.clear();
        self.undo.push((action, time));
        if self.undo.len() > MAX_UNDO {
            self.undo.remove(0);
        }
    }

    pub fn undo(&mut self, world: &mut hecs::World) -> Option<UndoResult> {
        let (action, _time) = self.undo.pop()?;
        let (result, forward_action) = apply_inverse(action, world);
        self.redo.push(forward_action);
        Some(result)
    }

    pub fn redo(&mut self, world: &mut hecs::World) -> Option<UndoResult> {
        let action = self.redo.pop()?;
        let (result, inverse_action) = apply_forward(action, world);
        self.undo.push((inverse_action, 0.0));
        Some(result)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

/// Try to merge `incoming` into `top`. Returns true if merged.
fn try_merge(top: &mut UndoAction, incoming: &UndoAction) -> bool {
    match (top, incoming) {
        (
            UndoAction::EditTransform {
                entity: e1,
                new: n,
                ..
            },
            UndoAction::EditTransform {
                entity: e2, new: n2, ..
            },
        ) if *e1 == *e2 => {
            *n = *n2;
            true
        }
        (
            UndoAction::EditPointLightIntensity {
                entity: e1,
                new: n,
                ..
            },
            UndoAction::EditPointLightIntensity {
                entity: e2, new: n2, ..
            },
        ) if *e1 == *e2 => {
            *n = *n2;
            true
        }
        (
            UndoAction::EditPointLightRange {
                entity: e1,
                new: n,
                ..
            },
            UndoAction::EditPointLightRange {
                entity: e2, new: n2, ..
            },
        ) if *e1 == *e2 => {
            *n = *n2;
            true
        }
        (
            UndoAction::EditSpotLightIntensity {
                entity: e1,
                new: n,
                ..
            },
            UndoAction::EditSpotLightIntensity {
                entity: e2, new: n2, ..
            },
        ) if *e1 == *e2 => {
            *n = *n2;
            true
        }
        (
            UndoAction::EditSpotLightRange {
                entity: e1,
                new: n,
                ..
            },
            UndoAction::EditSpotLightRange {
                entity: e2, new: n2, ..
            },
        ) if *e1 == *e2 => {
            *n = *n2;
            true
        }
        (
            UndoAction::EditDirLightIntensity {
                entity: e1,
                new: n,
                ..
            },
            UndoAction::EditDirLightIntensity {
                entity: e2, new: n2, ..
            },
        ) if *e1 == *e2 => {
            *n = *n2;
            true
        }
        _ => false,
    }
}

/// Apply the inverse of an action. Returns (result, forward_action_for_redo).
fn apply_inverse(action: UndoAction, world: &mut hecs::World) -> (UndoResult, UndoAction) {
    let no_select = UndoResult { select: None };
    match action {
        UndoAction::EditTransform { entity, old, new } => {
            if let Ok(mut t) = world.get::<&mut Transform3D>(entity) {
                *t = old;
            }
            (no_select, UndoAction::EditTransform { entity, old, new })
        }
        UndoAction::EditPointLightIntensity { entity, old, new } => {
            if let Ok(mut pl) = world.get::<&mut PointLightComponent>(entity) {
                pl.intensity = old;
            }
            (
                no_select,
                UndoAction::EditPointLightIntensity { entity, old, new },
            )
        }
        UndoAction::EditPointLightRange { entity, old, new } => {
            if let Ok(mut pl) = world.get::<&mut PointLightComponent>(entity) {
                pl.range = old;
            }
            (
                no_select,
                UndoAction::EditPointLightRange { entity, old, new },
            )
        }
        UndoAction::EditSpotLightIntensity { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) {
                sl.intensity = old;
            }
            (
                no_select,
                UndoAction::EditSpotLightIntensity { entity, old, new },
            )
        }
        UndoAction::EditSpotLightRange { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) {
                sl.range = old;
            }
            (
                no_select,
                UndoAction::EditSpotLightRange { entity, old, new },
            )
        }
        UndoAction::EditDirLightIntensity { entity, old, new } => {
            if let Ok(mut dl) = world.get::<&mut DirectionalLightComponent>(entity) {
                dl.intensity = old;
            }
            (
                no_select,
                UndoAction::EditDirLightIntensity { entity, old, new },
            )
        }
        UndoAction::SpawnEntity { entity, snapshot: _ } => {
            // Undo spawn = despawn
            let new_snapshot = EntitySnapshot::capture(world, entity);
            let _ = world.despawn(entity);
            (
                UndoResult {
                    select: Some(None),
                },
                UndoAction::SpawnEntity {
                    entity,
                    snapshot: new_snapshot,
                },
            )
        }
        UndoAction::DeleteEntity {
            stale_entity: _,
            snapshot,
        } => {
            // Undo delete = respawn
            let new_entity = snapshot.respawn(world);
            let new_snapshot = EntitySnapshot::capture(world, new_entity);
            (
                UndoResult {
                    select: Some(Some(new_entity)),
                },
                UndoAction::DeleteEntity {
                    stale_entity: new_entity,
                    snapshot: new_snapshot,
                },
            )
        }
        UndoAction::GizmoDrag {
            entity,
            old_pos,
            new_pos,
        } => {
            if let Ok(mut t) = world.get::<&mut Transform3D>(entity) {
                t.position = old_pos;
            }
            (
                no_select,
                UndoAction::GizmoDrag {
                    entity,
                    old_pos,
                    new_pos,
                },
            )
        }
    }
}

/// Apply an action forward. Returns (result, same_action_for_undo).
fn apply_forward(action: UndoAction, world: &mut hecs::World) -> (UndoResult, UndoAction) {
    let no_select = UndoResult { select: None };
    match action {
        UndoAction::EditTransform { entity, old, new } => {
            if let Ok(mut t) = world.get::<&mut Transform3D>(entity) {
                *t = new;
            }
            (no_select, UndoAction::EditTransform { entity, old, new })
        }
        UndoAction::EditPointLightIntensity { entity, old, new } => {
            if let Ok(mut pl) = world.get::<&mut PointLightComponent>(entity) {
                pl.intensity = new;
            }
            (
                no_select,
                UndoAction::EditPointLightIntensity { entity, old, new },
            )
        }
        UndoAction::EditPointLightRange { entity, old, new } => {
            if let Ok(mut pl) = world.get::<&mut PointLightComponent>(entity) {
                pl.range = new;
            }
            (
                no_select,
                UndoAction::EditPointLightRange { entity, old, new },
            )
        }
        UndoAction::EditSpotLightIntensity { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) {
                sl.intensity = new;
            }
            (
                no_select,
                UndoAction::EditSpotLightIntensity { entity, old, new },
            )
        }
        UndoAction::EditSpotLightRange { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) {
                sl.range = new;
            }
            (
                no_select,
                UndoAction::EditSpotLightRange { entity, old, new },
            )
        }
        UndoAction::EditDirLightIntensity { entity, old, new } => {
            if let Ok(mut dl) = world.get::<&mut DirectionalLightComponent>(entity) {
                dl.intensity = new;
            }
            (
                no_select,
                UndoAction::EditDirLightIntensity { entity, old, new },
            )
        }
        UndoAction::SpawnEntity {
            entity: _,
            snapshot,
        } => {
            // Redo spawn = respawn
            let new_entity = snapshot.respawn(world);
            let new_snapshot = EntitySnapshot::capture(world, new_entity);
            (
                UndoResult {
                    select: Some(Some(new_entity)),
                },
                UndoAction::SpawnEntity {
                    entity: new_entity,
                    snapshot: new_snapshot,
                },
            )
        }
        UndoAction::DeleteEntity {
            stale_entity,
            snapshot: _,
        } => {
            // Redo delete = despawn again
            let new_snapshot = EntitySnapshot::capture(world, stale_entity);
            let _ = world.despawn(stale_entity);
            (
                UndoResult {
                    select: Some(None),
                },
                UndoAction::DeleteEntity {
                    stale_entity,
                    snapshot: new_snapshot,
                },
            )
        }
        UndoAction::GizmoDrag {
            entity,
            old_pos,
            new_pos,
        } => {
            if let Ok(mut t) = world.get::<&mut Transform3D>(entity) {
                t.position = new_pos;
            }
            (
                no_select,
                UndoAction::GizmoDrag {
                    entity,
                    old_pos,
                    new_pos,
                },
            )
        }
    }
}
