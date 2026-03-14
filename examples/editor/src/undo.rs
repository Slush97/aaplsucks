use esox_engine::esox_gfx::mesh3d::{MaterialDescriptor, MaterialHandle, MeshHandle};
use esox_engine::glam::{Quat, Vec3};
use esox_engine::hecs::{self, Entity};
use esox_engine::{
    Camera3D, DirectionalLightComponent, GlobalTransform, MeshRenderer, PointLightComponent,
    SpotLightComponent, Tag, Transform3D,
};

use crate::ComponentKind;

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
    pub camera: Option<Camera3D>,
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
            camera: world.get::<&Camera3D>(entity).ok().map(|c| *c),
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
        if let Some(cam) = self.camera {
            world.insert_one(entity, cam).unwrap();
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
    EditPointLightColor {
        entity: Entity,
        old: [f32; 3],
        new: [f32; 3],
    },
    EditSpotLightColor {
        entity: Entity,
        old: [f32; 3],
        new: [f32; 3],
    },
    EditSpotLightInnerCone {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditSpotLightOuterCone {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditDirLightColor {
        entity: Entity,
        old: [f32; 3],
        new: [f32; 3],
    },
    EditTag {
        entity: Entity,
        old: String,
        new: String,
    },
    EditCameraFov {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditCameraNear {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditCameraFar {
        entity: Entity,
        old: f32,
        new: f32,
    },
    EditMeshVisible {
        entity: Entity,
        old: bool,
        new: bool,
    },
    GizmoDrag {
        entity: Entity,
        old_pos: Vec3,
        new_pos: Vec3,
    },
    GizmoScale {
        entity: Entity,
        old_scale: Vec3,
        new_scale: Vec3,
    },
    GizmoRotate {
        entity: Entity,
        old_rot: Quat,
        new_rot: Quat,
    },
    EditMeshTint {
        entity: Entity,
        old: [f32; 4],
        new: [f32; 4],
    },
    EditPointLightShadows {
        entity: Entity,
        old: bool,
        new: bool,
    },
    EditSpotLightShadows {
        entity: Entity,
        old: bool,
        new: bool,
    },
    AddComponent {
        entity: Entity,
        kind: ComponentKind,
        snapshot: ComponentSnapshot,
    },
    RemoveComponent {
        entity: Entity,
        kind: ComponentKind,
        snapshot: ComponentSnapshot,
    },
    EditMesh {
        entity: Entity,
        old: MeshHandle,
        new: MeshHandle,
    },
    EditMaterial {
        entity: Entity,
        old: MaterialHandle,
        new: MaterialHandle,
    },
    EditMaterialDescriptor {
        entity: Entity,
        handle: MaterialHandle,
        old: MaterialDescriptor,
        new: MaterialDescriptor,
    },
}

/// Snapshot of a single removed component for undo restoration.
pub enum ComponentSnapshot {
    PointLight(PointLightComponent),
    SpotLight(SpotLightComponent),
    DirLight(DirectionalLightComponent),
    Camera(Camera3D),
    MeshRenderer(MeshRenderer),
}

pub struct UndoResult {
    pub select: Option<Option<Entity>>,
    pub material_update: Option<(MaterialHandle, MaterialDescriptor)>,
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
        (
            UndoAction::EditPointLightColor { entity: e1, new: n, .. },
            UndoAction::EditPointLightColor { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditSpotLightColor { entity: e1, new: n, .. },
            UndoAction::EditSpotLightColor { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditSpotLightInnerCone { entity: e1, new: n, .. },
            UndoAction::EditSpotLightInnerCone { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditSpotLightOuterCone { entity: e1, new: n, .. },
            UndoAction::EditSpotLightOuterCone { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditDirLightColor { entity: e1, new: n, .. },
            UndoAction::EditDirLightColor { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditTag { entity: e1, new: n, .. },
            UndoAction::EditTag { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = n2.clone(); true }
        (
            UndoAction::EditCameraFov { entity: e1, new: n, .. },
            UndoAction::EditCameraFov { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditCameraNear { entity: e1, new: n, .. },
            UndoAction::EditCameraNear { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditCameraFar { entity: e1, new: n, .. },
            UndoAction::EditCameraFar { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::GizmoDrag { entity: e1, new_pos: n, .. },
            UndoAction::GizmoDrag { entity: e2, new_pos: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::GizmoScale { entity: e1, new_scale: n, .. },
            UndoAction::GizmoScale { entity: e2, new_scale: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::GizmoRotate { entity: e1, new_rot: n, .. },
            UndoAction::GizmoRotate { entity: e2, new_rot: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditMeshTint { entity: e1, new: n, .. },
            UndoAction::EditMeshTint { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditMesh { entity: e1, new: n, .. },
            UndoAction::EditMesh { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditMaterial { entity: e1, new: n, .. },
            UndoAction::EditMaterial { entity: e2, new: n2, .. },
        ) if *e1 == *e2 => { *n = *n2; true }
        (
            UndoAction::EditMaterialDescriptor { entity: e1, handle: h1, new: n, .. },
            UndoAction::EditMaterialDescriptor { entity: e2, handle: h2, new: n2, .. },
        ) if *e1 == *e2 && *h1 == *h2 => { *n = n2.clone(); true }
        // Bool toggles and add/remove don't merge
        _ => false,
    }
}

/// Apply the inverse of an action. Returns (result, forward_action_for_redo).
fn apply_inverse(action: UndoAction, world: &mut hecs::World) -> (UndoResult, UndoAction) {
    let no_select = UndoResult { select: None, material_update: None };
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
                    material_update: None,
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
                    material_update: None,
                },
                UndoAction::DeleteEntity {
                    stale_entity: new_entity,
                    snapshot: new_snapshot,
                },
            )
        }
        UndoAction::EditPointLightColor { entity, old, new } => {
            if let Ok(mut pl) = world.get::<&mut PointLightComponent>(entity) { pl.color = old; }
            (no_select, UndoAction::EditPointLightColor { entity, old, new })
        }
        UndoAction::EditSpotLightColor { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) { sl.color = old; }
            (no_select, UndoAction::EditSpotLightColor { entity, old, new })
        }
        UndoAction::EditSpotLightInnerCone { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) { sl.inner_cone_angle = old; }
            (no_select, UndoAction::EditSpotLightInnerCone { entity, old, new })
        }
        UndoAction::EditSpotLightOuterCone { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) { sl.outer_cone_angle = old; }
            (no_select, UndoAction::EditSpotLightOuterCone { entity, old, new })
        }
        UndoAction::EditDirLightColor { entity, old, new } => {
            if let Ok(mut dl) = world.get::<&mut DirectionalLightComponent>(entity) { dl.color = old; }
            (no_select, UndoAction::EditDirLightColor { entity, old, new })
        }
        UndoAction::EditTag { entity, old, new } => {
            if let Ok(mut tag) = world.get::<&mut Tag>(entity) { tag.0 = old.clone(); }
            (no_select, UndoAction::EditTag { entity, old, new })
        }
        UndoAction::EditCameraFov { entity, old, new } => {
            if let Ok(mut cam) = world.get::<&mut Camera3D>(entity) { cam.fov_y = old; }
            (no_select, UndoAction::EditCameraFov { entity, old, new })
        }
        UndoAction::EditCameraNear { entity, old, new } => {
            if let Ok(mut cam) = world.get::<&mut Camera3D>(entity) { cam.near = old; }
            (no_select, UndoAction::EditCameraNear { entity, old, new })
        }
        UndoAction::EditCameraFar { entity, old, new } => {
            if let Ok(mut cam) = world.get::<&mut Camera3D>(entity) { cam.far = old; }
            (no_select, UndoAction::EditCameraFar { entity, old, new })
        }
        UndoAction::EditMeshVisible { entity, old, new } => {
            if let Ok(mut mr) = world.get::<&mut MeshRenderer>(entity) { mr.visible = old; }
            (no_select, UndoAction::EditMeshVisible { entity, old, new })
        }
        UndoAction::EditMeshTint { entity, old, new } => {
            if let Ok(mut mr) = world.get::<&mut MeshRenderer>(entity) { mr.tint = old; }
            (no_select, UndoAction::EditMeshTint { entity, old, new })
        }
        UndoAction::EditPointLightShadows { entity, old, new } => {
            if let Ok(mut pl) = world.get::<&mut PointLightComponent>(entity) { pl.cast_shadows = old; }
            (no_select, UndoAction::EditPointLightShadows { entity, old, new })
        }
        UndoAction::EditSpotLightShadows { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) { sl.cast_shadows = old; }
            (no_select, UndoAction::EditSpotLightShadows { entity, old, new })
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
        UndoAction::GizmoScale {
            entity,
            old_scale,
            new_scale,
        } => {
            if let Ok(mut t) = world.get::<&mut Transform3D>(entity) {
                t.scale = old_scale;
            }
            (
                no_select,
                UndoAction::GizmoScale {
                    entity,
                    old_scale,
                    new_scale,
                },
            )
        }
        UndoAction::GizmoRotate {
            entity,
            old_rot,
            new_rot,
        } => {
            if let Ok(mut t) = world.get::<&mut Transform3D>(entity) {
                t.rotation = old_rot;
            }
            (
                no_select,
                UndoAction::GizmoRotate {
                    entity,
                    old_rot,
                    new_rot,
                },
            )
        }
        UndoAction::AddComponent { entity, kind, snapshot: _ } => {
            // Undo add = remove the component (snapshot current state first)
            let cur_snapshot = snapshot_component(world, entity, &kind);
            remove_component(world, entity, &kind);
            (
                no_select,
                UndoAction::AddComponent { entity, kind, snapshot: cur_snapshot },
            )
        }
        UndoAction::RemoveComponent { entity, kind, snapshot } => {
            // Undo remove = re-add the component from snapshot
            restore_component(world, entity, snapshot);
            let cur_snapshot = snapshot_component(world, entity, &kind);
            (
                no_select,
                UndoAction::RemoveComponent {
                    entity,
                    kind,
                    snapshot: cur_snapshot,
                },
            )
        }
        UndoAction::EditMesh { entity, old, new } => {
            if let Ok(mut mr) = world.get::<&mut MeshRenderer>(entity) { mr.mesh = old; }
            (no_select, UndoAction::EditMesh { entity, old, new })
        }
        UndoAction::EditMaterial { entity, old, new } => {
            if let Ok(mut mr) = world.get::<&mut MeshRenderer>(entity) { mr.material = old; }
            (no_select, UndoAction::EditMaterial { entity, old, new })
        }
        UndoAction::EditMaterialDescriptor { entity, handle, old, new } => {
            let result = UndoResult {
                select: None,
                material_update: Some((handle, old.clone())),
            };
            (result, UndoAction::EditMaterialDescriptor { entity, handle, old, new })
        }
    }
}

fn snapshot_component(world: &hecs::World, entity: Entity, kind: &ComponentKind) -> ComponentSnapshot {
    match kind {
        ComponentKind::PointLight => ComponentSnapshot::PointLight(
            world.get::<&PointLightComponent>(entity).map(|c| *c).unwrap_or(PointLightComponent {
                color: [1.0, 0.9, 0.8], intensity: 10.0, range: 15.0, cast_shadows: false,
            }),
        ),
        ComponentKind::SpotLight => ComponentSnapshot::SpotLight(
            world.get::<&SpotLightComponent>(entity).map(|c| *c).unwrap_or(SpotLightComponent {
                color: [1.0, 1.0, 1.0], intensity: 20.0, range: 20.0,
                inner_cone_angle: 0.3, outer_cone_angle: 0.5, cast_shadows: false,
            }),
        ),
        ComponentKind::DirLight => ComponentSnapshot::DirLight(
            world.get::<&DirectionalLightComponent>(entity).map(|c| *c).unwrap_or(DirectionalLightComponent {
                color: [1.0, 1.0, 1.0], intensity: 2.0,
            }),
        ),
        ComponentKind::Camera => ComponentSnapshot::Camera(
            world.get::<&Camera3D>(entity).map(|c| *c).unwrap_or_default(),
        ),
        ComponentKind::MeshRenderer => ComponentSnapshot::MeshRenderer(
            *world.get::<&MeshRenderer>(entity).expect("MeshRenderer must exist to snapshot"),
        ),
    }
}

fn remove_component(world: &mut hecs::World, entity: Entity, kind: &ComponentKind) {
    match kind {
        ComponentKind::PointLight => { let _ = world.remove_one::<PointLightComponent>(entity); }
        ComponentKind::SpotLight => { let _ = world.remove_one::<SpotLightComponent>(entity); }
        ComponentKind::DirLight => { let _ = world.remove_one::<DirectionalLightComponent>(entity); }
        ComponentKind::Camera => { let _ = world.remove_one::<Camera3D>(entity); }
        ComponentKind::MeshRenderer => { let _ = world.remove_one::<MeshRenderer>(entity); }
    }
}

fn restore_component(world: &mut hecs::World, entity: Entity, snapshot: ComponentSnapshot) {
    match snapshot {
        ComponentSnapshot::PointLight(c) => { let _ = world.insert_one(entity, c); }
        ComponentSnapshot::SpotLight(c) => { let _ = world.insert_one(entity, c); }
        ComponentSnapshot::DirLight(c) => { let _ = world.insert_one(entity, c); }
        ComponentSnapshot::Camera(c) => { let _ = world.insert_one(entity, c); }
        ComponentSnapshot::MeshRenderer(c) => { let _ = world.insert_one(entity, c); }
    }
}

/// Apply an action forward. Returns (result, same_action_for_undo).
fn apply_forward(action: UndoAction, world: &mut hecs::World) -> (UndoResult, UndoAction) {
    let no_select = UndoResult { select: None, material_update: None };
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
                    material_update: None,
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
                    material_update: None,
                },
                UndoAction::DeleteEntity {
                    stale_entity,
                    snapshot: new_snapshot,
                },
            )
        }
        UndoAction::EditPointLightColor { entity, old, new } => {
            if let Ok(mut pl) = world.get::<&mut PointLightComponent>(entity) { pl.color = new; }
            (no_select, UndoAction::EditPointLightColor { entity, old, new })
        }
        UndoAction::EditSpotLightColor { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) { sl.color = new; }
            (no_select, UndoAction::EditSpotLightColor { entity, old, new })
        }
        UndoAction::EditSpotLightInnerCone { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) { sl.inner_cone_angle = new; }
            (no_select, UndoAction::EditSpotLightInnerCone { entity, old, new })
        }
        UndoAction::EditSpotLightOuterCone { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) { sl.outer_cone_angle = new; }
            (no_select, UndoAction::EditSpotLightOuterCone { entity, old, new })
        }
        UndoAction::EditDirLightColor { entity, old, new } => {
            if let Ok(mut dl) = world.get::<&mut DirectionalLightComponent>(entity) { dl.color = new; }
            (no_select, UndoAction::EditDirLightColor { entity, old, new })
        }
        UndoAction::EditTag { entity, old, new } => {
            if let Ok(mut tag) = world.get::<&mut Tag>(entity) { tag.0 = new.clone(); }
            (no_select, UndoAction::EditTag { entity, old, new })
        }
        UndoAction::EditCameraFov { entity, old, new } => {
            if let Ok(mut cam) = world.get::<&mut Camera3D>(entity) { cam.fov_y = new; }
            (no_select, UndoAction::EditCameraFov { entity, old, new })
        }
        UndoAction::EditCameraNear { entity, old, new } => {
            if let Ok(mut cam) = world.get::<&mut Camera3D>(entity) { cam.near = new; }
            (no_select, UndoAction::EditCameraNear { entity, old, new })
        }
        UndoAction::EditCameraFar { entity, old, new } => {
            if let Ok(mut cam) = world.get::<&mut Camera3D>(entity) { cam.far = new; }
            (no_select, UndoAction::EditCameraFar { entity, old, new })
        }
        UndoAction::EditMeshVisible { entity, old, new } => {
            if let Ok(mut mr) = world.get::<&mut MeshRenderer>(entity) { mr.visible = new; }
            (no_select, UndoAction::EditMeshVisible { entity, old, new })
        }
        UndoAction::EditMeshTint { entity, old, new } => {
            if let Ok(mut mr) = world.get::<&mut MeshRenderer>(entity) { mr.tint = new; }
            (no_select, UndoAction::EditMeshTint { entity, old, new })
        }
        UndoAction::EditPointLightShadows { entity, old, new } => {
            if let Ok(mut pl) = world.get::<&mut PointLightComponent>(entity) { pl.cast_shadows = new; }
            (no_select, UndoAction::EditPointLightShadows { entity, old, new })
        }
        UndoAction::EditSpotLightShadows { entity, old, new } => {
            if let Ok(mut sl) = world.get::<&mut SpotLightComponent>(entity) { sl.cast_shadows = new; }
            (no_select, UndoAction::EditSpotLightShadows { entity, old, new })
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
        UndoAction::GizmoScale {
            entity,
            old_scale,
            new_scale,
        } => {
            if let Ok(mut t) = world.get::<&mut Transform3D>(entity) {
                t.scale = new_scale;
            }
            (
                no_select,
                UndoAction::GizmoScale {
                    entity,
                    old_scale,
                    new_scale,
                },
            )
        }
        UndoAction::GizmoRotate {
            entity,
            old_rot,
            new_rot,
        } => {
            if let Ok(mut t) = world.get::<&mut Transform3D>(entity) {
                t.rotation = new_rot;
            }
            (
                no_select,
                UndoAction::GizmoRotate {
                    entity,
                    old_rot,
                    new_rot,
                },
            )
        }
        UndoAction::AddComponent { entity, kind, snapshot } => {
            // Redo add = re-add from snapshot
            restore_component(world, entity, snapshot);
            let cur_snapshot = snapshot_component(world, entity, &kind);
            (
                no_select,
                UndoAction::AddComponent { entity, kind, snapshot: cur_snapshot },
            )
        }
        UndoAction::RemoveComponent { entity, kind, snapshot: _ } => {
            // Redo remove = remove again (snapshot current first)
            let cur_snapshot = snapshot_component(world, entity, &kind);
            remove_component(world, entity, &kind);
            (
                no_select,
                UndoAction::RemoveComponent { entity, kind, snapshot: cur_snapshot },
            )
        }
        UndoAction::EditMesh { entity, old, new } => {
            if let Ok(mut mr) = world.get::<&mut MeshRenderer>(entity) { mr.mesh = new; }
            (no_select, UndoAction::EditMesh { entity, old, new })
        }
        UndoAction::EditMaterial { entity, old, new } => {
            if let Ok(mut mr) = world.get::<&mut MeshRenderer>(entity) { mr.material = new; }
            (no_select, UndoAction::EditMaterial { entity, old, new })
        }
        UndoAction::EditMaterialDescriptor { entity, handle, old, new } => {
            let result = UndoResult {
                select: None,
                material_update: Some((handle, new.clone())),
            };
            (result, UndoAction::EditMaterialDescriptor { entity, handle, old, new })
        }
    }
}
