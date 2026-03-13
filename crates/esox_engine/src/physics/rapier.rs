//! Rapier3d physics backend.

use std::collections::{HashMap, HashSet};

use glam::{Quat, Vec3};
use rapier3d::prelude::*;
use rapier3d::na::UnitQuaternion;

use super::{
    BodyDesc, BodyHandle, BodyType, ColliderDesc, ColliderShape, ContactEvent, PhysicsBackend,
    RayHit, TriggerEvent, TriggerPhase,
};

/// Physics backend powered by rapier3d.
pub struct RapierPhysics {
    pipeline: PhysicsPipeline,
    gravity: Vector<f32>,
    integration_params: IntegrationParameters,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    islands: IslandManager,
    query_pipeline: QueryPipeline,

    /// Map engine handles → rapier handles.
    handle_map: HashMap<BodyHandle, RigidBodyHandle>,
    /// Reverse map: rapier handles → engine handles.
    reverse_map: HashMap<RigidBodyHandle, BodyHandle>,
    next_handle: u32,

    /// Contact events accumulated during the last step.
    contacts: Vec<ContactEvent>,

    /// Trigger (sensor overlap) events accumulated during the last step.
    triggers: Vec<TriggerEvent>,
    /// Currently active sensor overlap pairs (sorted so (min, max)).
    active_overlaps: HashSet<(BodyHandle, BodyHandle)>,
}

impl RapierPhysics {
    /// Create a new rapier physics world with the given gravity.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            gravity: vector![gravity.x, gravity.y, gravity.z],
            integration_params: IntegrationParameters::default(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            islands: IslandManager::new(),
            query_pipeline: QueryPipeline::new(),
            handle_map: HashMap::new(),
            reverse_map: HashMap::new(),
            next_handle: 1,
            contacts: Vec::new(),
            triggers: Vec::new(),
            active_overlaps: HashSet::new(),
        }
    }

    fn alloc_handle(&mut self) -> BodyHandle {
        let h = BodyHandle(self.next_handle);
        self.next_handle += 1;
        h
    }

    fn build_collider(desc: &ColliderDesc) -> Collider {
        let shape: SharedShape = match desc.shape {
            ColliderShape::Box { half_extents } => SharedShape::cuboid(
                half_extents.x,
                half_extents.y,
                half_extents.z,
            ),
            ColliderShape::Sphere { radius } => SharedShape::ball(radius),
            ColliderShape::Capsule {
                half_height,
                radius,
            } => SharedShape::capsule_y(half_height, radius),
        };

        let mut builder = ColliderBuilder::new(shape)
            .friction(desc.friction)
            .restitution(desc.restitution);

        if desc.is_sensor {
            builder = builder
                .sensor(true)
                .active_collision_types(
                    ActiveCollisionTypes::default()
                        | ActiveCollisionTypes::KINEMATIC_FIXED
                        | ActiveCollisionTypes::KINEMATIC_KINEMATIC,
                );
        }

        builder.build()
    }
}

impl PhysicsBackend for RapierPhysics {
    fn step(&mut self, dt: f32) {
        self.integration_params.dt = dt;

        self.pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );

        // Collect contact events.
        self.contacts.clear();
        for pair in self.narrow_phase.contact_pairs() {
            if !pair.has_any_active_contact {
                continue;
            }

            let body_a = self
                .colliders
                .get(pair.collider1)
                .and_then(|c| c.parent())
                .and_then(|rb| self.reverse_map.get(&rb));
            let body_b = self
                .colliders
                .get(pair.collider2)
                .and_then(|c| c.parent())
                .and_then(|rb| self.reverse_map.get(&rb));

            if let (Some(&a), Some(&b)) = (body_a, body_b) {
                // Get the strongest contact normal (world-space) and impulse.
                // local_n1 is in collider1's local frame; rotate by collider1's world rotation.
                let col1_rot = self
                    .colliders
                    .get(pair.collider1)
                    .map(|c| c.position().rotation)
                    .unwrap_or(UnitQuaternion::identity());

                let (normal, impulse) = pair
                    .manifolds
                    .iter()
                    .flat_map(|m| {
                        let local_n = nalgebra::Vector3::new(m.local_n1.x, m.local_n1.y, m.local_n1.z);
                        let world_n = col1_rot * local_n;
                        let n = Vec3::new(world_n.x, world_n.y, world_n.z);
                        m.points.iter().map(move |p| (n, p.data.impulse))
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((Vec3::Y, 0.0));

                self.contacts.push(ContactEvent {
                    body_a: a,
                    body_b: b,
                    normal,
                    impulse,
                });
            }
        }

        // Collect trigger (sensor intersection) events.
        self.triggers.clear();
        let mut current_overlaps: HashSet<(BodyHandle, BodyHandle)> = HashSet::new();

        for pair in self.narrow_phase.intersection_pairs() {
            let (collider1, collider2, intersecting) = pair;
            if !intersecting {
                continue;
            }

            let body_a = self
                .colliders
                .get(collider1)
                .and_then(|c| c.parent())
                .and_then(|rb| self.reverse_map.get(&rb));
            let body_b = self
                .colliders
                .get(collider2)
                .and_then(|c| c.parent())
                .and_then(|rb| self.reverse_map.get(&rb));

            if let (Some(&a), Some(&b)) = (body_a, body_b) {
                // Normalize pair ordering for consistent lookup.
                let pair_key = if a.0 <= b.0 { (a, b) } else { (b, a) };
                current_overlaps.insert(pair_key);

                let phase = if self.active_overlaps.contains(&pair_key) {
                    TriggerPhase::Stay
                } else {
                    TriggerPhase::Enter
                };
                self.triggers.push(TriggerEvent {
                    body_a: a,
                    body_b: b,
                    phase,
                });
            }
        }

        // Emit Exit events for pairs that were active last frame but not this frame.
        for &(a, b) in &self.active_overlaps {
            if !current_overlaps.contains(&(a, b)) {
                self.triggers.push(TriggerEvent {
                    body_a: a,
                    body_b: b,
                    phase: TriggerPhase::Exit,
                });
            }
        }

        self.active_overlaps = current_overlaps;
    }

    fn add_body(&mut self, desc: BodyDesc) -> BodyHandle {
        let rb_type = match desc.body_type {
            BodyType::Static => RigidBodyType::Fixed,
            BodyType::Dynamic => RigidBodyType::Dynamic,
            BodyType::Kinematic => RigidBodyType::KinematicPositionBased,
        };

        let iso = Isometry::from_parts(
            Translation::new(desc.position.x, desc.position.y, desc.position.z),
            UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                desc.rotation.w, desc.rotation.x, desc.rotation.y, desc.rotation.z,
            )),
        );
        let rb = RigidBodyBuilder::new(rb_type).position(iso).build();

        let rb_handle = self.bodies.insert(rb);

        // Attach inline collider if provided.
        if let Some(ref collider_desc) = desc.collider {
            let collider = Self::build_collider(collider_desc);
            self.colliders
                .insert_with_parent(collider, rb_handle, &mut self.bodies);
        }

        let engine_handle = self.alloc_handle();
        self.handle_map.insert(engine_handle, rb_handle);
        self.reverse_map.insert(rb_handle, engine_handle);
        engine_handle
    }

    fn remove_body(&mut self, handle: BodyHandle) {
        if let Some(rb_handle) = self.handle_map.remove(&handle) {
            self.reverse_map.remove(&rb_handle);
            self.bodies.remove(
                rb_handle,
                &mut self.islands,
                &mut self.colliders,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
                true,
            );
        }
    }

    fn query_transform(&self, handle: BodyHandle) -> Option<(Vec3, Quat)> {
        let rb_handle = self.handle_map.get(&handle)?;
        let rb = self.bodies.get(*rb_handle)?;
        let pos = rb.position();
        let t = pos.translation;
        let r = pos.rotation;
        Some((
            Vec3::new(t.x, t.y, t.z),
            Quat::from_xyzw(r.i, r.j, r.k, r.w),
        ))
    }

    fn set_transform(&mut self, handle: BodyHandle, pos: Vec3, rot: Quat) {
        if let Some(&rb_handle) = self.handle_map.get(&handle) {
            if let Some(rb) = self.bodies.get_mut(rb_handle) {
                rb.set_position(
                    Isometry::from_parts(
                        Translation::new(pos.x, pos.y, pos.z),
                        UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                            rot.w, rot.x, rot.y, rot.z,
                        )),
                    ),
                    true,
                );
            }
        }
    }

    fn apply_force(&mut self, handle: BodyHandle, force: Vec3) {
        if let Some(&rb_handle) = self.handle_map.get(&handle) {
            if let Some(rb) = self.bodies.get_mut(rb_handle) {
                rb.add_force(vector![force.x, force.y, force.z], true);
            }
        }
    }

    fn apply_impulse(&mut self, handle: BodyHandle, impulse: Vec3) {
        if let Some(&rb_handle) = self.handle_map.get(&handle) {
            if let Some(rb) = self.bodies.get_mut(rb_handle) {
                rb.apply_impulse(vector![impulse.x, impulse.y, impulse.z], true);
            }
        }
    }

    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit> {
        let ray = Ray::new(
            point![origin.x, origin.y, origin.z],
            vector![dir.x, dir.y, dir.z],
        );

        let (collider_handle, toi) = self.query_pipeline.cast_ray(
            &self.bodies,
            &self.colliders,
            &ray,
            max_dist,
            true,
            QueryFilter::default(),
        )?;

        let collider = self.colliders.get(collider_handle)?;
        let hit_point = ray.point_at(toi);
        let normal = collider
            .shape()
            .cast_ray_and_get_normal(
                collider.position(),
                &ray,
                max_dist,
                true,
            )
            .map(|hit| Vec3::new(hit.normal.x, hit.normal.y, hit.normal.z))
            .unwrap_or(Vec3::Y);

        let body_handle = collider
            .parent()
            .and_then(|rb| self.reverse_map.get(&rb))
            .copied()
            .unwrap_or(BodyHandle(0));

        Some(RayHit {
            point: Vec3::new(hit_point.x, hit_point.y, hit_point.z),
            normal,
            distance: toi,
            body: body_handle,
        })
    }

    fn drain_contacts(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.contacts)
    }

    fn drain_triggers(&mut self) -> Vec<TriggerEvent> {
        std::mem::take(&mut self.triggers)
    }

    fn set_gravity(&mut self, gravity: Vec3) {
        self.gravity = vector![gravity.x, gravity.y, gravity.z];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_query_body() {
        let mut physics = RapierPhysics::new(Vec3::new(0.0, -9.81, 0.0));
        let handle = physics.add_body(BodyDesc {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
            body_type: BodyType::Static,
            collider: None,
        });
        let (pos, _rot) = physics.query_transform(handle).unwrap();
        assert!((pos.x - 1.0).abs() < 1e-5);
        assert!((pos.y - 2.0).abs() < 1e-5);
        assert!((pos.z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn dynamic_body_falls_under_gravity() {
        let mut physics = RapierPhysics::new(Vec3::new(0.0, -9.81, 0.0));
        let handle = physics.add_body(BodyDesc {
            position: Vec3::new(0.0, 10.0, 0.0),
            rotation: Quat::IDENTITY,
            body_type: BodyType::Dynamic,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Sphere { radius: 0.5 },
                ..Default::default()
            }),
        });

        // Step a few times.
        for _ in 0..60 {
            physics.step(1.0 / 60.0);
        }

        let (pos, _) = physics.query_transform(handle).unwrap();
        // Should have fallen significantly.
        assert!(pos.y < 10.0, "body should have fallen: y={}", pos.y);
    }

    #[test]
    fn collision_produces_contacts() {
        let mut physics = RapierPhysics::new(Vec3::new(0.0, -9.81, 0.0));

        // Static floor.
        physics.add_body(BodyDesc {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            body_type: BodyType::Static,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Box {
                    half_extents: Vec3::new(50.0, 0.5, 50.0),
                },
                ..Default::default()
            }),
        });

        // Dynamic ball above floor.
        physics.add_body(BodyDesc {
            position: Vec3::new(0.0, 2.0, 0.0),
            rotation: Quat::IDENTITY,
            body_type: BodyType::Dynamic,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Sphere { radius: 0.5 },
                ..Default::default()
            }),
        });

        // Step until ball hits the floor.
        let mut had_contacts = false;
        for _ in 0..120 {
            physics.step(1.0 / 60.0);
            let contacts = physics.drain_contacts();
            if !contacts.is_empty() {
                had_contacts = true;
            }
        }
        assert!(had_contacts, "expected contacts between ball and floor");
    }

    #[test]
    fn raycast_hits_body() {
        let mut physics = RapierPhysics::new(Vec3::ZERO);
        physics.add_body(BodyDesc {
            position: Vec3::new(0.0, 0.0, 5.0),
            rotation: Quat::IDENTITY,
            body_type: BodyType::Static,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Sphere { radius: 1.0 },
                ..Default::default()
            }),
        });

        // Need to step once so the query pipeline is updated.
        physics.step(1.0 / 60.0);

        let hit = physics.raycast(Vec3::ZERO, Vec3::Z, 100.0);
        assert!(hit.is_some(), "raycast should hit the sphere");
        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.1, "distance should be ~4.0, got {}", hit.distance);
    }

    #[test]
    fn sensor_overlap_produces_enter_stay_exit() {
        let mut physics = RapierPhysics::new(Vec3::new(0.0, -9.81, 0.0));

        // Static sensor volume.
        physics.add_body(BodyDesc {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            body_type: BodyType::Static,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Box {
                    half_extents: Vec3::new(2.0, 2.0, 2.0),
                },
                is_sensor: true,
                ..Default::default()
            }),
        });

        // Dynamic ball that will fall into the sensor.
        physics.add_body(BodyDesc {
            position: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::IDENTITY,
            body_type: BodyType::Dynamic,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Sphere { radius: 0.5 },
                ..Default::default()
            }),
        });

        let mut saw_enter = false;
        let mut saw_stay = false;
        let mut saw_exit = false;

        // Step until we see all three phases.
        for _ in 0..300 {
            physics.step(1.0 / 60.0);
            let triggers = physics.drain_triggers();
            for t in &triggers {
                match t.phase {
                    TriggerPhase::Enter => saw_enter = true,
                    TriggerPhase::Stay => saw_stay = true,
                    TriggerPhase::Exit => saw_exit = true,
                }
            }
        }

        assert!(saw_enter, "expected Enter trigger event");
        assert!(saw_stay, "expected Stay trigger event");
        assert!(saw_exit, "expected Exit trigger event");
    }

    #[test]
    fn no_overlap_means_no_triggers() {
        let mut physics = RapierPhysics::new(Vec3::ZERO);

        // Two sensors far apart.
        physics.add_body(BodyDesc {
            position: Vec3::new(-100.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            body_type: BodyType::Static,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Sphere { radius: 1.0 },
                is_sensor: true,
                ..Default::default()
            }),
        });

        physics.add_body(BodyDesc {
            position: Vec3::new(100.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            body_type: BodyType::Static,
            collider: Some(ColliderDesc {
                shape: ColliderShape::Sphere { radius: 1.0 },
                is_sensor: true,
                ..Default::default()
            }),
        });

        for _ in 0..60 {
            physics.step(1.0 / 60.0);
            let triggers = physics.drain_triggers();
            assert!(triggers.is_empty(), "expected no trigger events");
        }
    }

    #[test]
    fn remove_body_works() {
        let mut physics = RapierPhysics::new(Vec3::ZERO);
        let handle = physics.add_body(BodyDesc {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            body_type: BodyType::Static,
            collider: None,
        });
        assert!(physics.query_transform(handle).is_some());
        physics.remove_body(handle);
        assert!(physics.query_transform(handle).is_none());
    }
}
