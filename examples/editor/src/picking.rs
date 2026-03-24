// Re-export core picking utilities from the engine.
pub use esox_engine::picking::{project_ray_to_plane, ray_aabb, screen_to_ray};

use esox_engine::esox_gfx::mesh3d::Aabb;
use esox_engine::glam::Vec3;
use esox_engine::hecs;
use esox_engine::Ctx;

/// Pick the closest entity with a MeshRenderer that the ray hits.
///
/// Thin wrapper around [`esox_engine::picking::pick_entity`] that takes a [`Ctx`].
pub fn pick_entity(
    ctx: &Ctx,
    ray_origin: Vec3,
    ray_dir: Vec3,
    exclude: Option<hecs::Entity>,
) -> Option<(hecs::Entity, f32)> {
    esox_engine::picking::pick_entity(ctx.world, ctx.renderer, ray_origin, ray_dir, exclude)
}

/// Axis for gizmo interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

/// Test the ray against gizmo arrow AABBs at `gizmo_origin` with given `gizmo_scale`.
/// Returns the closest hit axis, if any.
pub fn pick_gizmo_axis(
    ray_origin: Vec3,
    ray_dir: Vec3,
    gizmo_origin: Vec3,
    gizmo_scale: f32,
) -> Option<GizmoAxis> {
    let half_len = gizmo_scale * 0.5;
    let half_thick = gizmo_scale * 0.06;

    let axes = [
        (GizmoAxis::X, Vec3::X),
        (GizmoAxis::Y, Vec3::Y),
        (GizmoAxis::Z, Vec3::Z),
    ];

    let mut best: Option<(GizmoAxis, f32)> = None;
    for (axis, dir) in axes {
        let center = gizmo_origin + dir * half_len;
        let extent = dir * half_len + (Vec3::ONE - dir.abs()) * half_thick;
        let aabb = Aabb::new(center - extent, center + extent);
        if let Some(dist) = ray_aabb(ray_origin, ray_dir, &aabb) {
            if best.is_none() || dist < best.unwrap().1 {
                best = Some((axis, dist));
            }
        }
    }

    best.map(|(axis, _)| axis)
}

/// Compute the parameter along an axis line where a ray is closest.
pub fn closest_point_on_axis(
    ray_origin: Vec3,
    ray_dir: Vec3,
    axis_origin: Vec3,
    axis_dir: Vec3,
) -> f32 {
    let w = ray_origin - axis_origin;
    let a = ray_dir.dot(ray_dir);
    let b = ray_dir.dot(axis_dir);
    let c = axis_dir.dot(axis_dir);
    let d = ray_dir.dot(w);
    let e = axis_dir.dot(w);

    let denom = a * c - b * b;
    if denom.abs() < 1e-10 {
        return -e / c;
    }

    (b * d - a * e) / denom
}

/// Compute the angle of a point relative to a center on a plane perpendicular to `axis`.
pub fn angle_on_plane(point: Vec3, center: Vec3, axis: Vec3) -> f32 {
    let (u, v) = perpendicular_axes(axis);
    let d = point - center;
    d.dot(v).atan2(d.dot(u))
}

/// Build two perpendicular axes for a given normal direction.
fn perpendicular_axes(axis: Vec3) -> (Vec3, Vec3) {
    let up = if axis.dot(Vec3::Y).abs() > 0.9 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let u = axis.cross(up).normalize();
    let v = u.cross(axis).normalize();
    (u, v)
}
