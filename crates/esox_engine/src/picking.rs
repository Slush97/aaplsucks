//! Ray-casting and entity picking utilities.

use glam::{Mat4, Vec3, Vec4};
use hecs;

use esox_gfx::mesh3d::{Aabb, Renderer3D};

use crate::{GlobalTransform, MeshRenderer};

/// Compute a world-space ray from the camera through the given screen pixel.
///
/// Returns `(ray_origin, ray_direction)`.
pub fn screen_to_ray(
    mouse_x: f64,
    mouse_y: f64,
    viewport: (u32, u32),
    view: Mat4,
    projection: Mat4,
) -> (Vec3, Vec3) {
    let (w, h) = viewport;
    let ndc_x = (2.0 * mouse_x as f32 / w as f32) - 1.0;
    let ndc_y = 1.0 - (2.0 * mouse_y as f32 / h as f32);

    let inv_vp = (projection * view).inverse();

    let near_point = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far_point = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

    let near = near_point.truncate() / near_point.w;
    let far = far_point.truncate() / far_point.w;

    let direction = (far - near).normalize_or_zero();
    (near, direction)
}

/// Ray-AABB intersection test. Returns the distance along the ray if hit.
pub fn ray_aabb(ray_origin: Vec3, ray_dir: Vec3, aabb: &Aabb) -> Option<f32> {
    let inv_dir = Vec3::new(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);

    let t1 = (aabb.min - ray_origin) * inv_dir;
    let t2 = (aabb.max - ray_origin) * inv_dir;

    let tmin = t1.min(t2);
    let tmax = t1.max(t2);

    let tmin = tmin.x.max(tmin.y).max(tmin.z);
    let tmax = tmax.x.min(tmax.y).min(tmax.z);

    if tmax >= tmin.max(0.0) {
        Some(tmin.max(0.0))
    } else {
        None
    }
}

/// Pick the closest entity with a [`MeshRenderer`] that the ray hits.
pub fn pick_entity(
    world: &hecs::World,
    renderer: &Renderer3D,
    ray_origin: Vec3,
    ray_dir: Vec3,
    exclude: Option<hecs::Entity>,
) -> Option<(hecs::Entity, f32)> {
    let mut closest: Option<(hecs::Entity, f32)> = None;

    for (entity, (gt, mr)) in world
        .query::<(&GlobalTransform, &MeshRenderer)>()
        .iter()
    {
        if !mr.visible {
            continue;
        }
        if exclude == Some(entity) {
            continue;
        }

        let local_aabb = match renderer.mesh_local_aabb(mr.mesh) {
            Some(aabb) => aabb,
            None => Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5)),
        };

        let world_aabb = local_aabb.transformed(&gt.0);

        if let Some(dist) = ray_aabb(ray_origin, ray_dir, &world_aabb) {
            if closest.is_none() || dist < closest.unwrap().1 {
                closest = Some((entity, dist));
            }
        }
    }

    closest
}

/// Project a ray onto a plane, returning the intersection point.
///
/// Returns `None` if the ray is nearly parallel to the plane or points away.
pub fn project_ray_to_plane(
    ray_origin: Vec3,
    ray_dir: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denom = ray_dir.dot(plane_normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_point - ray_origin).dot(plane_normal) / denom;
    if t < 0.0 {
        return None;
    }
    Some(ray_origin + ray_dir * t)
}

/// Intersect a ray with the Y=0 ground plane. Returns the world-space hit point.
pub fn ray_ground_plane(ray_origin: Vec3, ray_dir: Vec3) -> Option<Vec3> {
    project_ray_to_plane(ray_origin, ray_dir, Vec3::ZERO, Vec3::Y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_aabb_hit() {
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let hit = ray_aabb(origin, dir, &aabb);
        assert!(hit.is_some());
        assert!((hit.unwrap() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn ray_aabb_miss() {
        let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let origin = Vec3::new(5.0, 5.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let hit = ray_aabb(origin, dir, &aabb);
        assert!(hit.is_none());
    }

    #[test]
    fn ground_plane_hit() {
        let origin = Vec3::new(5.0, 10.0, 5.0);
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let hit = ray_ground_plane(origin, dir);
        assert!(hit.is_some());
        let p = hit.unwrap();
        assert!((p.x - 5.0).abs() < 1e-5);
        assert!(p.y.abs() < 1e-5);
        assert!((p.z - 5.0).abs() < 1e-5);
    }

    #[test]
    fn ground_plane_miss_looking_up() {
        let origin = Vec3::new(0.0, 10.0, 0.0);
        let dir = Vec3::new(0.0, 1.0, 0.0);
        assert!(ray_ground_plane(origin, dir).is_none());
    }
}
