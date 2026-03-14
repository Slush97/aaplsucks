use esox_engine::esox_gfx::mesh3d::Aabb;
use esox_engine::glam::{Mat4, Vec3, Vec4};
use esox_engine::hecs;
use esox_engine::{Ctx, GlobalTransform, MeshRenderer};

/// Axis for gizmo interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

/// Compute a world-space ray from the camera through the given screen pixel.
pub fn screen_to_ray(
    mouse_x: f64,
    mouse_y: f64,
    viewport: (u32, u32),
    view: Mat4,
    projection: Mat4,
) -> (Vec3, Vec3) {
    let (w, h) = viewport;
    // Convert to NDC [-1, 1]
    let ndc_x = (2.0 * mouse_x as f32 / w as f32) - 1.0;
    let ndc_y = 1.0 - (2.0 * mouse_y as f32 / h as f32);

    let inv_vp = (projection * view).inverse();

    let near_point = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far_point = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

    let near = Vec3::new(
        near_point.x / near_point.w,
        near_point.y / near_point.w,
        near_point.z / near_point.w,
    );
    let far = Vec3::new(
        far_point.x / far_point.w,
        far_point.y / far_point.w,
        far_point.z / far_point.w,
    );

    let direction = (far - near).normalize_or_zero();
    (near, direction)
}

/// Ray-AABB intersection test. Returns the distance along the ray if hit.
fn ray_aabb(ray_origin: Vec3, ray_dir: Vec3, aabb: &Aabb) -> Option<f32> {
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

/// Pick the closest entity with a MeshRenderer that the ray hits.
pub fn pick_entity(
    ctx: &Ctx,
    ray_origin: Vec3,
    ray_dir: Vec3,
    exclude: Option<hecs::Entity>,
) -> Option<(hecs::Entity, f32)> {
    let mut closest: Option<(hecs::Entity, f32)> = None;

    for (entity, (gt, mr)) in ctx
        .world
        .query::<(&GlobalTransform, &MeshRenderer)>()
        .iter()
    {
        if !mr.visible {
            continue;
        }
        if exclude == Some(entity) {
            continue;
        }

        // Get local AABB for this mesh
        let local_aabb = match ctx.renderer.mesh_local_aabb(mr.mesh) {
            Some(aabb) => aabb,
            None => {
                // Fallback: unit cube AABB
                Aabb::new(Vec3::splat(-0.5), Vec3::splat(0.5))
            }
        };

        // Transform AABB to world space
        let world_aabb = local_aabb.transformed(&gt.0);

        if let Some(dist) = ray_aabb(ray_origin, ray_dir, &world_aabb) {
            if closest.is_none() || dist < closest.unwrap().1 {
                closest = Some((entity, dist));
            }
        }
    }

    closest
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
    let half_thick = gizmo_scale * 0.06; // slightly wider than the visual cylinder for easier picking

    let axes = [
        (GizmoAxis::X, Vec3::X),
        (GizmoAxis::Y, Vec3::Y),
        (GizmoAxis::Z, Vec3::Z),
    ];

    let mut best: Option<(GizmoAxis, f32)> = None;
    for (axis, dir) in axes {
        let center = gizmo_origin + dir * half_len;
        // Build an AABB for the arrow along this axis
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
///
/// Given a ray (ray_origin, ray_dir) and an axis line (axis_origin, axis_dir),
/// returns `t` such that `axis_origin + axis_dir * t` is the closest point on
/// the axis to the ray.
pub fn closest_point_on_axis(
    ray_origin: Vec3,
    ray_dir: Vec3,
    axis_origin: Vec3,
    axis_dir: Vec3,
) -> f32 {
    // Solve for the closest point between two lines using the standard formula.
    let w = ray_origin - axis_origin;
    let a = ray_dir.dot(ray_dir);
    let b = ray_dir.dot(axis_dir);
    let c = axis_dir.dot(axis_dir);
    let d = ray_dir.dot(w);
    let e = axis_dir.dot(w);

    let denom = a * c - b * b;
    if denom.abs() < 1e-10 {
        // Lines are nearly parallel; just project origin difference.
        return -e / c;
    }

    // t along axis_dir
    (b * d - a * e) / denom
}
