use esox_engine::esox_gfx::mesh3d::Aabb;
use esox_engine::glam::{Mat4, Vec3, Vec4};
use esox_engine::hecs;
use esox_engine::{Ctx, GlobalTransform, MeshRenderer};

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
