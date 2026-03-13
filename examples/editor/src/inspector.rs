use esox_engine::esox_ui::Ui;
use esox_engine::glam::{EulerRot, Quat};
use esox_engine::hecs;
use esox_engine::{
    Camera3D, Ctx, DirectionalLightComponent, MeshRenderer, PointLightComponent,
    SpotLightComponent, Tag, Transform3D,
};

use crate::PendingEdit;

/// Draw the inspector panel for the selected entity.
pub fn draw_inspector(
    ui: &mut Ui,
    ctx: &Ctx,
    selected: Option<hecs::Entity>,
    edits: &mut Vec<PendingEdit>,
) {
    let entity = match selected {
        Some(e) => {
            if ctx.world.contains(e) {
                e
            } else {
                ui.muted_label("(deleted entity)");
                return;
            }
        }
        None => {
            ui.muted_label("No entity selected");
            return;
        }
    };

    let scroll_id = super::hash("inspector_scroll");
    ui.scrollable(scroll_id, 800.0, |ui| {
        ui.muted_label(&format!("Entity {}", entity.to_bits().get()));
        ui.spacing(8.0);

        draw_tag_section(ui, ctx, entity);
        draw_transform_section(ui, ctx, entity, edits);
        draw_camera_section(ui, ctx, entity);
        draw_mesh_renderer_section(ui, ctx, entity);
        draw_point_light_section(ui, ctx, entity, edits);
        draw_spot_light_section(ui, ctx, entity, edits);
        draw_dir_light_section(ui, ctx, entity, edits);
    });
}

fn section_header(ui: &mut Ui, label: &str) {
    ui.spacing(8.0);
    ui.header_label(label);
    ui.spacing(4.0);
}

fn draw_tag_section(ui: &mut Ui, ctx: &Ctx, entity: hecs::Entity) {
    if let Ok(tag) = ctx.world.get::<&Tag>(entity) {
        section_header(ui, "Tag");
        ui.label(&tag.0);
    }
}

fn draw_transform_section(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    let t = match ctx.world.get::<&Transform3D>(entity) {
        Ok(t) => *t,
        Err(_) => return,
    };

    section_header(ui, "Transform");

    let (euler_x, euler_y, euler_z) = t.rotation.to_euler(EulerRot::XYZ);

    let mut px = t.position.x as f64;
    let mut py = t.position.y as f64;
    let mut pz = t.position.z as f64;
    let mut rx = euler_x.to_degrees() as f64;
    let mut ry = euler_y.to_degrees() as f64;
    let mut rz = euler_z.to_degrees() as f64;
    let mut sx = t.scale.x as f64;
    let mut sy = t.scale.y as f64;
    let mut sz = t.scale.z as f64;

    let mut changed = false;

    ui.muted_label("Position");
    ui.columns(&[1.0, 1.0, 1.0], |ui, col| match col {
        0 => { changed |= ui.number_input(super::hash("pos_x"), &mut px, 0.1).changed; }
        1 => { changed |= ui.number_input(super::hash("pos_y"), &mut py, 0.1).changed; }
        2 => { changed |= ui.number_input(super::hash("pos_z"), &mut pz, 0.1).changed; }
        _ => {}
    });

    ui.muted_label("Rotation");
    ui.columns(&[1.0, 1.0, 1.0], |ui, col| match col {
        0 => { changed |= ui.number_input(super::hash("rot_x"), &mut rx, 1.0).changed; }
        1 => { changed |= ui.number_input(super::hash("rot_y"), &mut ry, 1.0).changed; }
        2 => { changed |= ui.number_input(super::hash("rot_z"), &mut rz, 1.0).changed; }
        _ => {}
    });

    ui.muted_label("Scale");
    ui.columns(&[1.0, 1.0, 1.0], |ui, col| match col {
        0 => { changed |= ui.number_input(super::hash("scl_x"), &mut sx, 0.1).changed; }
        1 => { changed |= ui.number_input(super::hash("scl_y"), &mut sy, 0.1).changed; }
        2 => { changed |= ui.number_input(super::hash("scl_z"), &mut sz, 0.1).changed; }
        _ => {}
    });

    if changed {
        let new_t = Transform3D {
            position: esox_engine::glam::Vec3::new(px as f32, py as f32, pz as f32),
            rotation: Quat::from_euler(
                EulerRot::XYZ,
                (rx as f32).to_radians(),
                (ry as f32).to_radians(),
                (rz as f32).to_radians(),
            ),
            scale: esox_engine::glam::Vec3::new(sx as f32, sy as f32, sz as f32),
        };
        edits.push(PendingEdit::SetTransform(entity, new_t));
    }
}

fn draw_camera_section(ui: &mut Ui, ctx: &Ctx, entity: hecs::Entity) {
    let cam = match ctx.world.get::<&Camera3D>(entity) {
        Ok(c) => *c,
        Err(_) => return,
    };

    section_header(ui, "Camera");

    let mut fov = cam.fov_y.to_degrees() as f64;
    let mut near = cam.near as f64;
    let mut far = cam.far as f64;

    ui.muted_label("FOV (degrees)");
    ui.number_input_clamped(super::hash("cam_fov"), &mut fov, 1.0, 1.0, 179.0);
    ui.muted_label("Near");
    ui.number_input_clamped(super::hash("cam_near"), &mut near, 0.01, 0.001, 100.0);
    ui.muted_label("Far");
    ui.number_input_clamped(super::hash("cam_far"), &mut far, 1.0, 1.0, 10000.0);
    ui.muted_label(if cam.active { "Active" } else { "Inactive" });
}

fn draw_mesh_renderer_section(ui: &mut Ui, ctx: &Ctx, entity: hecs::Entity) {
    let (visible, tint) = match ctx.world.get::<&MeshRenderer>(entity) {
        Ok(m) => (m.visible, m.tint),
        Err(_) => return,
    };

    section_header(ui, "Mesh Renderer");
    ui.muted_label(&format!("Visible: {visible}"));
    ui.muted_label(&format!(
        "Tint: [{:.2}, {:.2}, {:.2}, {:.2}]",
        tint[0], tint[1], tint[2], tint[3]
    ));
}

fn draw_point_light_section(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    let pl = match ctx.world.get::<&PointLightComponent>(entity) {
        Ok(l) => *l,
        Err(_) => return,
    };

    section_header(ui, "Point Light");

    let mut intensity = pl.intensity as f64;
    let mut range = pl.range as f64;

    ui.muted_label(&format!(
        "Color: [{:.2}, {:.2}, {:.2}]",
        pl.color[0], pl.color[1], pl.color[2]
    ));

    ui.muted_label("Intensity");
    if ui.number_input_clamped(super::hash("pl_intensity"), &mut intensity, 0.5, 0.0, 1000.0).changed {
        edits.push(PendingEdit::SetPointLightIntensity(entity, intensity as f32));
    }

    ui.muted_label("Range");
    if ui.number_input_clamped(super::hash("pl_range"), &mut range, 0.5, 0.1, 500.0).changed {
        edits.push(PendingEdit::SetPointLightRange(entity, range as f32));
    }

    ui.muted_label(&format!("Shadows: {}", pl.cast_shadows));
}

fn draw_spot_light_section(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    let sl = match ctx.world.get::<&SpotLightComponent>(entity) {
        Ok(l) => *l,
        Err(_) => return,
    };

    section_header(ui, "Spot Light");

    let mut intensity = sl.intensity as f64;
    let mut range = sl.range as f64;
    let inner = sl.inner_cone_angle.to_degrees() as f64;
    let outer = sl.outer_cone_angle.to_degrees() as f64;

    ui.muted_label(&format!(
        "Color: [{:.2}, {:.2}, {:.2}]",
        sl.color[0], sl.color[1], sl.color[2]
    ));

    ui.muted_label("Intensity");
    if ui.number_input_clamped(super::hash("sl_intensity"), &mut intensity, 0.5, 0.0, 1000.0).changed {
        edits.push(PendingEdit::SetSpotLightIntensity(entity, intensity as f32));
    }

    ui.muted_label("Range");
    if ui.number_input_clamped(super::hash("sl_range"), &mut range, 0.5, 0.1, 500.0).changed {
        edits.push(PendingEdit::SetSpotLightRange(entity, range as f32));
    }

    ui.muted_label(&format!("Inner: {:.1}", inner));
    ui.muted_label(&format!("Outer: {:.1}", outer));
    ui.muted_label(&format!("Shadows: {}", sl.cast_shadows));
}

fn draw_dir_light_section(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    let dl = match ctx.world.get::<&DirectionalLightComponent>(entity) {
        Ok(l) => *l,
        Err(_) => return,
    };

    section_header(ui, "Directional Light");

    let mut intensity = dl.intensity as f64;

    ui.muted_label(&format!(
        "Color: [{:.2}, {:.2}, {:.2}]",
        dl.color[0], dl.color[1], dl.color[2]
    ));

    ui.muted_label("Intensity");
    if ui.number_input_clamped(super::hash("dl_intensity"), &mut intensity, 0.1, 0.0, 100.0).changed {
        edits.push(PendingEdit::SetDirLightIntensity(entity, intensity as f32));
    }
}
