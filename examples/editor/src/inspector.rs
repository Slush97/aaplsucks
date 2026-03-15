use esox_engine::esox_gfx::mesh3d::{BlendMode3D, MaterialType};
use esox_engine::esox_ui::{InputState, SelectState, Ui};
use esox_engine::glam::{EulerRot, Quat, Vec3};
use esox_engine::hecs;
use esox_engine::{
    Camera3D, Ctx, DirectionalLightComponent, MeshRenderer, PointLightComponent,
    SpotLightComponent, Tag, Transform3D,
};

use crate::{ComponentKind, PendingEdit};

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

        draw_tag_section(ui, ctx, entity, edits);
        draw_transform_section(ui, ctx, entity, edits);
        draw_camera_section(ui, ctx, entity, edits);
        draw_mesh_renderer_section(ui, ctx, entity, edits);
        draw_material_section(ui, ctx, entity, edits);
        draw_point_light_section(ui, ctx, entity, edits);
        draw_spot_light_section(ui, ctx, entity, edits);
        draw_dir_light_section(ui, ctx, entity, edits);

        // "Add Component" dropdown
        ui.spacing(12.0);
        draw_add_component(ui, ctx, entity, edits);
    });
}

fn draw_tag_section(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    let tag = match ctx.world.get::<&Tag>(entity) {
        Ok(t) => t.0.clone(),
        Err(_) => return,
    };

    ui.collapsing_header(super::hash("sec_tag"), "Tag", true, |ui| {
        let mut input = InputState::new();
        input.text = tag;
        if ui.text_input(super::hash("tag_name"), &mut input, "name...").changed {
            edits.push(PendingEdit::SetTag(entity, input.text));
        }
    });
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

    ui.collapsing_header(super::hash("sec_transform"), "Transform", true, |ui| {
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
        ui.flex_row().gap(4.0).show(|ui| {
            changed |= ui.number_input(super::hash("pos_x"), &mut px, 0.1).changed;
            changed |= ui.number_input(super::hash("pos_y"), &mut py, 0.1).changed;
            changed |= ui.number_input(super::hash("pos_z"), &mut pz, 0.1).changed;
        });

        ui.muted_label("Rotation");
        ui.flex_row().gap(4.0).show(|ui| {
            changed |= ui.number_input(super::hash("rot_x"), &mut rx, 1.0).changed;
            changed |= ui.number_input(super::hash("rot_y"), &mut ry, 1.0).changed;
            changed |= ui.number_input(super::hash("rot_z"), &mut rz, 1.0).changed;
        });

        ui.muted_label("Scale");
        ui.flex_row().gap(4.0).show(|ui| {
            changed |= ui.number_input(super::hash("scl_x"), &mut sx, 0.1).changed;
            changed |= ui.number_input(super::hash("scl_y"), &mut sy, 0.1).changed;
            changed |= ui.number_input(super::hash("scl_z"), &mut sz, 0.1).changed;
        });

        if changed {
            let new_t = Transform3D {
                position: Vec3::new(px as f32, py as f32, pz as f32),
                rotation: Quat::from_euler(
                    EulerRot::XYZ,
                    (rx as f32).to_radians(),
                    (ry as f32).to_radians(),
                    (rz as f32).to_radians(),
                ),
                scale: Vec3::new(sx as f32, sy as f32, sz as f32),
            };
            edits.push(PendingEdit::SetTransform(entity, new_t));
        }

        // Reset transform button
        if ui.ghost_button(super::hash("reset_transform"), "Reset").clicked {
            edits.push(PendingEdit::SetTransform(entity, Transform3D::default()));
        }
    });
}

fn draw_camera_section(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    let cam = match ctx.world.get::<&Camera3D>(entity) {
        Ok(c) => *c,
        Err(_) => return,
    };

    ui.collapsing_header(super::hash("sec_camera"), "Camera", true, |ui| {
        if ui.ghost_button(super::hash("rm_camera"), "X Remove").clicked {
            edits.push(PendingEdit::RemoveComponent(entity, ComponentKind::Camera));
            return;
        }
        // FOV slider
        let mut fov = cam.fov_y.to_degrees();
        ui.labeled("FOV (degrees)", |ui| {
            if ui.slider_f32(super::hash("cam_fov"), &mut fov, 1.0, 179.0).changed {
                edits.push(PendingEdit::SetCameraFov(entity, fov.to_radians()));
            }
        });

        let mut near = cam.near as f64;
        let mut far = cam.far as f64;

        ui.labeled("Near", |ui| {
            if ui.number_input_clamped(super::hash("cam_near"), &mut near, 0.01, 0.001, 100.0).changed {
                edits.push(PendingEdit::SetCameraNear(entity, near as f32));
            }
        });
        ui.labeled("Far", |ui| {
            if ui.number_input_clamped(super::hash("cam_far"), &mut far, 1.0, 1.0, 10000.0).changed {
                edits.push(PendingEdit::SetCameraFar(entity, far as f32));
            }
        });
        ui.muted_label(if cam.active { "Active" } else { "Inactive" });
    });
}

fn draw_mesh_renderer_section(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    let (visible, tint) = match ctx.world.get::<&MeshRenderer>(entity) {
        Ok(m) => (m.visible, m.tint),
        Err(_) => return,
    };

    ui.collapsing_header(super::hash("sec_mesh_renderer"), "Mesh Renderer", true, |ui| {
        if ui.ghost_button(super::hash("rm_mesh_renderer"), "X Remove").clicked {
            edits.push(PendingEdit::RemoveComponent(entity, ComponentKind::MeshRenderer));
            return;
        }

        // Mesh name + assignment
        let mr = ctx.world.get::<&MeshRenderer>(entity).unwrap();
        let mesh_name = ctx.assets.name_for_gpu_mesh(mr.mesh)
            .unwrap_or("(unnamed)").to_string();
        let mat_name = ctx.assets.name_for_gpu_material(mr.material)
            .unwrap_or("(unnamed)").to_string();

        let mesh_names = ctx.assets.mesh_name_list();
        if !mesh_names.is_empty() {
            ui.muted_label(&format!("Mesh: {mesh_name}"));
            let choices: Vec<&str> = mesh_names.iter().map(|s| s.as_str()).collect();
            let mut sel = SelectState::new();
            sel.selected_index = mesh_names.iter().position(|n| n == &mesh_name).unwrap_or(0);
            if ui.select(super::hash("mr_mesh_sel"), &mut sel, &choices).changed {
                if let Some(handle) = ctx.assets.find_mesh_by_name(&mesh_names[sel.selected_index]) {
                    edits.push(PendingEdit::SetMesh(entity, handle));
                }
            }
        }

        let material_names = ctx.assets.material_name_list();
        if !material_names.is_empty() {
            ui.muted_label(&format!("Material: {mat_name}"));
            let choices: Vec<&str> = material_names.iter().map(|s| s.as_str()).collect();
            let mut sel = SelectState::new();
            sel.selected_index = material_names.iter().position(|n| n == &mat_name).unwrap_or(0);
            if ui.select(super::hash("mr_mat_sel"), &mut sel, &choices).changed {
                if let Some(handle) = ctx.assets.find_material_by_name(&material_names[sel.selected_index]) {
                    edits.push(PendingEdit::SetMaterial(entity, handle));
                }
            }
        }

        let label = if visible { "Visible: ON" } else { "Visible: OFF" };
        if ui.button(super::hash("mr_visible"), label).clicked {
            edits.push(PendingEdit::SetMeshVisible(entity, !visible));
        }

        // Tint RGBA sliders
        ui.muted_label("Tint");
        let mut tint = tint;
        let mut tint_changed = false;

        let labels = ["R", "G", "B", "A"];
        for i in 0..4 {
            ui.labeled(labels[i], |ui| {
                if ui.slider_f32(super::hash(&format!("mr_tint_{i}")), &mut tint[i], 0.0, 1.0).changed {
                    tint_changed = true;
                }
            });
        }
        if tint_changed {
            edits.push(PendingEdit::SetMeshTint(entity, tint));
        }
    });
}

fn draw_material_section(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    let handle = match ctx.world.get::<&MeshRenderer>(entity) {
        Ok(mr) => mr.material,
        Err(_) => return,
    };

    let desc = match ctx.renderer.material_descriptor(handle) {
        Some(d) => d.clone(),
        None => return,
    };

    ui.collapsing_header(super::hash("sec_material"), "Material Properties", false, |ui| {
        let mut new_desc = desc.clone();
        let mut changed = false;

        // Material type select
        ui.muted_label("Type");
        let type_choices = ["Unlit", "Lit", "PBR"];
        let mut sel = SelectState::new();
        sel.selected_index = match desc.material_type {
            MaterialType::Unlit => 0,
            MaterialType::Lit => 1,
            MaterialType::PBR => 2,
            _ => 2,
        };
        if ui.select(super::hash("mat_type"), &mut sel, &type_choices).changed {
            new_desc.material_type = match sel.selected_index {
                0 => MaterialType::Unlit,
                1 => MaterialType::Lit,
                _ => MaterialType::PBR,
            };
            changed = true;
        }

        // Albedo RGBA sliders
        ui.muted_label("Albedo");
        let labels = ["R", "G", "B", "A"];
        for i in 0..4 {
            ui.labeled(labels[i], |ui| {
                if ui.slider_f32(super::hash(&format!("mat_alb_{i}")), &mut new_desc.albedo[i], 0.0, 1.0).changed {
                    changed = true;
                }
            });
        }

        // Emissive RGB sliders (HDR range)
        ui.muted_label("Emissive");
        let em_labels = ["R", "G", "B"];
        for i in 0..3 {
            ui.labeled(em_labels[i], |ui| {
                if ui.slider_f32(super::hash(&format!("mat_em_{i}")), &mut new_desc.emissive[i], 0.0, 10.0).changed {
                    changed = true;
                }
            });
        }

        // PBR-only: metallic + roughness
        if new_desc.material_type == MaterialType::PBR {
            ui.labeled("Metallic", |ui| {
                if ui.slider_f32(super::hash("mat_metallic"), &mut new_desc.metallic, 0.0, 1.0).changed {
                    changed = true;
                }
            });

            ui.labeled("Roughness", |ui| {
                if ui.slider_f32(super::hash("mat_roughness"), &mut new_desc.roughness, 0.0, 1.0).changed {
                    changed = true;
                }
            });
        }

        // Blend mode select
        ui.muted_label("Blend Mode");
        let blend_choices = ["Opaque", "Alpha Blend", "Additive"];
        let mut sel = SelectState::new();
        sel.selected_index = match desc.blend_mode {
            BlendMode3D::Opaque => 0,
            BlendMode3D::AlphaBlend => 1,
            BlendMode3D::Additive => 2,
            _ => 0,
        };
        if ui.select(super::hash("mat_blend"), &mut sel, &blend_choices).changed {
            new_desc.blend_mode = match sel.selected_index {
                0 => BlendMode3D::Opaque,
                1 => BlendMode3D::AlphaBlend,
                _ => BlendMode3D::Additive,
            };
            changed = true;
        }

        // Double-sided toggle
        let ds_label = if new_desc.double_sided { "Double Sided: ON" } else { "Double Sided: OFF" };
        if ui.button(super::hash("mat_double_sided"), ds_label).clicked {
            new_desc.double_sided = !new_desc.double_sided;
            changed = true;
        }

        if changed {
            edits.push(PendingEdit::SetMaterialDescriptor(entity, new_desc));
        }
    });
}

/// Helper to draw an RGB color editor with sliders.
fn draw_color_editor(
    ui: &mut Ui,
    id_base: &str,
    color: &mut [f32; 3],
) -> bool {
    let mut changed = false;
    let labels = ["R", "G", "B"];
    for i in 0..3 {
        ui.labeled(labels[i], |ui| {
            if ui.slider_f32(super::hash(&format!("{id_base}_{i}")), &mut color[i], 0.0, 1.0).changed {
                changed = true;
            }
        });
    }
    changed
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

    ui.collapsing_header(super::hash("sec_point_light"), "Point Light", true, |ui| {
        if ui.ghost_button(super::hash("rm_point_light"), "X Remove").clicked {
            edits.push(PendingEdit::RemoveComponent(entity, ComponentKind::PointLight));
            return;
        }
        // Color
        ui.muted_label("Color");
        let mut color = pl.color;
        if draw_color_editor(ui, "pl_c", &mut color) {
            edits.push(PendingEdit::SetPointLightColor(entity, color));
        }

        // Intensity slider
        let mut intensity = pl.intensity;
        ui.labeled("Intensity", |ui| {
            if ui.slider_f32(super::hash("pl_intensity"), &mut intensity, 0.0, 1000.0).changed {
                edits.push(PendingEdit::SetPointLightIntensity(entity, intensity));
            }
        });

        // Range slider
        let mut range = pl.range;
        ui.labeled("Range", |ui| {
            if ui.slider_f32(super::hash("pl_range"), &mut range, 0.1, 500.0).changed {
                edits.push(PendingEdit::SetPointLightRange(entity, range));
            }
        });

        // Shadow toggle
        let shadow_label = if pl.cast_shadows { "Shadows: ON" } else { "Shadows: OFF" };
        if ui.button(super::hash("pl_shadows"), shadow_label).clicked {
            edits.push(PendingEdit::SetPointLightShadows(entity, !pl.cast_shadows));
        }
    });
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

    ui.collapsing_header(super::hash("sec_spot_light"), "Spot Light", true, |ui| {
        if ui.ghost_button(super::hash("rm_spot_light"), "X Remove").clicked {
            edits.push(PendingEdit::RemoveComponent(entity, ComponentKind::SpotLight));
            return;
        }
        // Color
        ui.muted_label("Color");
        let mut color = sl.color;
        if draw_color_editor(ui, "sl_c", &mut color) {
            edits.push(PendingEdit::SetSpotLightColor(entity, color));
        }

        // Intensity slider
        let mut intensity = sl.intensity;
        ui.labeled("Intensity", |ui| {
            if ui.slider_f32(super::hash("sl_intensity"), &mut intensity, 0.0, 1000.0).changed {
                edits.push(PendingEdit::SetSpotLightIntensity(entity, intensity));
            }
        });

        // Range slider
        let mut range = sl.range;
        ui.labeled("Range", |ui| {
            if ui.slider_f32(super::hash("sl_range"), &mut range, 0.1, 500.0).changed {
                edits.push(PendingEdit::SetSpotLightRange(entity, range));
            }
        });

        // Inner cone slider (degrees)
        let mut inner = sl.inner_cone_angle.to_degrees();
        ui.labeled("Inner Cone (deg)", |ui| {
            if ui.slider_f32(super::hash("sl_inner"), &mut inner, 0.0, 90.0).changed {
                edits.push(PendingEdit::SetSpotLightInnerCone(entity, inner.to_radians()));
            }
        });

        // Outer cone slider (degrees)
        let mut outer = sl.outer_cone_angle.to_degrees();
        ui.labeled("Outer Cone (deg)", |ui| {
            if ui.slider_f32(super::hash("sl_outer"), &mut outer, 0.0, 90.0).changed {
                edits.push(PendingEdit::SetSpotLightOuterCone(entity, outer.to_radians()));
            }
        });

        // Shadow toggle
        let shadow_label = if sl.cast_shadows { "Shadows: ON" } else { "Shadows: OFF" };
        if ui.button(super::hash("sl_shadows"), shadow_label).clicked {
            edits.push(PendingEdit::SetSpotLightShadows(entity, !sl.cast_shadows));
        }
    });
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

    ui.collapsing_header(super::hash("sec_dir_light"), "Directional Light", true, |ui| {
        if ui.ghost_button(super::hash("rm_dir_light"), "X Remove").clicked {
            edits.push(PendingEdit::RemoveComponent(entity, ComponentKind::DirLight));
            return;
        }
        // Color
        ui.muted_label("Color");
        let mut color = dl.color;
        if draw_color_editor(ui, "dl_c", &mut color) {
            edits.push(PendingEdit::SetDirLightColor(entity, color));
        }

        // Intensity slider
        let mut intensity = dl.intensity;
        ui.labeled("Intensity", |ui| {
            if ui.slider_f32(super::hash("dl_intensity"), &mut intensity, 0.0, 100.0).changed {
                edits.push(PendingEdit::SetDirLightIntensity(entity, intensity));
            }
        });
    });
}

fn draw_add_component(
    ui: &mut Ui,
    ctx: &Ctx,
    entity: hecs::Entity,
    edits: &mut Vec<PendingEdit>,
) {
    // Build list of components that can be added (exclude ones already present)
    let has_pl = ctx.world.get::<&PointLightComponent>(entity).is_ok();
    let has_sl = ctx.world.get::<&SpotLightComponent>(entity).is_ok();
    let has_dl = ctx.world.get::<&DirectionalLightComponent>(entity).is_ok();
    let has_cam = ctx.world.get::<&Camera3D>(entity).is_ok();
    let has_mr = ctx.world.get::<&MeshRenderer>(entity).is_ok();

    let mut choices = Vec::new();
    let mut kinds = Vec::new();
    if !has_pl { choices.push("Point Light"); kinds.push(ComponentKind::PointLight); }
    if !has_sl { choices.push("Spot Light"); kinds.push(ComponentKind::SpotLight); }
    if !has_dl { choices.push("Dir Light"); kinds.push(ComponentKind::DirLight); }
    if !has_cam { choices.push("Camera"); kinds.push(ComponentKind::Camera); }
    if !has_mr { choices.push("Mesh Renderer"); kinds.push(ComponentKind::MeshRenderer); }

    if choices.is_empty() {
        return;
    }

    ui.muted_label("Add Component");
    let mut sel = SelectState::new();
    if ui.select(super::hash("add_component_sel"), &mut sel, &choices).changed {
        edits.push(PendingEdit::AddComponent(entity, kinds[sel.selected_index]));
    }
}
