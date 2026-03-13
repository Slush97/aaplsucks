use esox_engine::esox_ui::{TreeState, Ui};
use esox_engine::hecs;
use esox_engine::{Children, Ctx, Parent, Tag, Transform3D};

/// Draw the scene hierarchy panel.
pub fn draw_hierarchy(
    ui: &mut Ui,
    ctx: &Ctx,
    tree_state: &mut TreeState,
    selected: &mut Option<hecs::Entity>,
    camera_entity: Option<hecs::Entity>,
) {
    let scroll_id = super::hash("hierarchy_scroll");
    ui.scrollable(scroll_id, 800.0, |ui| {
        // Collect root entities (have Transform3D but no Parent).
        let roots: Vec<(hecs::Entity, String)> = ctx
            .world
            .query::<&Transform3D>()
            .without::<&Parent>()
            .iter()
            .map(|(e, _)| {
                let label = entity_label(ctx, e, camera_entity);
                (e, label)
            })
            .collect();

        for (entity, label) in &roots {
            draw_entity_node(ui, ctx, tree_state, selected, *entity, label, camera_entity);
        }

        if roots.is_empty() {
            ui.muted_label("(empty scene)");
        }
    });
}

fn draw_entity_node(
    ui: &mut Ui,
    ctx: &Ctx,
    tree_state: &mut TreeState,
    selected: &mut Option<hecs::Entity>,
    entity: hecs::Entity,
    label: &str,
    camera_entity: Option<hecs::Entity>,
) {
    let id = entity.to_bits().get();
    let children_list: Vec<hecs::Entity> = ctx
        .world
        .get::<&Children>(entity)
        .map(|c| c.0.clone())
        .unwrap_or_default();
    let has_children = !children_list.is_empty();

    let response = ui.tree_node(id, tree_state, label, has_children);

    // Sync selection
    if response.response.clicked {
        *selected = Some(entity);
    }

    // Sync tree_state.selected to match our selected entity
    if *selected == Some(entity) && tree_state.selected != Some(id) {
        tree_state.selected = Some(id);
        tree_state.selected_nodes.clear();
        tree_state.selected_nodes.insert(id);
    }

    if response.expanded {
        ui.tree_indent(|ui| {
            for child in &children_list {
                let child_label = entity_label(ctx, *child, camera_entity);
                draw_entity_node(
                    ui,
                    ctx,
                    tree_state,
                    selected,
                    *child,
                    &child_label,
                    camera_entity,
                );
            }
        });
    }
}

fn entity_label(ctx: &Ctx, entity: hecs::Entity, camera_entity: Option<hecs::Entity>) -> String {
    // Use Tag name if available
    if let Ok(tag) = ctx.world.get::<&Tag>(entity) {
        return tag.0.clone();
    }

    // Check for component type hints
    if camera_entity == Some(entity) {
        return "Editor Camera".to_string();
    }
    if ctx.world.get::<&esox_engine::Camera3D>(entity).is_ok() {
        return format!("Camera ({})", entity.to_bits().get());
    }
    if ctx
        .world
        .get::<&esox_engine::PointLightComponent>(entity)
        .is_ok()
    {
        return format!("Point Light ({})", entity.to_bits().get());
    }
    if ctx
        .world
        .get::<&esox_engine::SpotLightComponent>(entity)
        .is_ok()
    {
        return format!("Spot Light ({})", entity.to_bits().get());
    }
    if ctx
        .world
        .get::<&esox_engine::DirectionalLightComponent>(entity)
        .is_ok()
    {
        return format!("Dir Light ({})", entity.to_bits().get());
    }
    if ctx
        .world
        .get::<&esox_engine::MeshRenderer>(entity)
        .is_ok()
    {
        return format!("Mesh ({})", entity.to_bits().get());
    }

    format!("Entity ({})", entity.to_bits().get())
}
