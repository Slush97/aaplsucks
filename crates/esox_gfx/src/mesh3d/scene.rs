//! Scene graph — hierarchical transform management and batch drawing for loaded scenes.

use glam::Mat4;

use super::instance::InstanceData;
use super::material::MaterialHandle;
use super::mesh::MeshHandle;
use super::renderer::Renderer3D;
use super::transform::Transform;

/// A renderable scene with hierarchical transforms.
///
/// Built from a loaded glTF scene. Manages local → world transform propagation
/// and queues draw calls for all meshes in the hierarchy.
pub struct Scene3D {
    nodes: Vec<SceneNode>,
    roots: Vec<usize>,
    world_transforms: Vec<Mat4>,
    dirty: bool,
}

struct SceneNode {
    local_transform: Transform,
    /// (MeshHandle, MaterialHandle) pairs for this node.
    meshes: Vec<(MeshHandle, MaterialHandle)>,
    children: Vec<usize>,
}

impl Scene3D {
    /// Build a scene from uploaded glTF handles.
    ///
    /// Automatically resolves mesh-to-material assignments using the stored
    /// `mesh_material_indices` from the glTF loader.
    pub fn from_gltf(handles: &super::gltf_loader::GltfSceneHandles) -> Self {
        let default_material = MaterialHandle(0);

        let nodes: Vec<SceneNode> = handles
            .nodes
            .iter()
            .map(|gltf_node| {
                let meshes: Vec<(MeshHandle, MaterialHandle)> = gltf_node
                    .mesh_indices
                    .iter()
                    .map(|&mi| {
                        let mesh_handle = handles.meshes[mi];
                        // Look up the material from the stored mesh→material mapping.
                        let mat_handle = handles
                            .mesh_material_indices
                            .get(mi)
                            .and_then(|opt_idx| {
                                opt_idx.and_then(|idx| handles.materials.get(idx).copied())
                            })
                            .unwrap_or(default_material);
                        (mesh_handle, mat_handle)
                    })
                    .collect();

                SceneNode {
                    local_transform: gltf_node.transform,
                    meshes,
                    children: gltf_node.children.clone(),
                }
            })
            .collect();

        let node_count = nodes.len();

        let mut scene = Self {
            nodes,
            roots: handles.roots.clone(),
            world_transforms: vec![Mat4::IDENTITY; node_count],
            dirty: true,
        };

        scene.update_transforms();
        scene
    }

    /// Build a scene from glTF handles with explicit per-mesh material assignments.
    ///
    /// `mesh_materials` maps mesh index → MaterialHandle (parallel to handles.meshes).
    pub fn from_gltf_with_materials(
        handles: &super::gltf_loader::GltfSceneHandles,
        mesh_materials: &[MaterialHandle],
    ) -> Self {
        let default_material = MaterialHandle(0);

        let nodes: Vec<SceneNode> = handles
            .nodes
            .iter()
            .map(|gltf_node| {
                let meshes: Vec<(MeshHandle, MaterialHandle)> = gltf_node
                    .mesh_indices
                    .iter()
                    .map(|&mi| {
                        let mesh_handle = handles.meshes[mi];
                        let mat_handle = mesh_materials
                            .get(mi)
                            .copied()
                            .unwrap_or(default_material);
                        (mesh_handle, mat_handle)
                    })
                    .collect();

                SceneNode {
                    local_transform: gltf_node.transform,
                    meshes,
                    children: gltf_node.children.clone(),
                }
            })
            .collect();

        let node_count = nodes.len();

        let mut scene = Self {
            nodes,
            roots: handles.roots.clone(),
            world_transforms: vec![Mat4::IDENTITY; node_count],
            dirty: true,
        };

        scene.update_transforms();
        scene
    }

    /// Set the local transform of a specific node.
    pub fn set_node_transform(&mut self, index: usize, t: Transform) {
        if index < self.nodes.len() {
            self.nodes[index].local_transform = t;
            self.dirty = true;
        }
    }

    /// Set a root transform that affects the entire scene.
    /// Modifies the local transform of all root nodes.
    pub fn set_root_transform(&mut self, t: Transform) {
        for &root in &self.roots.clone() {
            if root < self.nodes.len() {
                self.nodes[root].local_transform = t;
            }
        }
        self.dirty = true;
    }

    /// Recompute world matrices from the node hierarchy.
    pub fn update_transforms(&mut self) {
        if !self.dirty {
            return;
        }
        for &root in &self.roots.clone() {
            self.propagate_transform(root, Mat4::IDENTITY);
        }
        self.dirty = false;
    }

    fn propagate_transform(&mut self, node_idx: usize, parent_world: Mat4) {
        if node_idx >= self.nodes.len() {
            return;
        }
        let local = self.nodes[node_idx].local_transform.matrix();
        let world = parent_world * local;
        self.world_transforms[node_idx] = world;

        let children = self.nodes[node_idx].children.clone();
        for child in children {
            self.propagate_transform(child, world);
        }
    }

    /// Queue draw calls for all meshes in the scene.
    pub fn draw(&self, renderer: &mut Renderer3D) {
        for (i, node) in self.nodes.iter().enumerate() {
            if node.meshes.is_empty() {
                continue;
            }
            let world = self.world_transforms[i];
            let instance = InstanceData {
                model: world.to_cols_array_2d(),
                color: [1.0, 1.0, 1.0, 1.0],
                params: [0.0; 4],
            };
            for &(mesh, material) in &node.meshes {
                renderer.draw_with_material(mesh, material, &[instance]);
            }
        }
    }

    /// Get the world transform matrix for a node.
    pub fn world_transform(&self, index: usize) -> Mat4 {
        self.world_transforms.get(index).copied().unwrap_or(Mat4::IDENTITY)
    }

    /// Number of nodes in the scene.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn make_scene(
        transforms: Vec<Transform>,
        children: Vec<Vec<usize>>,
        roots: Vec<usize>,
    ) -> Scene3D {
        let nodes: Vec<SceneNode> = transforms
            .into_iter()
            .zip(children)
            .map(|(t, c)| SceneNode {
                local_transform: t,
                meshes: Vec::new(),
                children: c,
            })
            .collect();
        let node_count = nodes.len();
        let mut scene = Scene3D {
            nodes,
            roots,
            world_transforms: vec![Mat4::IDENTITY; node_count],
            dirty: true,
        };
        scene.update_transforms();
        scene
    }

    #[test]
    fn identity_hierarchy() {
        let scene = make_scene(
            vec![Transform::IDENTITY, Transform::IDENTITY],
            vec![vec![1], vec![]],
            vec![0],
        );
        assert_eq!(scene.world_transforms[0], Mat4::IDENTITY);
        assert_eq!(scene.world_transforms[1], Mat4::IDENTITY);
    }

    #[test]
    fn parent_child_translation() {
        let scene = make_scene(
            vec![
                Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
                Transform::from_position(Vec3::new(0.0, 2.0, 0.0)),
            ],
            vec![vec![1], vec![]],
            vec![0],
        );

        let child_world = scene.world_transforms[1];
        let translation = child_world.col(3).truncate();
        assert!(
            (translation - Vec3::new(1.0, 2.0, 0.0)).length() < 1e-5,
            "child world position should be (1,2,0), got {:?}",
            translation
        );
    }

    #[test]
    fn three_level_hierarchy() {
        let scene = make_scene(
            vec![
                Transform::from_position(Vec3::new(1.0, 0.0, 0.0)),
                Transform::from_position(Vec3::new(0.0, 1.0, 0.0)),
                Transform::from_position(Vec3::new(0.0, 0.0, 1.0)),
            ],
            vec![vec![1], vec![2], vec![]],
            vec![0],
        );

        let leaf_world = scene.world_transforms[2];
        let translation = leaf_world.col(3).truncate();
        assert!(
            (translation - Vec3::new(1.0, 1.0, 1.0)).length() < 1e-5,
            "leaf should accumulate all translations, got {:?}",
            translation
        );
    }

    #[test]
    fn set_node_transform_marks_dirty() {
        let mut scene = make_scene(
            vec![Transform::IDENTITY],
            vec![vec![]],
            vec![0],
        );
        assert!(!scene.dirty);
        scene.set_node_transform(0, Transform::from_position(Vec3::new(5.0, 0.0, 0.0)));
        assert!(scene.dirty);
        scene.update_transforms();
        assert!(!scene.dirty);
        let t = scene.world_transforms[0].col(3).truncate();
        assert!((t - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
    }
}
