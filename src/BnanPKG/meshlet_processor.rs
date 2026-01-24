use std::io::{Write, stdout};
use cgmath::{Vector3, Vector4};
use meshopt::{
    build_meshlets, simplify, partition_clusters,
    SimplifyOptions, VertexDataAdapter,
    compute_meshlet_bounds
};

use BnanR::core::bnan_mesh::{BnanMeshletDAG, BnanMeshletDAGNode, BnanMeshletData, BnanMeshletRawData, Vertex};

const MAX_VERTICES: usize = 64;
const MAX_TRIANGLES: usize = 124;
const TARGET_PARTITION_SIZE: usize = 4;

/// Print progress bar to terminal
fn print_progress(msg: &str, current: usize, total: usize) {
    let pct = if total > 0 { (current as f32 / total as f32 * 100.0) as u32 } else { 100 };
    let bar_len = 40;
    let filled = (pct as usize * bar_len) / 100;
    print!("\r{}: [{}{}] {}%  ",
        msg,
        "=".repeat(filled),
        " ".repeat(bar_len - filled),
        pct);
    let _ = stdout().flush();
}

/// Create a VertexDataAdapter from flat f32 positions
fn make_vertex_adapter(positions: &[f32]) -> VertexDataAdapter<'_> {
    let positions_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(positions.as_ptr() as *const u8, positions.len() * 4)
    };
    VertexDataAdapter::new(positions_bytes, 12, 0).unwrap()
}

/// Represents a meshlet's geometry for processing
struct MeshletGeometry {
    positions: Vec<Vector3<f32>>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,  // Global indices into positions/vertices
}

/// Build complete meshlet DAG from mesh data
/// 
/// Algorithm:
/// 1. Optimize mesh and build base meshlets (obeying MAX_VERTICES, MAX_TRIANGLES)
/// 2. Use partition_clusters to group adjacent meshlets into groups of 3-4
/// 3. Merge geometry in each group
/// 4. Simplify merged geometry to fit under MAX constraints
/// 5. Repeat until < 4 meshlets remain
/// 6. Save entire DAG tree
pub fn generate_meshlet_dag(
    positions: &[f32],
    vertices: &[Vertex],
    indices: &[u32],
) -> (BnanMeshletDAG, Vec<BnanMeshletRawData>) {
    let vertex_count = positions.len() / 3;
    
    println!("Generating meshlet DAG for {} vertices, {} triangles", 
             vertex_count, indices.len() / 3);
    
    // Create vertex data adapter for meshopt
    let vertex_adapter = make_vertex_adapter(positions);
    
    // Step 1: Generate base meshlets
    print_progress("Building base meshlets", 0, 1);
    let base_meshlets = build_meshlets(indices, &vertex_adapter, MAX_VERTICES, MAX_TRIANGLES, 0.5);
    let base_count = base_meshlets.meshlets.len();
    print_progress("Building base meshlets", 1, 1);
    println!();
    println!("Generated {} base meshlets", base_count);
    
    let mut all_nodes: Vec<BnanMeshletDAGNode> = Vec::new();
    let mut all_raw_data: Vec<BnanMeshletRawData> = Vec::new();
    let mut leaf_indices: Vec<u32> = Vec::new();
    
    // Convert base meshlets to DAG nodes (LOD level 0)
    // Store the geometry for each meshlet so we can merge later
    let mut current_level_geometry: Vec<MeshletGeometry> = Vec::new();
    
    print_progress("Processing base meshlets", 0, base_count);
    for (i, _meshlet_desc) in base_meshlets.meshlets.iter().enumerate() {
        let meshlet = base_meshlets.get(i);
        
        // Extract positions for this meshlet (local copies)
        let mut meshlet_positions: Vec<Vector3<f32>> = Vec::new();
        for &vid in meshlet.vertices {
            let idx = vid as usize;
            meshlet_positions.push(Vector3::new(
                positions[idx * 3],
                positions[idx * 3 + 1],
                positions[idx * 3 + 2],
            ));
        }
        
        // Extract vertex attributes
        let mut meshlet_vertices: Vec<Vertex> = Vec::new();
        for &vid in meshlet.vertices {
            meshlet_vertices.push(vertices[vid as usize].clone());
        }
        
        // Triangle indices are local to meshlet (u8 indices into meshlet.vertices)
        let meshlet_triangles = meshlet.triangles.to_vec();
        
        // Build global indices for this meshlet's geometry
        let meshlet_indices: Vec<u32> = meshlet_triangles.iter()
            .map(|&local_idx| local_idx as u32)
            .collect();
        
        // Store geometry for LOD processing
        current_level_geometry.push(MeshletGeometry {
            positions: meshlet_positions.clone(),
            vertices: meshlet_vertices.clone(),
            indices: meshlet_indices,
        });
        
        // Compute bounds
        let bounds = compute_meshlet_bounds(meshlet, &vertex_adapter);
        
        let node_idx = all_nodes.len() as u32;
        leaf_indices.push(node_idx);
        
        all_nodes.push(BnanMeshletDAGNode {
            meshlet: BnanMeshletData {
                position_offset: 0,
                vertex_offset: 0,
                vertex_count: meshlet.vertices.len() as u32,
                triangle_offset: 0,
                triangle_count: (meshlet.triangles.len() / 3) as u32,
            },
            lod_level: 0,
            child_indices: Vec::new(),
            parent_index: None,
            bounds: Vector4::new(bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius),
        });
        
        all_raw_data.push(BnanMeshletRawData {
            positions: meshlet_positions,
            vertices: meshlet_vertices,
            triangles: meshlet_triangles,
        });
        
        if (i + 1) % 10 == 0 || i + 1 == base_count {
            print_progress("Processing base meshlets", i + 1, base_count);
        }
    }
    println!();
    
    // Step 2: Build LOD hierarchy iteratively
    let mut current_level_node_indices: Vec<u32> = leaf_indices.clone();
    let mut lod_level = 1u32;
    
    while current_level_node_indices.len() >= TARGET_PARTITION_SIZE {
        println!("Building LOD level {} from {} meshlets", lod_level, current_level_node_indices.len());
        
        // Build data for partition_clusters
        // cluster_indices: all vertex indices from all meshlets concatenated
        // cluster_index_counts: number of indices in each cluster (meshlet)
        let mut cluster_indices: Vec<u32> = Vec::new();
        let mut cluster_index_counts: Vec<u32> = Vec::new();
        
        for geom in &current_level_geometry {
            cluster_indices.extend(&geom.indices);
            cluster_index_counts.push(geom.indices.len() as u32);
        }
        
        // Find max vertex index to determine vertex_count for partition
        let max_vertex_idx = current_level_geometry.iter()
            .map(|g| g.positions.len())
            .max()
            .unwrap_or(0);
        
        // For partition_clusters, we need to provide the total vertex count
        // across all clusters. Since each meshlet has its own local vertex space,
        // we need to compute a global vertex count.
        let total_vertices: usize = current_level_geometry.iter()
            .map(|g| g.positions.len())
            .sum();
        
        // Destination array for partition assignments
        let mut partition_dest: Vec<u32> = vec![0; current_level_geometry.len()];
        
        let num_partitions = partition_clusters(
            &mut partition_dest,
            &cluster_indices,
            &cluster_index_counts,
            total_vertices,
            TARGET_PARTITION_SIZE,
        );
        
        println!("  Partitioned into {} groups", num_partitions);
        
        // Group meshlets by partition
        let mut partition_groups: Vec<Vec<usize>> = vec![Vec::new(); num_partitions];
        for (meshlet_idx, &partition_id) in partition_dest.iter().enumerate() {
            partition_groups[partition_id as usize].push(meshlet_idx);
        }
        
        let mut next_level_geometry: Vec<MeshletGeometry> = Vec::new();
        let mut next_level_node_indices: Vec<u32> = Vec::new();
        
        print_progress(&format!("Building LOD {}", lod_level), 0, num_partitions);
        
        for (group_idx, group) in partition_groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            
            // Merge geometry from all meshlets in this group
            let mut merged_positions: Vec<Vector3<f32>> = Vec::new();
            let mut merged_vertices: Vec<Vertex> = Vec::new();
            let mut merged_indices: Vec<u32> = Vec::new();
            let mut child_node_indices: Vec<u32> = Vec::new();
            
            for &meshlet_idx in group {
                let geom = &current_level_geometry[meshlet_idx];
                let node_idx = current_level_node_indices[meshlet_idx];
                child_node_indices.push(node_idx);
                
                let vertex_offset = merged_positions.len() as u32;
                
                merged_positions.extend(&geom.positions);
                merged_vertices.extend(geom.vertices.iter().cloned());
                
                // Remap indices with offset
                for &idx in &geom.indices {
                    merged_indices.push(idx + vertex_offset);
                }
            }
            
            // Flatten positions for meshopt
            let merged_positions_flat: Vec<f32> = merged_positions.iter()
                .flat_map(|v| [v.x, v.y, v.z])
                .collect();
            let merged_adapter = make_vertex_adapter(&merged_positions_flat);
            
            // Simplify the merged geometry
            // Target: reduce to fit in MAX_TRIANGLES for a single meshlet
            // We aim for MAX_TRIANGLES * 3 indices, but may need less
            let current_tri_count = merged_indices.len() / 3;
            let target_indices = (MAX_TRIANGLES * 3).min(merged_indices.len());
            
            let simplified_indices = simplify(
                &merged_indices,
                &merged_adapter,
                target_indices,
                f32::MAX,
                SimplifyOptions::empty(),
                None,
            );
            
            if simplified_indices.is_empty() {
                // Simplification failed, propagate children unchanged
                for &meshlet_idx in group {
                    next_level_geometry.push(current_level_geometry[meshlet_idx].clone());
                    next_level_node_indices.push(current_level_node_indices[meshlet_idx]);
                }
                continue;
            }
            
            // Check if simplified geometry fits in single meshlet constraints
            // Find unique vertices used in simplified mesh
            let mut used_vertices: Vec<u32> = simplified_indices.clone();
            used_vertices.sort();
            used_vertices.dedup();
            
            let simplified_vertex_count = used_vertices.len();
            let simplified_tri_count = simplified_indices.len() / 3;
            
            // If simplified result still exceeds limits, we need to re-meshletize
            if simplified_vertex_count > MAX_VERTICES || simplified_tri_count > MAX_TRIANGLES {
                // Re-meshletize the simplified geometry
                let sub_meshlets = build_meshlets(
                    &simplified_indices,
                    &merged_adapter,
                    MAX_VERTICES,
                    MAX_TRIANGLES,
                    0.5,
                );
                
                // Create a node for each sub-meshlet, all sharing the same children
                for (sub_idx, _) in sub_meshlets.meshlets.iter().enumerate() {
                    let sub_meshlet = sub_meshlets.get(sub_idx);
                    
                    // Extract geometry for this sub-meshlet
                    let mut sub_positions: Vec<Vector3<f32>> = Vec::new();
                    let mut sub_vertices: Vec<Vertex> = Vec::new();
                    
                    for &vid in sub_meshlet.vertices {
                        let idx = vid as usize;
                        if idx < merged_positions.len() {
                            sub_positions.push(merged_positions[idx]);
                            sub_vertices.push(merged_vertices[idx].clone());
                        }
                    }
                    
                    let sub_triangles = sub_meshlet.triangles.to_vec();
                    let sub_indices: Vec<u32> = sub_triangles.iter()
                        .map(|&t| t as u32)
                        .collect();
                    
                    // Compute bounds for sub-meshlet
                    let sub_positions_flat: Vec<f32> = sub_positions.iter()
                        .flat_map(|v| [v.x, v.y, v.z])
                        .collect();
                    let sub_adapter = make_vertex_adapter(&sub_positions_flat);
                    
                    // Build synthetic meshlet for bounds computation
                    let sub_verts_u32: Vec<u32> = (0..sub_positions.len() as u32).collect();
                    let sub_meshlet_for_bounds = meshopt::Meshlet {
                        vertices: &sub_verts_u32,
                        triangles: &sub_triangles,
                    };
                    let bounds = compute_meshlet_bounds(sub_meshlet_for_bounds, &sub_adapter);
                    
                    let parent_idx = all_nodes.len() as u32;
                    next_level_node_indices.push(parent_idx);
                    
                    // Update children to point to this parent
                    for &child_idx in &child_node_indices {
                        all_nodes[child_idx as usize].parent_index = Some(parent_idx);
                    }
                    
                    all_nodes.push(BnanMeshletDAGNode {
                        meshlet: BnanMeshletData {
                            position_offset: 0,
                            vertex_offset: 0,
                            vertex_count: sub_positions.len() as u32,
                            triangle_offset: 0,
                            triangle_count: (sub_triangles.len() / 3) as u32,
                        },
                        lod_level,
                        child_indices: child_node_indices.clone(),
                        parent_index: None,
                        bounds: Vector4::new(bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius),
                    });
                    
                    all_raw_data.push(BnanMeshletRawData {
                        positions: sub_positions.clone(),
                        vertices: sub_vertices.clone(),
                        triangles: sub_triangles,
                    });
                    
                    next_level_geometry.push(MeshletGeometry {
                        positions: sub_positions,
                        vertices: sub_vertices,
                        indices: sub_indices,
                    });
                }
            } else {
                // Simplified geometry fits in single meshlet!
                // Build vertex remap to compact the vertex buffer
                let mut vertex_remap: Vec<u32> = vec![u32::MAX; merged_positions.len()];
                let mut parent_positions: Vec<Vector3<f32>> = Vec::new();
                let mut parent_vertices: Vec<Vertex> = Vec::new();
                
                for (new_idx, &old_idx) in used_vertices.iter().enumerate() {
                    vertex_remap[old_idx as usize] = new_idx as u32;
                    parent_positions.push(merged_positions[old_idx as usize]);
                    parent_vertices.push(merged_vertices[old_idx as usize].clone());
                }
                
                // Remap indices to local
                let parent_triangles: Vec<u8> = simplified_indices.iter()
                    .map(|&idx| vertex_remap[idx as usize] as u8)
                    .collect();
                
                let parent_indices: Vec<u32> = parent_triangles.iter()
                    .map(|&t| t as u32)
                    .collect();
                
                // Compute bounds
                let parent_positions_flat: Vec<f32> = parent_positions.iter()
                    .flat_map(|v| [v.x, v.y, v.z])
                    .collect();
                let parent_adapter = make_vertex_adapter(&parent_positions_flat);
                
                let parent_verts_u32: Vec<u32> = (0..parent_positions.len() as u32).collect();
                let meshlet_for_bounds = meshopt::Meshlet {
                    vertices: &parent_verts_u32,
                    triangles: &parent_triangles,
                };
                let bounds = compute_meshlet_bounds(meshlet_for_bounds, &parent_adapter);
                
                let parent_idx = all_nodes.len() as u32;
                next_level_node_indices.push(parent_idx);
                
                // Update children to point to this parent
                for &child_idx in &child_node_indices {
                    all_nodes[child_idx as usize].parent_index = Some(parent_idx);
                }
                
                all_nodes.push(BnanMeshletDAGNode {
                    meshlet: BnanMeshletData {
                        position_offset: 0,
                        vertex_offset: 0,
                        vertex_count: parent_positions.len() as u32,
                        triangle_offset: 0,
                        triangle_count: simplified_tri_count as u32,
                    },
                    lod_level,
                    child_indices: child_node_indices,
                    parent_index: None,
                    bounds: Vector4::new(bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius),
                });
                
                all_raw_data.push(BnanMeshletRawData {
                    positions: parent_positions.clone(),
                    vertices: parent_vertices.clone(),
                    triangles: parent_triangles,
                });
                
                next_level_geometry.push(MeshletGeometry {
                    positions: parent_positions,
                    vertices: parent_vertices,
                    indices: parent_indices,
                });
            }
            
            print_progress(&format!("Building LOD {}", lod_level), group_idx + 1, num_partitions);
        }
        println!();
        
        if next_level_node_indices.is_empty() || next_level_node_indices.len() >= current_level_node_indices.len() {
            // No progress made, stop
            break;
        }
        
        current_level_geometry = next_level_geometry;
        current_level_node_indices = next_level_node_indices;
        lod_level += 1;
    }
    
    // Root indices are the final level
    let root_indices = current_level_node_indices;
    
    println!("DAG complete: {} nodes, {} LOD levels, {} roots", 
             all_nodes.len(), lod_level, root_indices.len());
    
    let dag = BnanMeshletDAG {
        nodes: all_nodes,
        root_indices,
        leaf_indices,
        max_lod_level: lod_level - 1,
    };
    
    (dag, all_raw_data)
}

impl Clone for MeshletGeometry {
    fn clone(&self) -> Self {
        MeshletGeometry {
            positions: self.positions.clone(),
            vertices: self.vertices.clone(),
            indices: self.indices.clone(),
        }
    }
}
