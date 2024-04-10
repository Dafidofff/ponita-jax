from jax import vmap

import jax.numpy as jnp

def invariant_attr_rn(pos, edge_index):
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]] # [num_edges, n]
    dists = jnp.linalg.norm(pos_send - pos_receive, axis=-1, keepdims=True)    # [num_edges, 1]
    return dists

def invariant_attr_r3s2_fiber_bundle(pos, ori_grid, edge_index, separable=False):
    pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]                # [num_edges, 3]
    rel_pos = (pos_send - pos_receive)                                            # [num_edges, 3]

    # Convenient shape
    rel_pos = rel_pos[:, None, :]                                                 # [num_edges, 1, 3]
    ori_grid_a = ori_grid[None,:,:]                                               # [1, num_ori, 3]
    ori_grid_b = ori_grid[:, None,:]                                              # [num_ori, 1, 3]

    invariant1 = (rel_pos * ori_grid_a).sum(axis=-1, keepdims=True)                 # [num_edges, num_ori, 1]
    invariant2 = jnp.linalg.norm(rel_pos - invariant1 * ori_grid_a, axis=-1, keepdims=True)   # [num_edges, num_ori, 1]
    invariant3 = (ori_grid_a * ori_grid_b).sum(axis=-1, keepdims=True)              # [num_ori, num_ori, 1]
    
    if separable:
        return jnp.concatenate([invariant1, invariant2],axis=-1), invariant3             # [num_edges, num_ori, 2], [num_ori, num_ori, 1]
    else:
        invariant1 = jnp.broadcast_to(invariant1[:,:,None,:], (invariant1.shape[0], invariant1.shape[1], ori_grid.shape[0], 1))    # [num_edges, num_ori, num_ori, 1]
        invariant2 = jnp.broadcast_to(invariant2[:,:,None,:], (invariant2.shape[0], invariant2.shape[1], ori_grid.shape[0], 1))    # [num_edges, num_ori, num_ori, 1]
        invariant3 = jnp.broadcast_to(invariant3[None,:,:,:], (invariant1.shape[0], invariant3.shape[1], invariant3.shape[2], 1))  # [num_edges, num_ori, num_ori, 1]
        return jnp.concatenate([invariant1, invariant2, invariant3],axis=-1)             # [num_edges, num_ori, num_ori, 3]

def invariant_attr_r3s2_point_cloud(pos, edge_index):
    pos_send, pos_receive = pos[edge_index[0],:3], pos[edge_index[1],:3]          # [num_edges, 3]
    ori_send, ori_receive = pos[edge_index[0],3:], pos[edge_index[1],3:]          # [num_edges, 3]
    rel_pos = pos_send - pos_receive                                              # [num_edges, 3]

    invariant1 = jnp.sum(rel_pos * ori_receive, axis=-1, keepdims=True)
    invariant2 = jnp.linalg.norm(rel_pos - ori_receive * invariant1, axis=-1, keepdims=True)
    invariant3 = jnp.sum(ori_send * ori_receive, axis=-1, keepdims=True)

    return jnp.concatenate([invariant1, invariant2, invariant3],axis=-1)             # [num_edges, num_ori, num_ori, 3]
