import jax.numpy as jnp


def scatter_mean(src, index, dim, dim_size):
    # Step 1: Perform scatter add (sum)
    out_shape = [dim_size] + list(src.shape[1:])
    out_sum = jnp.zeros(out_shape, dtype=src.dtype)
    dims_to_add = src.ndim - index.ndim
    for _ in range(dims_to_add):
        index = jnp.expand_dims(index, -1)
    index_expanded = jnp.broadcast_to(index, src.shape)
    # out_sum = jax.ops.index_add(out_sum, jax.ops.index[index_expanded], src)
    out_sum = out_sum.at[jnp.index_exp[index_expanded]].add(src)
    
    # Step 2: Count occurrences of each index to calculate the mean
    ones = jnp.ones_like(src)
    out_count = jnp.zeros(out_shape, dtype=jnp.float32)
    # out_count = jax.ops.index_add(out_count, jax.ops.index[index_expanded], ones)
    out_count = out_count.at[jnp.index_exp[index_expanded]].add(ones)
    out_count = jnp.where(out_count == 0, 1, out_count)  # Avoid division by zero
    
    # Calculate mean by dividing sum by count
    out_mean = out_sum / out_count
    return out_mean

def fully_connected_edge_index(batch_idx):
    edge_indices = []
    for batch_num in jnp.unique(batch_idx):
        # Find indices of nodes in the current batch
        node_indices = jnp.where(batch_idx == batch_num)[0]
        grid = jnp.meshgrid(node_indices, node_indices, indexing='ij')
        edge_indices.append(jnp.stack([grid[0].reshape(-1), grid[1].reshape(-1)], axis=0))
    edge_index = jnp.concatenate(edge_indices, axis=1)
    return edge_index

def subtract_mean(pos, batch, batch_size=None):
    """NOTE: Jax does not like this batch.max().item() since it is not static. So better to provide a set batch_size."""
    if batch_size is None:
        batch_size = batch.max().item() + 1
    means = scatter_mean(src=pos, index=batch, dim=0, dim_size=batch_size)
    return pos - means[batch]