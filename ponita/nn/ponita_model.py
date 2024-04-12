import jax
import jax.numpy as jnp
import flax.linen as nn

from ponita.nn.embedding import PolynomialFeatures
from ponita.nn.convnext import SeparableFiberBundleConvNext
from ponita.utils.geometry.rotations import GridGenerator


class Ponita(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    output_dim_vec: int = 0
    radius: float = None
    num_ori: int = 20
    basis_dim: int = None
    degree: int = 3
    widening_factor: int = 4
    layer_scale: float = None
    task_level: str = 'graph'
    multiple_readouts: bool = True
    batch_size: int = 2

    def setup(self):
        # Create grid
        # TODO: make it a functional, this does not need to be a module/class
        self.grid_generator = GridGenerator(self.num_ori, steps=1000)
        self.ori_grid = self.grid_generator()

        # Input output settings
        self.global_pooling = self.task_level == 'graph'

        # Activation function to use internally
        act_fn = nn.gelu

        # Kernel basis functions and spatial window
        basis_dim = self.hidden_dim if (self.basis_dim is None) else self.basis_dim
        self.basis_fn = nn.Sequential([PolynomialFeatures(self.degree), nn.Dense(self.hidden_dim), act_fn, nn.Dense(basis_dim), act_fn])
        self.fiber_basis_fn = nn.Sequential([PolynomialFeatures(self.degree), nn.Dense(self.hidden_dim), act_fn, nn.Dense(basis_dim), act_fn])
        
        # Initial node embedding
        # Paper sets bias positional encoding to false. 
        self.x_embedder = nn.Dense(self.hidden_dim, use_bias=False)
        
        # Make feedforward network
        interaction_layers = []
        read_out_layers = []
        for i in range(self.num_layers):
            interaction_layers.append(SeparableFiberBundleConvNext(self.hidden_dim, basis_dim, activation=act_fn, layer_scale_val=self.layer_scale, widening_factor=self.widening_factor))
            if self.multiple_readouts or i == (self.num_layers - 1):
                read_out_layers.append(nn.Dense(self.output_dim + self.output_dim_vec))
            else:
                read_out_layers.append(None)
        self.interaction_layers = interaction_layers
        self.read_out_layers = read_out_layers

            
    def __call__(self, x, pos, edge_index, batch):
        ori_grid = self.ori_grid.astype(pos.dtype)
        edge_index = edge_index.astype(jnp.int8)

        # Compute the invariants
        pos_send, pos_receive = pos[edge_index[0]], pos[edge_index[1]]                          # [num_edges, 3]
        rel_pos = (pos_send - pos_receive)                                                      # [num_edges, 3]
        rel_pos = rel_pos[:, None, :]                                                           # [num_edges, 1, 3]
        ori_grid_a = ori_grid[None,:,:]                                                         # [1, num_ori, 3]
        ori_grid_b = ori_grid[:, None,:]                                                        # [num_ori, 1, 3]
        invariant1 = (rel_pos * ori_grid_a).sum(axis=-1, keepdims=True)                         # [num_edges, num_ori, 1]
        invariant2 = jnp.linalg.norm(rel_pos - invariant1 * ori_grid_a, axis=-1, keepdims=True) #.norm(axis=-1, keepdims=True)   # [num_edges, num_ori, 1]
        invariant3 = (ori_grid_a * ori_grid_b).sum(axis=-1, keepdims=True)                      # [num_ori, num_ori, 1]
        spatial_invariants = jnp.concatenate([invariant1, invariant2], axis=-1)                 # [num_edges, num_ori, 2]
        orientation_invariants = invariant3                                                     # [num_ori, num_ori, 1]
    
        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(spatial_invariants)                              # [num_edges, num_ori, basis_dim]
        fiber_kernel_basis = self.fiber_basis_fn(orientation_invariants)              # [num_ori, num_ori, basis_dim]
        
        # Initial feature embeding
        x = self.x_embedder(x)
        x = jnp.expand_dims(x, axis=-2).repeat(ori_grid.shape[-2], axis=-2)  # [B,N,O,C]

        # Interaction + readout layers
        readouts = []
        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            x = interaction_layer(x, kernel_basis, fiber_kernel_basis, edge_index)
            if readout_layer is not None: readouts.append(readout_layer(x))
        readout = sum(readouts) / len(readouts)
        
        # Read out the scalar and vector part of the output
        if self.output_dim_vec > 0:
            readout_scalar, readout_vec = jnp.split(readout, jnp.array([self.output_dim, self.output_dim_vec]), axis=-1)
        else:
            readout_scalar, readout_vec = readout, None

        # Read out scalar and vector predictions
        output_scalar = jnp.mean(readout_scalar, axis=-2)                                          # [B,N,C]
        if self.output_dim_vec > 0:
            output_vector = jnp.einsum('boc,od->bcd', readout_vec, ori_grid) / ori_grid.shape[-2]  # [B,N,C,3]
        else:
            output_vector = None

        if self.global_pooling:
    
            pooled_out_scalar = jnp.zeros((self.batch_size, self.output_dim))
            pooled_out_scalar = pooled_out_scalar.at[batch].add(output_scalar)
            output_scalar = pooled_out_scalar

            if self.output_dim_vec > 0:
                pooled_out_vector = jnp.zeros((self.batch_size, self.output_dim_vec, 3))
                pooled_out_vector = pooled_out_vector.at[batch].add(output_vector)
                output_vector = pooled_out_vector
        return output_scalar, output_vector
    


class FullyConnectedPonita(nn.Module):
    """ Steerable E(3) equivariant (non-linear) convolutional network """
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    output_dim_vec: int = 0
    radius: float = None
    num_ori: int = 20
    basis_dim: int = None
    degree: int = 3
    widening_factor: int = 4
    layer_scale: float = None
    task_level: str = 'graph'
    multiple_readouts: bool = True
    last_feature_conditioning: bool = False,
    batch_size: int = 2

    def setup(self):
        # Create grid
        # TODO: make it a functional, this does not need to be a module/class
        self.grid_generator = GridGenerator(self.num_ori, dimension = 1, steps=1000)
        self.ori_grid = self.grid_generator()

        # Input output settings
        self.global_pooling = self.task_level == 'graph'

        # Activation function to use internally
        act_fn = nn.gelu

        # Kernel basis functions and spatial window
        basis_dim = self.hidden_dim if (self.basis_dim is None) else self.basis_dim
        self.basis_fn = nn.Sequential([PolynomialFeatures(self.degree), nn.Dense(self.hidden_dim), act_fn, nn.Dense(basis_dim), act_fn])
        self.fiber_basis_fn = nn.Sequential([PolynomialFeatures(self.degree), nn.Dense(self.hidden_dim), act_fn, nn.Dense(basis_dim), act_fn])
        
        # Initial node embedding
        # Paper sets bias positional encoding to false. 
        self.x_embedder = nn.Dense(self.hidden_dim, use_bias=False)
        
        # Make feedforward network
        interaction_layers = []
        read_out_layers = []
        for i in range(self.num_layers):
            interaction_layers.append(SeparableFiberBundleConvNext(self.hidden_dim, basis_dim, activation=act_fn, layer_scale_val=self.layer_scale, widening_factor=self.widening_factor, fully_connected=True))
            if self.multiple_readouts or i == (self.num_layers - 1):
                read_out_layers.append(nn.Dense(self.output_dim + self.output_dim_vec))
            else:
                read_out_layers.append(None)
        self.interaction_layers = interaction_layers
        self.read_out_layers = read_out_layers

        # Define initial mask
        self._mask_default = jnp.ones((1,1))

            
    def __call__(self, x, pos, mask, batch):
        ori_grid = self.ori_grid.astype(pos.dtype)
        # edge_mask = mask[:,None,:] * mask[:,:,None]                                             # [B, M, N]

        # Compute the invariants
        rel_pos = pos[:,None,:,None,:] - pos[:,:,None,None,:]                                   # [B,M,N,1,3] 
        ori_grid_a = ori_grid[None,:,:]                                                         # [1, num_ori, 3]
        ori_grid_b = ori_grid[:, None,:]                                                        # [num_ori, 1, 3]
        invariant1 = (rel_pos * ori_grid_a).sum(axis=-1, keepdims=True)                         # [num_edges, num_ori, 1]
        invariant2 = jnp.linalg.norm(rel_pos - invariant1 * ori_grid_a, axis=-1, keepdims=True) # [num_edges, num_ori, 1]
        invariant3 = (ori_grid_a * ori_grid_b).sum(axis=-1, keepdims=True)                      # [num_ori, num_ori, 1]

        # Concatenate the invariants
        spatial_invariants = jnp.concatenate([invariant1, invariant2], axis=-1)                 # [num_edges, num_ori, 2]
        orientation_invariants = invariant3                                                     # [num_ori, num_ori, 1]

        # Add noise level to the spatial invariants
        # if self.last_feature_conditioning:
        #     noise_levels = x[..., 0, -1][:, None, None, None, None].expand(-1, *spatial_invariants.shape[1:-1], -1)
        #     spatial_invariants = jnp.concatenate((spatial_invariants, noise_levels), dim=-1)
    
        # Sample the kernel basis and window the spatial kernel with a smooth cut-off
        kernel_basis = self.basis_fn(spatial_invariants)                              # [num_edges, num_ori, basis_dim]
        fiber_kernel_basis = self.fiber_basis_fn(orientation_invariants)              # [num_ori, num_ori, basis_dim]
        
        # Initial feature embedding
        x = self.x_embedder(x)
        x = jnp.expand_dims(x, axis=-2).repeat(ori_grid.shape[-2], axis=-2)  # [B,N,O,C]

        # Interaction + readout layers
        readouts = []
        for interaction_layer, readout_layer in zip(self.interaction_layers, self.read_out_layers):
            x = interaction_layer(x, kernel_basis, fiber_kernel_basis, mask)
            if readout_layer is not None: readouts.append(readout_layer(x))
        readout = sum(readouts) / len(readouts)
        
        # Read out the scalar and vector part of the output
        if self.output_dim_vec > 0:
            readout_scalar, readout_vec = jnp.split(readout, jnp.array([self.output_dim, self.output_dim_vec]), axis=-1)
        else:
            readout_scalar, readout_vec = readout, None

        # Read out scalar and vector predictions
        output_scalar = jnp.mean(readout_scalar, axis=-2)                                            # [B,N,C]
        if self.output_dim_vec > 0:
            output_vector = jnp.einsum('bnoc,od->bncd', readout_vec, ori_grid) / ori_grid.shape[-2]  # [B,N,C,3]
        else:
            output_vector = None

        if self.global_pooling:
            mask = self._mask_default if mask is None else mask
            output_scalar = jnp.sum(output_scalar * mask[:,:,None], axis=1) / jnp.sum(mask[:,:,None], axis=1)  # [B,C]

            # Divide by number of nodes
            # output_scalar = output_scalar_pooled / output_scalar.shape[1]

            if self.output_dim_vec > 0:
                output_vector = jnp.sum(output_vector * mask[:,:,None,None], axis=1) / jnp.sum(mask[:,:,None,None], axis=1)  # [B,C,3]
                # output_vector = (output_vector * mask[..., None, None]).sum(-1) / mask.sum(-1)[..., None, None]
        return output_scalar, output_vector
    