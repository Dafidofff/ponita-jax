import jax
import jax.numpy as jnp
import flax.linen as nn

from ponita.nn.embedding import PolynomialFeatures
from ponita.utils.geometry.rotations import GridGenerator


class SepConvNextBlock(nn.Module):
    num_hidden: int
    basis_dim: int
    widening_factor: int = 4

    def setup(self):
        self.conv = SepConv(self.num_hidden, self.basis_dim)
        self.act_fn = nn.gelu
        self.linear_1 = nn.Dense(self.widening_factor * self.num_hidden)
        self.linear_2 = nn.Dense(self.num_hidden)
        self.norm = nn.LayerNorm()

    def __call__(self, x, kernel_basis, fiber_kernel_basis):
        input = x
        x = self.conv(x, kernel_basis, fiber_kernel_basis)
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        x = x + input
        return x


class SepConv(nn.Module):
    num_hidden: int
    basis_dim: int
    bias: bool = True

    def setup(self):
        # Set up kernel coefficient layers, one for the spatial kernel and one for the group kernel.
        # This maps from the invariants to a basis for the kernel 2/3->basis_dim.
        self.spatial_kernel = nn.Dense(self.num_hidden, use_bias=False)
        self.rotation_kernel = nn.Dense(self.num_hidden, use_bias=False)

        # Construct bias
        if self.bias:
            self.bias_param = self.param('bias', nn.initializers.zeros, (self.num_hidden,))

    def __call__(self, x, kernel_basis, fiber_kernel_basis):
        """ Perform separable convolution on fully connected pointcloud.

        Args:
            x: Array of shape (batch, num_points, num_ori, num_features)
            kernel_basis: Array of shape (batch, num_points, num_points, num_ori, basis_dim)
            fiber_kernel_basis: Array of shape (batch, num_points, num_points, basis_dim)
        """
        # Compute the spatial kernel
        spatial_kernel = self.spatial_kernel(kernel_basis)

        # Compute the group kernel
        rot_kernel = self.rotation_kernel(fiber_kernel_basis)

        # Perform the convolution
        x = jnp.einsum('bnoc,bmnoc->bmoc', x, spatial_kernel)
        x = jnp.einsum('bmoc,poc->bmpc', x, rot_kernel)

        # Add bias
        if self.bias:
            x = x + self.bias_param
        return x


class PonitaFixedSize(nn.Module):
    num_in: int
    num_hidden: int
    num_layers: int
    scalar_num_out: int
    vec_num_out: int

    spatial_dim: int
    num_ori: int
    basis_dim: int
    degree: int

    widening_factor: int
    global_pool: bool

    def setup(self):

        # Check input arguments
        assert self.spatial_dim in [2, 3], "spatial_dim must be 2 or 3."
        rot_group_dim = 1 if self.spatial_dim == 2 else 3

        # Create a grid generator for the dimensionality of the orientation
        self.grid_generator = GridGenerator(n=self.num_ori, dimension=rot_group_dim, steps=1000)
        self.ori_grid = self.grid_generator()

        # Set up kernel basis functions, one for the spatial kernel and one for the group kernel.
        # This maps from the invariants to a basis for the kernel 2/3->basis_dim.
        self.spatial_kernel_basis = nn.Sequential([
            PolynomialFeatures(degree=self.degree), nn.Dense(self.num_hidden), nn.gelu, nn.Dense(self.basis_dim), nn.gelu])
        self.rotation_kernel_basis = nn.Sequential([
            PolynomialFeatures(degree=self.degree), nn.Dense(self.num_hidden), nn.gelu, nn.Dense(self.basis_dim), nn.gelu])

        # Initial node embedding
        self.x_embedder = nn.Dense(self.num_hidden, use_bias=False)

        # Make feedforward network
        interaction_layers = []
        for i in range(self.num_layers):
            interaction_layers.append(SepConvNextBlock(self.num_hidden, self.basis_dim))
        self.interaction_layers = interaction_layers

        # Readout layers
        self.readout = nn.Dense(self.scalar_num_out + self.vec_num_out)

    @staticmethod
    def invariants_2d(pos, ori):
        """ Compute invariants for 2D positions.

        Args:
            pos: Array of shape (batch, num_points, 2)
            ori: Array of shape (num_ori, 2)

        Returns:
            spatial_invariants: Array of shape (batch, num_points, num_points, num_ori, 2)
            orientation_invariants: Array of shape (num_ori, num_ori, 1)
        """
        rel_pos = pos[:, None, :, None, :] - pos[:, :, None, None, :]  # (batch, num_points, num_points, 1, 2)

        invariant1 = (rel_pos[..., 0] * ori[..., 0] + rel_pos[..., 1] * ori[..., 1])
        invariant2 = (-rel_pos[..., 0] * ori[..., 1] + rel_pos[..., 1] * ori[..., 0])

        spatial_invariants = jnp.stack([invariant1, invariant2], axis=-1)
        orientation_invariants = (ori[:, None, :] * ori[None, :, :]).sum(axis=-1, keepdims=True)
        return spatial_invariants, orientation_invariants

    @staticmethod
    def invariants_3d(pos, ori):
        """ Compute invariants for 3D positions.

        Args:
            pos: Array of shape (batch, num_points, 3)
            ori: Array of shape (num_ori, 3)

        Returns:
            spatial_invariants: Array of shape (batch, num_points, num_points, num_ori, 2)
            orientation_invariants: Array of shape (num_ori, num_ori, 1)
        """
        rel_pos = pos[:, None, :, None, :] - pos[:, :, None, None, :]  # (batch, num_points, num_points, 1, 3)

        invariant1 = (rel_pos * ori[None, None, None, :, :]).sum(axis=-1, keepdims=True)  # (batch, num_points, num_points, num_ori, 1)
        invariant2 = (rel_pos - rel_pos * invariant1).norm(axis=-1, keepdims=True)  # (batch, num_points, num_points, 1, 3)
        spatial_invariants = jnp.concatenate([invariant1, invariant2], axis=-1)
        orientation_invariants = (ori[:, None, :] * ori[None, :, :]).sum(axis=-1, keepdims=True)
        return spatial_invariants, orientation_invariants

    def __call__(self, pos, x):
        """ Forward pass through the network.

        Args:
            pos: Array of shape (batch, num_points, spatial_dim)
            x: Array of shape (batch, num_points, num_in)
        """
        # Get invariants, shape (batch, num_points, num_ori, num_points, num_ori, num_in)
        spatial_invariants, rotation_invariants = self.invariants_2d(pos, self.ori_grid) if self.spatial_dim == 2 else self.invariants_3d(pos, self.ori_grid)

        # Sample the kernel basis
        kernel_basis = self.spatial_kernel_basis(spatial_invariants)
        fiber_kernel_basis = self.rotation_kernel_basis(rotation_invariants)

        # Initial feature embedding
        x = self.x_embedder(x)

        # Repeat features over the orientation dimension
        x = x[:, :, None, :].repeat(self.ori_grid.shape[-2], axis=-2)

        # Apply interaction layers
        for layer in self.interaction_layers:
            x = layer(x, kernel_basis, fiber_kernel_basis)

        # Readout layer
        readout = self.readout(x)

        # Split scalar and vector parts
        readout_scalar, readout_vec = jnp.split(readout, [self.scalar_num_out], axis=-1)

        # Average over the orientation dimension
        output_scalar = readout_scalar.mean(axis=-2)
        if self.vec_num_out > 0:
            output_vector = jnp.einsum('bnoc,od->bncd', readout_vec, self.ori_grid) / self.ori_grid.shape[-2]
        else:
            output_vector = None

        # Global pooling
        if self.global_pool:
            output_scalar = jnp.sum(output_scalar, axis=1) / output_scalar.shape[1]
            if self.vec_num_out > 0:
                output_vector = jnp.sum(output_vector, axis=1) / output_vector.shape[1]

        return output_scalar, output_vector


if __name__ == "__main__":
    model = PonitaFixedSize(
        num_in=3,
        num_hidden=64,
        num_layers=3,
        scalar_num_out=1,
        vec_num_out=1,
        spatial_dim=2,
        num_ori=4,
        basis_dim=64,
        degree=3,
        widening_factor=4,
        global_pool=True
    )

    # Create meshgrid
    pos = jnp.stack(jnp.meshgrid(jnp.linspace(-1, 1, 28), jnp.linspace(-1, 1, 28)), axis=-1).reshape(-1, 2)
    # Repeat over batch dim
    pos = jnp.repeat(pos[None, ...], 4, axis=0)
    x = jnp.ones((4, 28*28, 3))

    # Initialize and apply model
    params = model.init(jax.random.PRNGKey(0), pos, x)
    model.apply(params, pos, x)
    print("Success!")
