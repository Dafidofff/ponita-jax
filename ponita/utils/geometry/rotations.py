import jax
import jax.numpy as jnp

from jax import random
from flax import linen as nn
import optax
from optax import sgd


class RandomSOd:
    def __init__(self, d):
        """
        Initializes the RandomRotationGenerator.
        Args:
        - d (int): The dimension of the rotation matrices (2 or 3).
        """
        assert d in [2, 3], "d must be 2 or 3."
        self.d = d

    def __call__(self, n=None):
        """
        Generates random rotation matrices.
        Args:
        - n (int, optional): The number of rotation matrices to generate. If None, generates a single matrix.

        Returns:
        - Array: An array of shape [n, d, d] containing n rotation matrices, or [d, d] if n is None.
        """
        if self.d == 2:
            return self._generate_2d(n)
        else:
            return self._generate_3d(n)
    
    def _generate_2d(self, n):
        theta = jax.random.uniform(jax.random.PRNGKey(0), (n,) if n else (1,)) * 2 * jnp.pi
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        rotation_matrix = jnp.stack([cos_theta, -sin_theta, sin_theta, cos_theta], axis=-1)
        if n:
            return rotation_matrix.reshape(n, 2, 2)
        return rotation_matrix.reshape(2, 2)

    def _generate_3d(self, n):
        q = jax.random.normal(jax.random.PRNGKey(0), (n, 4) if n else (4,))
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        rotation_matrix = jnp.stack([
            1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2),
            2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1),
            2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)
        ], axis=-1)
        if n:
            return rotation_matrix.reshape(n, 3, 3)
        return rotation_matrix.reshape(3, 3)


class GridGenerator(nn.Module):
    n: int
    dimension: int = 2
    steps: int = 200
    step_size: float = 0.01
    alpha: float = 0.001

    def setup(self):
        self.params = self.param('params', nn.initializers.zeros, (self.n, self.dimension + 1))

    def __call__(self):
        if self.dimension == 1:
            return self.uniform_grid_s1()
        elif self.dimension == 2:
            return self.repulse(self.random_s2())
        else:
            raise ValueError("Dimension must be either 1 (circle) or 2 (sphere).")

    def uniform_grid_s1(self):
        theta = jnp.linspace(0, 2 * jnp.pi, self.n+1)[:-1]
        x = jnp.cos(theta)
        y = jnp.sin(theta)
        return jnp.stack([x, y], axis=-1)

    def random_s2(self):
        points = random.normal(random.PRNGKey(0), (self.n, 3))
        points /= jnp.linalg.norm(points, axis=-1, keepdims=True)
        return points

    def repulse(self, points):
        points = jax.lax.stop_gradient(points)
        optimizer = sgd(self.step_size)
        opt_state = optimizer.init(points)

        def energy_loss(points):
            dists = jnp.sum(jnp.square(points[:, None] - points[None, :]), axis=-1)
            dists = jnp.linalg.norm(dists, axis=-1)
            dists = jax.lax.clamp(min=1e-6, x=dists, max=1e4)
            energy = jnp.sum(jnp.power(dists, -2))
            return energy
        
        for step in range(1, self.steps + 1):
            grads = jax.grad(energy_loss)(points)
            updates, opt_state = optimizer.update(grads, opt_state)
            points = optax.apply_updates(points, updates)
            # points /= jnp.linalg.norm(points, axis=-1, keepdims=True)
        return points

if __name__ == "__main__":
    # Example usage:
    # generator_circle = GridGenerator(n=20, dimension=1)
    # uniform_grid_circle = generator_circle()
    # dists = jnp.arccos(jnp.clip((uniform_grid_circle[:,None] * uniform_grid_circle[None,:]).sum(-1), -1., 1.))
    # min_dists = jnp.min(dists + 100*jnp.eye(dists.shape[0]), axis=-1)[0]
    # print("Circle:", uniform_grid_circle)
    # print("Circle uniformity:", min_dists.std())

    generator_sphere = GridGenerator(n=20, dimension=2)
    uniform_grid_sphere = generator_sphere()
    dists = jnp.arccos(jnp.clip((uniform_grid_sphere[:,None] * uniform_grid_sphere[None,:]).sum(-1), -1., 1.))
    min_dists = jnp.min(dists + 100*jnp.eye(dists.shape[0]), axis=-1)[0]
    print("Sphere:", uniform_grid_sphere)
    print(min_dists)
    print("Sphere uniformity:", min_dists.std())
