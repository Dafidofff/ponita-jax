import jax
import flax.linen as nn
import jax.numpy as jnp

from ponita.nn.ponita_model import Ponita
from ponita.utils.graph import fully_connected_edge_index, subtract_mean


class EDMPrecond(nn.Module):
    model: nn.Module
    sigma_min: float = 0.0
    sigma_max: float = jnp.inf
    sigma_data: float = 0.5

    def setup(self):
        self.model = self.model

    def __call__(self, x, pos, edge_index, sigma):
        sigma = sigma.reshape(-1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = jnp.log(sigma) / 4

        # Simply condition by concatenating the noise level to the input
        x_in = jnp.concatenate([c_in * x, c_noise.expand(x.shape[0], 1)], axis=-1)
        if isinstance(self.model, Ponita):
            dx, dpos = self.model(x_in, (c_in * pos), edge_index)
            F_x = x - dx
            F_pos = pos - dpos[:,0,:]
        # elif isinstance(self.model, EGNN):
        #     dx, dpos = self.model(x_in, (c_in * pos), edge_index, None)
        #     F_x = dx
        #     F_pos = dpos
        else: # Other models not supported
            raise NotImplementedError
        
        # Noise dependent skip connection
        D_x = c_skip * x + c_out * F_x.astype(jnp.float32)
        D_pos = c_skip * pos + c_out * F_pos.astype(jnp.float32)
        return D_x, D_pos

    def round_sigma(self, sigma):
        return jnp.asarray(sigma)
    

class EDMLoss(nn.Module):
    P_mean: float = -1.2
    P_std: float = 1.2
    sigma_data: float = 0.5
    normalize_x_factor: float = 4.

    def __call__(self, net, inputs):
        # The point clouds and fully connected edge_index
        pos, x, edge_index, batch = inputs['pos'], inputs['x'], inputs['edge_index'], inputs['batch']
        edge_index = fully_connected_edge_index(batch)
        pos = subtract_mean(pos, batch)
        x = x / self.normalize_x_factor

        # Random noise level per point cloud in the batch
        rnd_normal = jax.random.normal(jax.random.PRNGKey(0), (jnp.max(batch) + 1, 1))
        rnd_normal = rnd_normal[batch]
        sigma = jnp.exp(rnd_normal * self.P_std + self.P_mean)
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Noised inputs
        x_noisy = x + jax.random.normal(jax.random.PRNGKey(0), x.shape) * sigma
        pos_noisy = pos + subtract_mean(jax.random.normal(jax.random.PRNGKey(0), pos.shape), batch) * sigma

        # The network net is a the precondioned version of the model, including noise dependent skip-connections
        D_x, D_pos = net(x_noisy, pos_noisy, edge_index, sigma)
        error_x = (D_x - x) ** 2
        error_pos = (D_pos - pos) ** 2
        loss = jnp.mean(weight * error_x) + jnp.mean(weight * error_pos)

        return loss, (D_x, D_pos)
