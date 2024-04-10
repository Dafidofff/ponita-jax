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
        else:
            raise NotImplementedError, ""
        
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


def edm_sampler(
    net,
    pos_0,
    x_0,
    edge_index,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=20,
    S_min=0,
    S_max=jnp.inf,
    S_noise=1,
    return_intermediate=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = jnp.maximum(sigma_min, net.sigma_min)
    sigma_max = jnp.minimum(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = jnp.arange(num_steps, dtype=jnp.float32)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = jnp.concatenate(
        [net.round_sigma(t_steps), jnp.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next, pos_next = x_0 * t_steps[0], pos_0 * t_steps[0]
    steps = [(x_next, pos_next)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur, pos_cur = x_next, pos_next

        # Increase noise temporarily.
        gamma = (jnp.minimum(S_churn / num_steps, jnp.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0)
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + jnp.sqrt(t_hat**2 - t_cur**2) * S_noise * jax.random.normal(jax.random.PRNGKey(0), x_cur.shape)
        pos_hat = pos_cur + jnp.sqrt(t_hat**2 - t_cur**2) * S_noise * jax.random.normal(jax.random.PRNGKey(0), pos_cur.shape)

        # Euler step.
        x_denoised, pos_denoised = net(x_hat, pos_hat, edge_index, t_hat)
        dx_cur = (x_hat - x_denoised) / t_hat
        dpos_cur = (pos_hat - pos_denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * dx_cur
        pos_next = pos_hat + (t_next - t_hat) * dpos_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_denoised, pos_denoised = net(x_next, pos_next, edge_index, t_next)
            dx_prime = (x_next - x_denoised) / t_next
            dpos_prime = (pos_next - pos_denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * dx_cur + 0.5 * dx_prime)
            pos_next = pos_hat + (t_next - t_hat) * (0.5 * dpos_cur + 0.5 * dpos_prime)

        steps.append((x_next, pos_next))

    if return_intermediate:
        return steps
    return x_next, pos_next
