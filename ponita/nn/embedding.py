import jax.numpy as jnp
from jax import lax
from flax import linen as nn
# from jax import jit


class PolynomialFeatures(nn.Module):
    degree: int

    def setup(self):
        pass  # No setup needed as there are no trainable parameters.

    def __call__(self, x):
        polynomial_list = [x]
        for it in range(1, self.degree+1):
            polynomial_list.append(jnp.einsum('...i,...j->...ij', polynomial_list[-1], x).reshape(*x.shape[:-1], -1))
        return jnp.concatenate(polynomial_list, axis=-1)


class RFFNet(nn.Module):
    output_dim: int
    hidden_dim: int
    num_layers: int
    learnable_coefficients: bool
    std: float
    numerator: float = 2.0

    def setup(self):
        assert (
            self.num_layers >= 2
        ), "At least two layers (the hidden plus the output one) are required."

        # Encoding
        self.encoding = RFFEmbedding(
            hidden_dim=self.hidden_dim,
            learnable_coefficients=self.learnable_coefficients,
            std=self.std,
        )

        # Hidden layers
        self.layers = [
            Layer(hidden_dim=self.hidden_dim, numerator=self.numerator)
            for _ in range(self.num_layers - 1)
        ]

        # Output layer
        self.linear_final = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(self.numerator, "fan_in", "uniform"),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )

    def __call__(self, x):
        x = self.encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.linear_final(x)
        return x


class Layer(nn.Module):
    hidden_dim: int
    numerator: float = 2.0

    def setup(self):
        self.linear = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(self.numerator, "fan_in", "normal"),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        self.activation = nn.relu

    def __call__(self, x):
        return self.activation(self.linear(x))


class RFFEmbedding(nn.Module):
    hidden_dim: int
    learnable_coefficients: bool
    std: float

    def setup(self):
        # Make sure we have an even number of hidden features.
        
        assert (
            not self.hidden_dim % 2.0
        ), "For the Fourier Features hidden_dim should be even to calculate them correctly."

        # Store pi
        self.pi = 2 * jnp.pi

        # Embedding layer
        self.coefficients = nn.Dense(
            self.hidden_dim // 2,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=self.std),
        )
        self.concat = lambda x: jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)

        if self.learnable_coefficients:
            self.parsed_coefficients = lambda x: self.coefficients(self.pi * x)
        else:
            self.parsed_coefficients = lambda x: lax.stop_gradient(self.coefficients(self.pi * x))

    def __call__(self, x):
        x_proj = self.parsed_coefficients(x)
        return self.concat(x_proj)