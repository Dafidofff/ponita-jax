import pytest

import jax
import jax.numpy as jnp

from ponita.nn.embedding import RFFNet, Layer, RFFEmbedding, PolynomialFeatures


class TestPolynomialFeatures:

    def test_polynomial_features_basic(self):
        # Simple case with degree 2
        degree = 2
        x = jnp.array([[1, 2], [3, 4]])  # Shape (2, 2)
        module = PolynomialFeatures(degree=degree)
        params = module.init(jax.random.PRNGKey(0), x)

        # Expected output calculation:
        # x^1 + x^2 for each element, but since x^2 is computed element-wise and then flattened,
        # for [1, 2] it would be [1, 2, 1^1, 2^1, 1^2, 2^2] -> [1, 2, 1, 4]
        expected_output = jnp.array([[[1, 2, 1, 2, 2, 4],[3, 4, 9, 12, 12, 16]]])
        output = module.apply(params, x)
        assert jnp.allclose(output, expected_output), "Output did not match expected polynomial features for degree 2."


class TestRFFNet:
    
    def test_output_shape_rffnet(self):
        model = RFFNet(output_dim=10, hidden_dim=20, num_layers=3, learnable_coefficients=False, std=1.0)
        x = jnp.ones((5, 10))  # Batch size of 5, input dimension of 10
        params = model.init(jax.random.PRNGKey(0), x)
        output = model.apply(params, x)
        assert output.shape == (5, 10), "Output shape should match (batch_size, output_dim)"

    def test_output_shape_layer(self):
        layer = Layer(hidden_dim=20)
        x = jnp.ones((5, 10))  # Batch size of 5, input dimension of 10
        params = layer.init(jax.random.PRNGKey(0), x)
        output = layer.apply(params, x)
        assert output.shape == (5, 20), "Output shape should match (batch_size, hidden_dim)"

    def test_even_hidden_dim_constraint_rffembedding(self):
        embedding = RFFEmbedding(hidden_dim=21, learnable_coefficients=False, std=1.0)
        x = jnp.ones((5, 10))  
        with pytest.raises(AssertionError):
            embedding.init(jax.random.PRNGKey(0), x)

    def test_output_shape_rffembedding(self):
        embedding = RFFEmbedding(hidden_dim=20, learnable_coefficients=False, std=1.0)
        x = jnp.ones((5, 10))  
        params = embedding.init(jax.random.PRNGKey(0), x)
        output = embedding.apply(params, x)
        assert output.shape == (5, 20), "Output shape should match (batch_size, hidden_dim)"