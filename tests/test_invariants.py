import pytest
import jax.numpy as jnp

from ponita.utils.geometry.invariants import invariant_attr_rn


def test_basic_functionality_invariant_attr_rn():
    # Test a basic scenario
    pos = jnp.array([[0, 0], [3, 4]])  # Points at (0,0) and (3,4)
    edge_index = jnp.array([[0], [1]])  # A single edge from point 0 to point 1
    expected_distance = jnp.array([[5.0]])  # The Euclidean distance is 5
    assert jnp.allclose(invariant_attr_rn(pos, edge_index), expected_distance)

def test_empty_pos_or_edge_index_invariant_attr_rn():
    # NOTE: Jax returns type error with empty arrays, not index error
    # Test with an empty 'pos' array
    pos = jnp.array([])
    edge_index = jnp.array([[0], [1]])

    with pytest.raises(TypeError):
        invariant_attr_rn(pos, edge_index)  # Expecting an error due to indexing empty array

    # Test with an empty 'edge_index'
    pos = jnp.array([[0, 0], [3, 4]])
    edge_index = jnp.array([[], []])
    with pytest.raises(TypeError):
        invariant_attr_rn(pos, edge_index)

def test_mismatched_dimensions_invariant_attr_rn():
    # Test with pos and edge_index arrays that don't align properly
    pos = jnp.array([[0, 0], [3, 4]])  # 2 points
    edge_index = jnp.array([[0, 1], [1, 2]])  # Attempting to index 3 points
    assert jnp.allclose(invariant_attr_rn(pos, edge_index), jnp.array([[5.], [0.]]))

def test_non_euclidean_dimensions_invariant_attr_rn():
    # Test with points in a non-Euclidean (more than 2D) space
    pos = jnp.array([[0, 0, 0], [3, 4, 0]])  # Points in 3D space
    edge_index = jnp.array([[0], [1]])  # A single edge from point 0 to point 1
    expected_distance = jnp.array([[5.0]])  # The Euclidean distance is still 5
    assert jnp.allclose(invariant_attr_rn(pos, edge_index), expected_distance)
