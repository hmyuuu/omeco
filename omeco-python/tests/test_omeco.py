"""Tests for omeco Python bindings."""

import pytest
from omeco import (
    GreedyMethod,
    TreeSA,
    optimize_greedy,
    optimize_treesa,
    contraction_complexity,
    sliced_complexity,
    SlicedEinsum,
    uniform_size_dict,
)


def test_optimize_greedy_basic():
    """Test basic greedy optimization."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    assert tree is not None
    assert tree.is_binary()
    assert tree.leaf_count() == 2


def test_optimize_greedy_chain():
    """Test greedy optimization on a chain."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = {0: 10, 1: 20, 2: 20, 3: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    assert tree.leaf_count() == 3
    assert tree.depth() >= 1


def test_optimize_treesa():
    """Test TreeSA optimization."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_treesa(ixs, out, sizes, TreeSA.fast())
    assert tree is not None
    assert tree.is_binary()


def test_contraction_complexity():
    """Test complexity computation."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    complexity = contraction_complexity(tree, ixs, sizes)
    
    assert complexity.tc > 0
    assert complexity.sc > 0
    assert complexity.flops() > 0
    assert complexity.peak_memory() > 0


def test_sliced_einsum():
    """Test sliced einsum."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    sliced = SlicedEinsum([1], tree)
    
    assert sliced.num_slices() == 1
    assert 1 in sliced.slicing()
    
    complexity = sliced_complexity(sliced, ixs, sizes)
    assert complexity.sc > 0


def test_uniform_size_dict():
    """Test uniform size dictionary creation."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    
    sizes = uniform_size_dict(ixs, out, 16)
    assert sizes[0] == 16
    assert sizes[1] == 16
    assert sizes[2] == 16


def test_greedy_method_params():
    """Test GreedyMethod with parameters."""
    opt = GreedyMethod(alpha=0.5, temperature=1.0)
    
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes, opt)
    assert tree is not None


def test_treesa_config():
    """Test TreeSA configuration methods."""
    opt = TreeSA().with_sc_target(10.0).with_ntrials(2)
    
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 4, 1: 8, 2: 4}
    
    tree = optimize_treesa(ixs, out, sizes, opt)
    assert tree is not None


def test_to_dict_leaf():
    """Test to_dict for a single tensor (leaf node)."""
    ixs = [[0, 1]]
    out = [0, 1]
    sizes = {0: 10, 1: 20}
    
    tree = optimize_greedy(ixs, out, sizes)
    d = tree.to_dict()
    
    # Single tensor should be a leaf
    assert "tensor_index" in d
    assert d["tensor_index"] == 0


def test_to_dict_binary():
    """Test to_dict for a binary contraction."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    d = tree.to_dict()
    
    # Should be a node with args and eins
    assert "args" in d
    assert "eins" in d
    assert len(d["args"]) == 2
    
    # Check eins structure
    assert "ixs" in d["eins"]
    assert "iy" in d["eins"]
    assert len(d["eins"]["ixs"]) == 2
    
    # Children should be leaves
    for arg in d["args"]:
        assert "tensor_index" in arg


def test_to_dict_chain():
    """Test to_dict for a chain of contractions."""
    ixs = [[0, 1], [1, 2], [2, 3]]
    out = [0, 3]
    sizes = {0: 10, 1: 20, 2: 20, 3: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    d = tree.to_dict()
    
    # Should be a node
    assert "args" in d
    assert "eins" in d
    
    # Count leaves by recursion
    def count_leaves(node):
        if "tensor_index" in node:
            return 1
        return sum(count_leaves(arg) for arg in node["args"])
    
    assert count_leaves(d) == 3


def test_to_dict_indices():
    """Test that to_dict preserves correct indices."""
    ixs = [[0, 1], [1, 2]]
    out = [0, 2]
    sizes = {0: 10, 1: 20, 2: 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    d = tree.to_dict()
    
    # Output should match
    assert d["eins"]["iy"] == out
    
    # Input indices should be the original tensor indices
    input_ixs = d["eins"]["ixs"]
    assert input_ixs == ixs
