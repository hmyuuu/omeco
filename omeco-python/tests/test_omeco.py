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
    ixs = [['i', 'j'], ['j', 'k']]
    out = ['i', 'k']
    sizes = {'i': 10, 'j': 20, 'k': 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    assert tree is not None
    assert tree.is_binary()
    assert tree.leaf_count() == 2


def test_optimize_greedy_chain():
    """Test greedy optimization on a chain."""
    ixs = [['i', 'j'], ['j', 'k'], ['k', 'l']]
    out = ['i', 'l']
    sizes = {'i': 10, 'j': 20, 'k': 20, 'l': 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    assert tree.leaf_count() == 3
    assert tree.depth() >= 1


def test_optimize_treesa():
    """Test TreeSA optimization."""
    ixs = [['i', 'j'], ['j', 'k']]
    out = ['i', 'k']
    sizes = {'i': 10, 'j': 20, 'k': 10}
    
    tree = optimize_treesa(ixs, out, sizes, TreeSA.fast())
    assert tree is not None
    assert tree.is_binary()


def test_contraction_complexity():
    """Test complexity computation."""
    ixs = [['i', 'j'], ['j', 'k']]
    out = ['i', 'k']
    sizes = {'i': 10, 'j': 20, 'k': 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    complexity = contraction_complexity(tree, ixs, sizes)
    
    assert complexity.tc > 0
    assert complexity.sc > 0
    assert complexity.flops() > 0
    assert complexity.peak_memory() > 0


def test_sliced_einsum():
    """Test sliced einsum."""
    ixs = [['i', 'j'], ['j', 'k']]
    out = ['i', 'k']
    sizes = {'i': 10, 'j': 20, 'k': 10}
    
    tree = optimize_greedy(ixs, out, sizes)
    sliced = SlicedEinsum(['j'], tree)
    
    assert sliced.num_slices() == 1
    assert 'j' in sliced.slicing()
    
    complexity = sliced_complexity(sliced, ixs, sizes)
    assert complexity.sc > 0


def test_uniform_size_dict():
    """Test uniform size dictionary creation."""
    ixs = [['i', 'j'], ['j', 'k']]
    out = ['i', 'k']
    
    sizes = uniform_size_dict(ixs, out, 16)
    assert sizes['i'] == 16
    assert sizes['j'] == 16
    assert sizes['k'] == 16


def test_greedy_method_params():
    """Test GreedyMethod with parameters."""
    opt = GreedyMethod(alpha=0.5, temperature=1.0)
    
    ixs = [['i', 'j'], ['j', 'k']]
    out = ['i', 'k']
    sizes = {'i': 10, 'j': 20, 'k': 10}
    
    tree = optimize_greedy(ixs, out, sizes, opt)
    assert tree is not None


def test_treesa_config():
    """Test TreeSA configuration methods."""
    opt = TreeSA().with_sc_target(10.0).with_ntrials(2)
    
    ixs = [['i', 'j'], ['j', 'k']]
    out = ['i', 'k']
    sizes = {'i': 4, 'j': 8, 'k': 4}
    
    tree = optimize_treesa(ixs, out, sizes, opt)
    assert tree is not None

