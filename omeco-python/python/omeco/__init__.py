"""
omeco - Tensor network contraction order optimization.

This package provides tools for optimizing tensor network contraction orders,
helping minimize computational cost (time and memory) when contracting tensors.

Example:
    >>> from omeco import optimize_code, contraction_complexity, TreeSA
    >>>
    >>> # Matrix chain: A[0,1] × B[1,2] × C[2,3] → D[0,3]
    >>> ixs = [[0, 1], [1, 2], [2, 3]]
    >>> out = [0, 3]
    >>> sizes = {0: 100, 1: 200, 2: 50, 3: 100}
    >>>
    >>> tree = optimize_code(ixs, out, sizes, TreeSA.fast())
    >>> complexity = contraction_complexity(tree, ixs, sizes)
    >>> print(f"Time: 2^{complexity.tc:.2f}, Space: 2^{complexity.sc:.2f}")

Slicing to reduce memory:
    >>> from omeco import slice_code, sliced_complexity, TreeSASlicer
    >>> sliced = slice_code(tree, ixs, sizes, TreeSASlicer.fast().with_sc_target(10.0))
    >>> print(f"Sliced indices: {sliced.slicing()}")
    >>> c = sliced_complexity(sliced, ixs, sizes)
    >>> print(f"Sliced sc: 2^{c.sc:.2f}")

Using with PyTorch:
    >>> tree_dict = tree.to_dict()  # Convert to dict for traversal
    >>> # tree_dict structure:
    >>> # - Leaf: {"tensor_index": int}
    >>> # - Node: {"args": [...], "eins": {"ixs": [[int]], "iy": [int]}}
    >>>
    >>> # See examples/pytorch_tensor_network_example.py for complete usage
"""

from omeco._core import (
    # Classes
    NestedEinsum,
    SlicedEinsum,
    ContractionComplexity,
    GreedyMethod,
    TreeSA,
    TreeSASlicer,
    # Functions
    optimize_code,
    optimize_greedy,
    optimize_treesa,
    contraction_complexity,
    sliced_complexity,
    slice_code,
    uniform_size_dict,
)

__version__ = "0.1.1"
__all__ = [
    "NestedEinsum",
    "SlicedEinsum",
    "ContractionComplexity",
    "GreedyMethod",
    "TreeSA",
    "TreeSASlicer",
    "optimize_code",
    "optimize_greedy",
    "optimize_treesa",
    "contraction_complexity",
    "sliced_complexity",
    "slice_code",
    "uniform_size_dict",
]

