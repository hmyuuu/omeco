"""
omeco - Tensor network contraction order optimization.

This package provides tools for optimizing tensor network contraction orders,
helping minimize computational cost (time and memory) when contracting tensors.

Example:
    >>> from omeco import optimize_greedy, contraction_complexity, GreedyMethod
    >>> 
    >>> # Matrix chain: A[i,j] × B[j,k] × C[k,l] → D[i,l]
    >>> ixs = [['i', 'j'], ['j', 'k'], ['k', 'l']]
    >>> out = ['i', 'l']
    >>> sizes = {'i': 100, 'j': 200, 'k': 50, 'l': 100}
    >>> 
    >>> tree = optimize_greedy(ixs, out, sizes)
    >>> complexity = contraction_complexity(tree, ixs, sizes)
    >>> print(f"Time: 2^{complexity.tc:.2f}, Space: 2^{complexity.sc:.2f}")
"""

from omeco._core import (
    # Classes
    NestedEinsum,
    SlicedEinsum,
    ContractionComplexity,
    GreedyMethod,
    TreeSA,
    # Functions
    optimize_greedy,
    optimize_treesa,
    contraction_complexity,
    sliced_complexity,
    uniform_size_dict,
)

__version__ = "0.1.0"
__all__ = [
    "NestedEinsum",
    "SlicedEinsum",
    "ContractionComplexity",
    "GreedyMethod",
    "TreeSA",
    "optimize_greedy",
    "optimize_treesa",
    "contraction_complexity",
    "sliced_complexity",
    "uniform_size_dict",
]

