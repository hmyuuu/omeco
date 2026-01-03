# omeco - Python bindings

Python bindings for the omeco tensor network contraction order optimization library.

## Installation

### From PyPI (when published)

```bash
pip install omeco
```

### From source

```bash
# Install maturin
pip install maturin

# Build and install
cd omeco-python
maturin develop  # Development install
# or
maturin build --release  # Build wheel
```

## Quick Start

```python
from omeco import optimize_greedy, contraction_complexity, GreedyMethod

# Matrix chain: A[i,j] × B[j,k] × C[k,l] → D[i,l]
ixs = [['i', 'j'], ['j', 'k'], ['k', 'l']]
out = ['i', 'l']
sizes = {'i': 100, 'j': 200, 'k': 50, 'l': 100}

# Optimize contraction order
tree = optimize_greedy(ixs, out, sizes)

# Check complexity
complexity = contraction_complexity(tree, ixs, sizes)
print(f"Time: 2^{complexity.tc:.2f}")
print(f"Space: 2^{complexity.sc:.2f}")
```

## API

### Optimizers

- `optimize_greedy(ixs, out, sizes, optimizer=None)` - Greedy optimization
- `optimize_treesa(ixs, out, sizes, optimizer=None)` - Simulated annealing

### Classes

- `GreedyMethod(alpha=0.0, temperature=0.0)` - Greedy optimizer config
- `TreeSA()` - Simulated annealing config
  - `.fast()` - Fast configuration
  - `.with_sc_target(target)` - Set space complexity target
  - `.with_ntrials(n)` - Set number of parallel trials

### Complexity

- `contraction_complexity(tree, ixs, sizes)` - Compute complexity metrics
- `sliced_complexity(sliced, ixs, sizes)` - Complexity for sliced contraction

### Slicing

- `SlicedEinsum(indices, tree)` - Create sliced contraction plan

### Utilities

- `uniform_size_dict(ixs, out, size)` - Create uniform size dictionary

