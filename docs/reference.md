# Reference

This page summarizes the most important public APIs. For full API details,
refer to rustdoc.

## Core types

- `EinCode<L>`: Einsum representation using label type `L`.
- `NestedEinsum<L>`: Binary contraction tree.
- `SlicedEinsum<L>`: A contraction plan with sliced indices.
- `Label`: Trait implemented by `char`, `usize`, and other common label types.

## Optimizers

- `GreedyMethod`: Greedy pairwise contraction optimizer.
- `TreeSA`: Simulated annealing optimizer.
- `CodeOptimizer`: Trait implemented by optimizers.

## Key functions

- `optimize_code`: Optimize an `EinCode` with a chosen optimizer.
- `contraction_complexity`: Complexity for a `NestedEinsum`.
- `eincode_complexity`: Complexity for an `EinCode` without an order.
- `sliced_complexity`: Complexity for a `SlicedEinsum`.
- `uniform_size_dict`: Build a size dictionary with a single size.
- `log2_size_dict`: Convert sizes to log2 space for internal use.
- `flop`, `nested_flop`, `peak_memory`: Helpers for scalar metrics.

## Notes

- The crate does not perform tensor contraction; it provides order optimization
  and complexity evaluation.
- `TreeSA` uses rayon for parallel trials. Control threads with
  `RAYON_NUM_THREADS`.

