# Algorithms

This crate currently provides two optimizers: `GreedyMethod` and `TreeSA`. Both
return a `NestedEinsum` that represents a binary contraction tree.

## GreedyMethod

GreedyMethod repeatedly contracts the pair of tensors with the lowest cost.
The cost function is:

```
loss = size(output) - alpha * (size(input1) + size(input2))
```

where `alpha` is a tunable hyperparameter in `[0.0, 1.0]`. When `alpha` is 0.0,
the optimizer prefers the smallest output tensor at each step. Setting
`temperature > 0.0` enables stochastic selection using Boltzmann sampling.

Time complexity is O(n^2 log n) for n tensors because each step selects a pair
from a priority queue.

## TreeSA

TreeSA uses simulated annealing on contraction trees. It starts from an initial
tree (greedy or random), applies local tree rewrites, and accepts or rejects
changes based on the Metropolis criterion.

The scoring function balances time, space, and read-write complexity:

```
score = tc_weight * 2^tc
      + rw_weight * 2^rwc
      + sc_weight * max(0, 2^sc - 2^sc_target)
```

Key parameters:

- `betas`: inverse temperature schedule.
- `ntrials`: number of parallel trials (rayon).
- `niters`: iterations per temperature level.
- `initializer`: greedy or random.
- `decomposition_type`: tree (default) or path (linear order).

TreeSA typically finds better contraction orders than greedy search, but it is
slower. For large networks, `TreeSA::fast()` provides a smaller schedule with
fewer trials.

## Choosing an optimizer

- Use `GreedyMethod` when you need speed and can accept lower-quality orders.
- Use `TreeSA` when contraction cost dominates and you can afford extra search.

