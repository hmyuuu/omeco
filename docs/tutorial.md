# Tutorial

This tutorial walks through the main workflow: represent an einsum, pick an
optimizer, and inspect complexity metrics.

## 1. Build an einsum and optimize

```rust
use omeco::{EinCode, GreedyMethod, optimize_code, contraction_complexity, uniform_size_dict};

let code = EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
    vec!['i', 'l'],
);
let sizes = uniform_size_dict(&code, 16);

let optimized = optimize_code(&code, &sizes, &GreedyMethod::default())
    .expect("optimizer failed");

let metrics = contraction_complexity(&optimized, &sizes, &code.ixs);
println!("time: 2^{:.2}", metrics.tc);
println!("space: 2^{:.2}", metrics.sc);
println!("read-write: 2^{:.2}", metrics.rwc);
```

## 2. Tune TreeSA for higher-quality orders

```rust
use omeco::{EinCode, ScoreFunction, TreeSA, optimize_code, uniform_size_dict};

let code = EinCode::new(
    vec![vec!['a', 'b'], vec!['b', 'c'], vec!['c', 'd'], vec!['d', 'e']],
    vec!['a', 'e'],
);
let sizes = uniform_size_dict(&code, 32);

let score = ScoreFunction::new(1.0, 2.0, 0.0, 18.0);
let treesa = TreeSA { score, ..TreeSA::default() };

let optimized = optimize_code(&code, &sizes, &treesa)
    .expect("optimizer failed");
```

TreeSA runs trials in parallel using rayon. You can control the thread count
via the `RAYON_NUM_THREADS` environment variable.

## 3. Slice indices to reduce memory

Slicing trades time for space by iterating over one or more indices.

```rust
use omeco::{EinCode, GreedyMethod, SlicedEinsum, sliced_complexity, optimize_code, uniform_size_dict};

let code = EinCode::new(
    vec![vec!['i', 'j'], vec!['j', 'k']],
    vec!['i', 'k'],
);
let sizes = uniform_size_dict(&code, 64);
let optimized = optimize_code(&code, &sizes, &GreedyMethod::default())
    .expect("optimizer failed");

let sliced = SlicedEinsum::new(vec!['j'], optimized);
let metrics = sliced_complexity(&sliced, &sizes, &code.ixs);
println!("sliced space: 2^{:.2}", metrics.sc);
```

`SlicedEinsum` represents the sliced contraction plan. The actual execution of
sliced contractions is up to your tensor backend.

