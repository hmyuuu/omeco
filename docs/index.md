# omeco

omeco is a Rust crate for optimizing tensor network contraction orders. It is a
port of the Julia package OMEinsumContractionOrders.jl.

This crate focuses on finding contraction orders and reporting complexity
metrics. It does not perform tensor contraction itself.

## Getting started

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
omeco = "0.1"
```

Define an einsum, sizes, and run an optimizer:

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
```

## Supported optimizers

- `GreedyMethod` for fast, deterministic or stochastic greedy search.
- `TreeSA` for simulated annealing with customizable scoring.

The Rust crate currently implements these two optimizers from the Julia
package. Other Julia optimizers are not yet available in this port.

## Documentation map

- `docs/tutorial.md` for step-by-step examples
- `docs/algorithms.md` for optimizer details and tuning
- `docs/reference.md` for a concise API reference

