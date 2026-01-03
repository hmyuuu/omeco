//! # omeco - Tensor Network Contraction Order Optimization
//!
//! A Rust library for optimizing tensor network contraction orders, ported from
//! the Julia package [OMEinsumContractionOrders.jl](https://github.com/TensorBFS/OMEinsumContractionOrders.jl).
//!
//! This crate focuses on finding contraction orders and reporting complexity
//! metrics. It does not perform tensor contraction itself.
//!
//! ## Background
//!
//! A tensor network is a graph-like representation of a multilinear computation.
//! Each tensor is a node, and each shared index is an edge (a hyperedge can connect
//! more than two tensors). Contracting the network means summing over shared
//! indices to evaluate the final result. The order of pairwise contractions
//! has a major impact on time and memory costs.
//!
//! Contraction orders are represented as binary trees: leaves are input tensors,
//! internal nodes are intermediate contractions. Finding the optimal order is
//! NP-complete, so practical tools focus on good heuristics.
//!
//! ## Quick Start
//!
//! ```rust
//! use omeco::{EinCode, GreedyMethod, contraction_complexity, optimize_code, uniform_size_dict};
//!
//! let code = EinCode::new(
//!     vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
//!     vec!['i', 'l'],
//! );
//! let sizes = uniform_size_dict(&code, 16);
//!
//! let optimized = optimize_code(&code, &sizes, &GreedyMethod::default())
//!     .expect("optimizer failed");
//! let metrics = contraction_complexity(&optimized, &sizes, &code.ixs);
//! println!("time: 2^{:.2}", metrics.tc);
//! println!("space: 2^{:.2}", metrics.sc);
//! ```
//!
//! ## Algorithms
//!
//! ### GreedyMethod
//!
//! Repeatedly contracts the pair of tensors with the lowest cost. Time complexity
//! is O(n² log n) for n tensors.
//!
//! The cost function is: `loss = size(output) - alpha * (size(input1) + size(input2))`
//!
//! Parameters:
//! - `alpha`: Balances output size vs input size reduction (0.0 to 1.0)
//! - `temperature`: Enables stochastic selection via Boltzmann sampling (0.0 = deterministic)
//!
//! ### TreeSA
//!
//! Simulated annealing on contraction trees. Starts from an initial tree,
//! applies local rewrites, and accepts/rejects changes via Metropolis criterion.
//! Runs multiple trials in parallel (using rayon) and returns the best result.
//!
//! Parameters:
//! - `betas`: Inverse temperature schedule
//! - `ntrials`: Number of parallel trials (control threads via `RAYON_NUM_THREADS`)
//! - `niters`: Iterations per temperature level
//! - `score`: Scoring function balancing time, space, and read-write complexity
//!
//! Use [`GreedyMethod`] when you need speed; use [`TreeSA`] when contraction cost
//! dominates and you can afford extra search time.
//!
//! ## Complexity Metrics
//!
//! Three metrics are computed (all in log2 scale):
//!
//! - **Time Complexity (tc)**: Total FLOP count
//! - **Space Complexity (sc)**: Maximum intermediate tensor size
//! - **Read-Write Complexity (rwc)**: Total I/O operations
//!
//! ## Slicing
//!
//! Slicing reduces peak memory by looping over selected indices, trading extra
//! work for a smaller intermediate footprint. Use [`SlicedEinsum`] and
//! [`sliced_complexity`] to model this trade-off.
//!
//! ```rust
//! use omeco::{EinCode, GreedyMethod, SlicedEinsum, optimize_code, sliced_complexity, uniform_size_dict};
//!
//! let code = EinCode::new(
//!     vec![vec!['i', 'j'], vec!['j', 'k']],
//!     vec!['i', 'k'],
//! );
//! let sizes = uniform_size_dict(&code, 64);
//!
//! let optimized = optimize_code(&code, &sizes, &GreedyMethod::default())
//!     .expect("optimizer failed");
//! let sliced = SlicedEinsum::new(vec!['j'], optimized);
//!
//! let metrics = sliced_complexity(&sliced, &sizes, &code.ixs);
//! println!("sliced space: 2^{:.2}", metrics.sc);
//! ```

pub mod complexity;
pub mod eincode;
pub mod expr_tree;
pub mod greedy;
pub mod incidence_list;
pub mod label;
pub mod score;
pub mod treesa;
pub mod utils;

// Re-export main types
pub use complexity::{
    eincode_complexity, flop, nested_complexity, nested_flop, peak_memory, sliced_complexity,
    ContractionComplexity,
};
pub use eincode::{log2_size_dict, uniform_size_dict, EinCode, NestedEinsum, SlicedEinsum};
pub use greedy::{optimize_greedy, ContractionTree, GreedyMethod, GreedyResult};
pub use label::Label;
pub use score::ScoreFunction;
pub use treesa::{optimize_treesa, Initializer, TreeSA};

use std::collections::HashMap;

/// Trait for contraction order optimizers.
pub trait CodeOptimizer {
    /// Optimize the contraction order for an EinCode.
    fn optimize<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
    ) -> Option<NestedEinsum<L>>;
}

impl CodeOptimizer for GreedyMethod {
    fn optimize<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
    ) -> Option<NestedEinsum<L>> {
        optimize_greedy(code, size_dict, self)
    }
}

impl CodeOptimizer for TreeSA {
    fn optimize<L: Label>(
        &self,
        code: &EinCode<L>,
        size_dict: &HashMap<L, usize>,
    ) -> Option<NestedEinsum<L>> {
        optimize_treesa(code, size_dict, self)
    }
}

/// Optimize the contraction order for an EinCode using the specified optimizer.
///
/// # Example
///
/// ```rust
/// use omeco::{EinCode, optimize_code, GreedyMethod};
/// use std::collections::HashMap;
///
/// let code = EinCode::new(
///     vec![vec!['i', 'j'], vec!['j', 'k']],
///     vec!['i', 'k']
/// );
///
/// let mut sizes = HashMap::new();
/// sizes.insert('i', 10);
/// sizes.insert('j', 20);
/// sizes.insert('k', 10);
///
/// let optimized = optimize_code(&code, &sizes, &GreedyMethod::default());
/// assert!(optimized.is_some());
/// ```
pub fn optimize_code<L: Label, O: CodeOptimizer>(
    code: &EinCode<L>,
    size_dict: &HashMap<L, usize>,
    optimizer: &O,
) -> Option<NestedEinsum<L>> {
    optimizer.optimize(code, size_dict)
}

/// Compute the contraction complexity of an optimized NestedEinsum.
///
/// This is a convenience function that wraps [`nested_complexity`].
pub fn contraction_complexity<L: Label>(
    code: &NestedEinsum<L>,
    size_dict: &HashMap<L, usize>,
    original_ixs: &[Vec<L>],
) -> ContractionComplexity {
    nested_complexity(code, size_dict, original_ixs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_code_greedy() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
            vec!['i', 'l'],
        );

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 8);
        sizes.insert('k', 8);
        sizes.insert('l', 4);

        let result = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(result.is_some());

        let nested = result.unwrap();
        assert!(nested.is_binary());
        assert_eq!(nested.leaf_count(), 3);
    }

    #[test]
    fn test_optimize_code_treesa() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 8);
        sizes.insert('k', 4);

        let result = optimize_code(&code, &sizes, &TreeSA::fast());
        assert!(result.is_some());

        let nested = result.unwrap();
        assert!(nested.is_binary());
    }

    #[test]
    fn test_contraction_complexity_wrapper() {
        let code = EinCode::new(vec![vec!['i', 'j'], vec!['j', 'k']], vec!['i', 'k']);

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 8);
        sizes.insert('k', 4);

        let result = optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap();
        let complexity = contraction_complexity(&result, &sizes, &code.ixs);

        assert!(complexity.tc > 0.0);
        assert!(complexity.sc > 0.0);
    }

    #[test]
    fn test_single_tensor() {
        let code = EinCode::new(vec![vec!['i', 'j']], vec!['i', 'j']);

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 4);

        let result = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(result.is_some());
        assert!(result.unwrap().is_leaf());
    }

    #[test]
    fn test_trace() {
        let code = EinCode::new(
            vec![vec!['i', 'j'], vec!['j', 'i']],
            vec![], // Trace - no output
        );

        let mut sizes = HashMap::new();
        sizes.insert('i', 4);
        sizes.insert('j', 4);

        let result = optimize_code(&code, &sizes, &GreedyMethod::default());
        assert!(result.is_some());
    }
}
