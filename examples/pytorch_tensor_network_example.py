"""
Example: Using omeco to optimize einsum contraction order with PyTorch.

Workflow:
1. Define einsum notation and input tensors
2. Use omeco to optimize the contraction order
3. Contract using PyTorch with the optimized order
4. Verify correctness against PyTorch native einsum
"""

import torch
from omeco import optimize_greedy, contraction_complexity, uniform_size_dict


def contract_with_tree(tree: dict, tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Contract tensors according to the optimized tree.
    
    Args:
        tree: Contraction tree from NestedEinsum.to_dict()
        tensors: List of input tensors
    
    Returns:
        Contracted result tensor
    """
    if "tensor_index" in tree:
        # Leaf node: return the tensor at the given index
        return tensors[tree["tensor_index"]]
    else:
        # Internal node: contract children
        args = [contract_with_tree(arg, tensors) for arg in tree["args"]]
        ixs = tree["eins"]["ixs"]
        iy = tree["eins"]["iy"]
        return einsum(ixs, iy, args)


def einsum(ixs: list[list[int]], iy: list[int], tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Perform einsum with integer index labels.
    
    Args:
        ixs: Input index lists (e.g., [[0, 1], [1, 2]])
        iy: Output indices (e.g., [0, 2])
        tensors: Input tensors
    
    Returns:
        Result of einsum contraction
    """
    # Map integer labels to ASCII characters for torch.einsum
    all_labels = set(sum(ixs, []) + iy)
    label_map = {l: chr(ord('a') + i) for i, l in enumerate(sorted(all_labels))}
    
    inputs = ",".join("".join(label_map[l] for l in ix) for ix in ixs)
    output = "".join(label_map[l] for l in iy)
    einsum_str = f"{inputs}->{output}"
    
    return torch.einsum(einsum_str, *tensors)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # =========================================================================
    # Step 1: Define einsum notation and input tensors
    # =========================================================================
    # Matrix chain multiplication: A(i,j) × B(j,k) × C(k,l) × D(l,m) → (i,m)
    ixs = [[0, 1], [1, 2], [2, 3], [3, 4]]  # Index labels for each tensor
    out = [0, 4]  # Output indices
    
    # Create random tensors
    dims = {0: 100, 1: 50, 2: 80, 3: 60, 4: 100}
    A = torch.randn(dims[0], dims[1], device=device)
    B = torch.randn(dims[1], dims[2], device=device)
    C = torch.randn(dims[2], dims[3], device=device)
    D = torch.randn(dims[3], dims[4], device=device)
    tensors = [A, B, C, D]

    print("Step 1: Define einsum notation")
    print(f"  ixs = {ixs}")
    print(f"  out = {out}")
    print(f"  Tensor shapes: {[t.shape for t in tensors]}")

    # =========================================================================
    # Step 2: Use omeco to optimize contraction order
    # =========================================================================
    tree = optimize_greedy(ixs, out, dims)
    complexity = contraction_complexity(tree, ixs, dims)

    print(f"\nStep 2: Optimize contraction order with omeco")
    print(f"  Time complexity: 2^{complexity.tc:.2f} FLOPs")
    print(f"  Space complexity: 2^{complexity.sc:.2f} elements")
    print(f"  Tree: {tree}")

    # =========================================================================
    # Step 3: Contract using PyTorch with optimized order
    # =========================================================================
    tree_dict = tree.to_dict()
    result_optimized = contract_with_tree(tree_dict, tensors)

    print(f"\nStep 3: Contract with optimized order")
    print(f"  Result shape: {result_optimized.shape}")

    # =========================================================================
    # Step 4: Verify against PyTorch native einsum
    # =========================================================================
    result_native = torch.einsum("ij,jk,kl,lm->im", A, B, C, D)
    
    max_diff = torch.max(torch.abs(result_optimized - result_native)).item()
    rel_diff = max_diff / torch.max(torch.abs(result_native)).item()
    print(f"\nStep 4: Verify correctness")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")
    print(f"  Results match (rtol=1e-4): {rel_diff < 1e-4}")


if __name__ == "__main__":
    main()
