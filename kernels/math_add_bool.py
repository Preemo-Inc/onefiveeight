import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=3,
                      num_warps=8),
        triton.Config({},num_stages=4,
                      num_warps=4),
        triton.Config({},num_stages=4,
                      num_warps=4),
        triton.Config({},num_stages=4,
                      num_warps=4),
        triton.Config({},num_stages=4,
                      num_warps=4),
        triton.Config({}, num_stages=4,
                      num_warps=4),
        triton.Config({},num_stages=5,
                      num_warps=2),
        triton.Config({},num_stages=5,
                      num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def matvec_add_kernel(Apos, Aneg, B, output, M, N):
    # Each program instance works on a different row
    row = tl.program_id(0)

    if row >= M:
        return

    # Calculate the starting index for this row in A
    row_start_index = row * N
    
    # Initialize the accumulator for the row
    acc = tl.zeros((), dtype=tl.float32)
    
    # We can improve performance by unrolling the loop and using software pipelining,
    # but this simple version follows the straightforward approach.
    for i in range(N):
        # Calculate index for the current element in A
        a_index = row_start_index + i
        apos = tl.load(Apos + a_index).to(tl.int1)
        aneg = tl.load(Aneg + a_index).to(tl.int1)
        b = tl.load(B + i)
        # We only add if a is not zero (i.e., -1 or 1)
        acc += tl.where(apos, b, tl.where(aneg, -b, 0))
    
    # Write the result to the output tensor
    tl.store(output + row, acc)

def matvec_addition(Apos: torch.Tensor, Aneg: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix-vector addition where the matrix contains values {-1, 0, 1}.
    This function acts as a wrapper around a Triton kernel.

    Parameters:
    - Apos: torch.Tensor, bool if the matrix contains +1.
    - Aneg: torch.Tensor, bool if the matrix contains -1.
    - B: torch.Tensor, the vector with floating point values.

    Returns:
    - torch.Tensor: The result of the matrix-vector addition.
    """
    # Get dimensions
    M, N = Apos.shape
    assert B.shape == (N,), "B must be a vector of length matching A's columns"
    assert Apos.shape == Aneg.shape, "Apos and Aneg must have the same shape"
    assert Apos.dtype == torch.bool, "Apos must be a boolean tensor"
    assert Aneg.dtype == torch.bool, "Aneg must be a boolean tensor"
    # Output tensor
    output = torch.empty(M, device=B.device, dtype=B.dtype)

    # Grid dimensions for the kernel launch
    
    grid = lambda meta: (M, )
    
    # Launch the Triton kernel
    matvec_add_kernel[grid](Apos, Aneg, B, output, M, N)

    return output

def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using Triton.

    Parameters:
    - A: torch.Tensor, the first matrix.
    - B: torch.Tensor, the second matrix.

    Returns:
    - torch.Tensor: The result of the matrix multiplication.
    """
    return A.to(torch.float32) @ B.to(torch.float32)
    

# Example usage
A = torch.tensor([[-1, 0, 1], [1, -1, 0], [0, 1, -1]], device='cuda', dtype=torch.int8)
B = torch.tensor([0.5, -0.2, 0.8], device='cuda', dtype=torch.float32)

# Perform the operation
def assert_equal(A, B):
    assert torch.allclose(matmul(A,B), matvec_addition((A== 1).bool(), (A==-1).bool(), B), atol=1e-4, rtol=1e-4)

assert_equal(A, B)
print("Success for small tensors.")

# write randomized A,B tensors
dim = 1000
A = torch.randint(-1, 2, (dim, dim), device='cuda', dtype=torch.int8)
B = torch.randn(dim, device='cuda', dtype=torch.float32)
assert_equal(A, B)

print("Success for large tensors.")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(2, 14)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'torch-native',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch (native)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')], # ('green', '--')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="add-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={},  # 'M': 4096 # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(N, provider):
    A = torch.randint(-1, 2, (N, N), device='cuda', dtype=torch.int8)
    Apos = A == 1
    Aneg = A == -1
    
    B = torch.randn(N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(A,B), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matvec_addition(Apos, Aneg, B), quantiles=quantiles)
    
    return ms, max_ms,  min_ms


benchmark.run(show_plots=True, print_data=True)
print("benchmark done.")