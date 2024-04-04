# TODO: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
# Inspired by the amazing work on DeltaBit https://github.com/FasterDecoding/BitDelta/tree/main
import torch
import triton
import triton.language as tl


def pack(x: torch.Tensor):
    """
    Pack 4 ternary values into uint8.
    
    x: bool tensor (*, K, N)
    return: uint8 tensor (*, K // n_bits, N)
    """
    n_bits=4
    assert x.shape[-2] % n_bits == 0, "K must be divisible by n_bits"
    x = x.clone()
    x[x == -1] = 2

    shift = torch.arange(n_bits, device=x.device) * 2
    shape = x.shape[:-2]
    x = x.view(-1, x.shape[-2]//n_bits, n_bits, x.shape[-1])
    x = x << shift[None, None, :, None]
    x = x.sum(-2)
    x = x.view(*shape, *x.shape[-2:])

    return x.to(torch.uint8)



def unpack(x: torch.Tensor, n_bits=4):
    """
    unpack n_bits of x into a single integer
    
    x: int tensor (*, K // n_bits, N)
    return: bool tensor (*, K, N)
    """
    shift = torch.arange(n_bits, device=x.device) * 2
    shape = x.shape[:-2]
    x = x.view(-1, x.shape[-2], 1, x.shape[-1])
    x = (x >> shift[None, None, :, None]) & 0x3
    x = x.view(*shape, -1, x.shape[-1])
    # replace 2 with -1
    x = torch.where(x == 2, torch.tensor(-1, device=x.device), x)
    return x

import os
# os.environ["TRITON_INTERPRET"] = "1"

@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=8, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),

    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _ternary_mm_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), float16
    B has shape (K//n_bits, N), uint8, packed 
    C has shape (M, N),
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    n_bits = 4
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    GROUP_SIZE_M = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # Adapted from GPTQ-Triton (https://github.com/fpgaminer/GPTQ-triton)
    # b_ptrs is set up such that it repeats elements along the K axis n_bits times
    b_ptrs = b_ptr + ((offs_k[:, None] // n_bits) * stride_bk + offs_bn[None, :] * stride_bn)  
    # shifter is used to extract each bit of each element in the int matrix
    shifter = (offs_k % n_bits) * 2 # 
    # shifter = shifter 
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs) #, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        # b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)
        b = tl.load(b_ptrs)

        # Convert B from int to a.dtype, for each bit in B, 0 becomes -1.0, 1 becomes 1.0
        # b: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        b = (b >> shifter[:, None]) & 0x3
        # map the value 2 -> -1
        b = tl.where(b == 0x2, -1, b)
        b = b.to(a.dtype)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # accumulator += tl.where(b == 2.0, -a, 0.0)
        # accumulator += tl.where(b == 1.0, a, 0.0)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        # b_ptrs += BLOCK_SIZE_K * stride_bk
        b_ptrs += (BLOCK_SIZE_K // n_bits) * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def bitmat(a, b, activation=""):
    """
        a: float tensor (M, K)
        b: int tensor (K//n_bits, N)
        n_bits: int, number of bits that each element in b represents
    """
    # Check constraints.
    int_per_2_bits=4
    assert a.shape[1] == b.shape[0]* int_per_2_bits, "Incompatible dimensions"
    assert a.is_contiguous(), "A must be contiguous"
    assert b.is_contiguous(), "B must be contiguous"
    assert b.dtype == torch.uint8, "B must be a packed tenary->uint8 tensor"
    assert int_per_2_bits in [4, 8, 16, 32], "n_bits must be 4, 8, 16, 32"
    M, K = a.shape
    _, N = b.shape

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # print(f"Launching kernel with M = {M}, N = {N}, K = {K}, n_bits = {n_bits}, activation = {activation}")

    _ternary_mm_kernel[grid](
        a, b, c,
        M, N, K,
        # int_per_2_bits,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    try:
        c[0][0].item()
    except RuntimeError:
        raise RuntimeError(
            "Illegal memory access, it means that the kernel failed most probably to OOM, try to reduce batch size or matrix size.")

    return c


def matmul_f32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using Triton.

    Parameters:
    - A: torch.Tensor, the first matrix.
    - B: torch.Tensor, the second matrix.

    Returns:
    - torch.Tensor: The result of the matrix multiplication.
    """
    return (A.to(torch.float32) @ B.to(torch.float32)).to(A.dtype)

def matmul_f16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication using Triton.

    Parameters:
    - A: torch.Tensor, the first matrix.
    - B: torch.Tensor, the second matrix.

    Returns:
    - torch.Tensor: The result of the matrix multiplication.
    """
    return (A.to(torch.float16) @ B.to(torch.float16))

def main():

    # Example usage
    M, N, K = 32, 24, 128
    A = torch.rand((M,K), device='cuda', dtype=torch.float16) * 10
    B = torch.randint(-1, 2, (K, N), device='cuda', dtype=torch.int8)
    # Perform the operation
    def assert_equal(A, B):
        A = A.clone()
        B = B.clone()
        assert (B - unpack(pack(B)) == 0).all()
        assert torch.allclose(matmul_f32(A,B), bitmat(A, pack(B)), atol=1e-3, rtol=1e-3)

    assert_equal(A, B)
    print("Success for small tensors.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],  # argument names to use as an x-axis for the plot
            x_vals=[2**i for i in range(2, 12)],  # different possible values for `x_name`
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
            args={'M': 1024, "K":1024},  # 'M': 4096 # values for function arguments not in `x_names` and `y_name`
        ))
    def benchmark(M, N, K, provider):
        assert N % 4 == 0, "N must be a multiple of 4" 
        A = torch.rand((M,K), device='cuda', dtype=torch.float32)       
        B = torch.randint(-1, 2, (K, N), device='cuda', dtype=torch.int8)
        assert_equal(A.to(torch.float16), B)
        B_pack = pack(B)
        
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch-native':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_f16(A,B), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: bitmat(A, B_pack), quantiles=quantiles)
        
        return ms, max_ms,  min_ms


    benchmark.run(show_plots=False, print_data=True)
    print("benchmark done.")

if __name__ == "__main__":
    main()