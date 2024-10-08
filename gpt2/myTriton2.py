import torch
import triton
import triton.language as tl
from triton.runtime import driver

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows,n_cols,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # stride : how much we need to increase the pointer to advance 1 row
        # ===================================#
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(input):
    reshaped_input = input.unsqueeze(0) if input.ndim == 1 else input
    reshaped_input = input.flatten(0, -2) # 1
    n_rows, n_cols = reshaped_input.shape
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)    

    # we can ask the compiler to use more threads per row by 
    # increasing the # of warps -> autotune possible
    num_warps = 4

    # # of     sw pipeling stages
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    # allocate output y which has a same shape as x
    output = torch.empty_like(reshaped_input)

    # pre-compile kernel to get register usage and compute thread occupancy
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None,0))
    if kernel is None:
        kernel = softmax_kernel.warmup( output, reshaped_input, 
                                       reshaped_input.stride(0), output.stride(0), 
                                       n_rows, n_cols,
                                       BLOCK_SIZE = BLOCK_SIZE,
                                       num_stages = num_stages,
                                       num_warps = num_warps,
                                       grid = (1, )
                                       )
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            # NUM_REGS : # of regular purpose registers.
            # On CDNA architectures this is half of all registers available.
            # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs.
            # Accumulation VGPRs are used with matrix VALU instructions & can be directly loaded from memory
            # A wave may have up to 512 total VGPRs, 256 of each type.
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2
                # MAX_NUM_THREADS : max # of resident threads per multi-processor
                # MAX_NUM_THREADS / WARP_SIZE = max # of waves that can execute on multi-processor in parallel
                MAX_NUM_THREADS = properties["max_threads_per_sm"]
                max_num_waves = MAX_NUM_THREADS // WARP_SIZE
                occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        output,
        reshaped_input,
        reshaped_input.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
    )
    return output.view_as(input)