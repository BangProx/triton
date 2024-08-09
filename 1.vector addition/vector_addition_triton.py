# coding up a Triton vector addition kernel

import triton
import triton.language as tl
import torch

@triton.jit # writing a kernel
def kernel_vector_addition(a_ptr, b_ptr, out_ptr, # a is a real pointer to the base of a or starting memory
                            num_elems: tl.constexpr, # constexpr means that the complier knows that this is fixed and not going to change
                            block_size : tl.constexpr): 
    
    pid = tl.program_id(axis=0) # what block are we
    # We need to compute the offsets to find out what chunk of the vectors are going to handle
    
    #tl.device_print("pid ", pid) # info for debugging aspect <- but cannot do string formatting stuffs
    # information above will be printed on the console while running.

    block_start = pid * block_size  # 0 * 2 = 0, 1 * 2 = 2 , ... = block size
    # This is the math that gives the offset to know where these threads are going to start
    # Threads in this block of this particular instance are going to be executing
    
    thread_offsets = block_start + tl.arange(0,block_size)
    # taking starting pointer from the block and setting up series of pointer
    # basically our thread pointer for which individual e lement they're working on respectively

    # we need a mask because we don't have a guarantee that total number of threads 
    # to exactly match the number of data lelements we are working on
    mask = thread_offsets < num_elems

    # load up a and b data, add and store them back out
    a_pointers = tl.load(a_ptr + thread_offsets, mask = mask)
    # with the thread offset, we are going to set "a pointer"
    # start at the a pointer and add our thread offsets
    
    b_pointers = tl.load(b_ptr + thread_offsets, mask = mask)

    result = a_pointers + b_pointers
    tl.store(out_ptr + thread_offsets, result, mask = mask)


# function that quickly chunk up our task into block size and creae a grid
def ceil_div(x: int, y: int)-> int:
    return ((x+y-1)//y)

# take a, b vector
def vector_addition(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    output_buffer = torch.empty_like(a) # assumption that a, b have the same size
    assert a.is_cuda and b.is_cuda
    
    # why do we have to assert? 
    # This confirms our inputs & outputs is going to be on cuda == Datas are processed on GPU
    num_elems = a.numel() # # of elements so that we can determine our masking and block size
    assert num_elems == b.numel() # todo - handle mismatched sizes

    # how to create block size to chunk up this problem
    # each block will work on subset or subchunk
    block_size = 1024 #128 # total block size has to exceed the actual work size
    # need to make grid and ceiling division -> this is the point when we need to write the ceil_div function

    grid_size = ceil_div(num_elems, block_size) 
    # We can replace the ceil_div function with the meta parameters
    grid = (grid_size,) # make the grid tuple
    num_warps = 8 # this is a meta parameter which compiler picks
                    # defualt is 4 : block is divided into 4; 32 threads for each warp; so if we use 8, it'll give it a finer division and have the ability to  swap out warps
                    # waiting on memory to be loaded and get performance gain

    # We need to actually call the kernel
    # Information about what the kernel register spills 
    k2 = kernel_vector_addition[grid](a, b, output_buffer, 
                                      num_elems,
                                      block_size,
                                      num_warps = num_warps,)
    
    return output_buffer

def verify_numerics() -> bool:
    # verify numerical fidelity
    torch.manual_seed(2024) # seed both cpu and gpu
    vec_size = 8192
    a = torch.rand(vec_size, device = "cuda")
    b = torch.rand_like(a) # clone a; same size and device preserved with random data in it
    torch_res = a + b # answer; reference
    triton_res = vector_addition(a, b)
    fidelity_correct = torch.allclose(torch_res,triton_res)
    print(f"{fidelity_correct=}")

#if __name__ == '__main__':
#    verify_numerics()

## Once we verify our triton kernel, we are certain that there are no differences between torch and triton in terms of fidelity.

@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'], # x-axis name  for the plot
            x_vals = [2**i for i in range(12,28,1)], # diff possible values for 'x_name'
            x_log = True, # x axis is logarithmatic
            line_arg = 'provider', # arg name corresponds to different line in the plot
            line_vals=['triton','torch'], # possible values for 'line_arg'
            line_names = ['Triton','Torch'], #Label name
            styles=[('blue', '-'), ('green', '-')], # Line style
            ylabel = 'GB/s', # y-axis label name
            plot_name = 'vector-add-performance', # name for plot, used as file name when plot is saved
            args = {}, #values for function args not in 'x_names' and 'y_names'
            ))

def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_addition(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True)#, save_path='./vec_add_perf'