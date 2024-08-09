import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel( x_ptr,  #ptr to 1st input vector
                y_ptr,  #ptr to 2nd input vector
                output_ptr, #ptr to output vector
                n_elements, #sie of the vector
                BLOCK_SIZE: tl.constexpr,   # #of elements each program should process
                # 'constexpr' so it can be uesd as a shape value
                ):
    # Multiple programs processing diff data -> identify programs
    pid = tl.program_id(axis=0) # 1D Launch grid so axis is 0
    # this program process inputs that are offset from the initial data
    # e.g. vector of 256, block_size of 64, the programs would each
    # access the elements [0:64, 64:128, 128:192, 192:256]
    # offsets are a list of pointers
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input
    # is not a multiple of the block sie
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y
    # Write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask = mask)

def add(x : torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # SPMD Launch grid denotes # of kernel instances that run in parallel
    # It is analogous to CUDA Launch grids -> Tuble[int], or Callable(metaparameters) -> Tuble[int]
    # In this case, we use a 1D grid where the size is the # of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    # - Each torch.tensor object is implicitly coverted to ptr to its 1st element
    # - 'triton.jit' ed functions can be indexed with a launch grid to obtain a callable GPU Kenrel
    # - Don't forget to pass meta-parameter as a kwargs
    add_kernel[grid](x,y,output,n_elements,BLOCK_SIZE = 1024)
    # We return a handle to z but, since 'torch.cuda.synchronize()' hasn't been called,
    # the kernel is running asynchronously at this point.
    return output

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device = 'cuda')
y = torch.rand(size, device = 'cuda')
output_torch = x + y
output_triton = add(x,y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
