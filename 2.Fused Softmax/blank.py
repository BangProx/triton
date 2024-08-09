# code eager softmax and PyTorch, Triton

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

def naive_softmax(x: torch.Tensor)-> torch.Tensor:
    """ eager mode softmax"""
    # pull the maximum value and subtract that from everything -> numerator portion to avoid overflow or underflow
    x_max = x.max(dim = 1)[0] # max function returns values and indices. We only need values.
    safe_x = x - x_max[:,None] # x_max is 1D Tensor so we need to unsqueeze -> create a column Tensor
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1) # row-wise
    softmax_out = numerator / denominator[:,None] # unsqueeze
    return softmax_out

# Triton을 사용하기 위해서는 2가지가 필요하다. Kernel과 Driver Program.
# Driver Program은 block size, 어떻게 병렬화할지 등의 메타정보를 세팅해준다.

@triton.jit # 이 데코레이터를 사용하면 Triton Compiler 사용을 강제한다. -> Triton kernel임을 알려주는 데코레이터
def _softmax_fwd_kernel(
    output_ptr, 
    stride_output_row, 
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
):
    # setup input ptr before starting
    row_index = tl.program_id(0)  # dim = 0

    row_start_prt = input_ptr + (row_index * stride_input_row)# input_ptr is base and add stride x row index. So in order of row 3, we need to jump 3 rows which is 3 strides to go to proper row.

    # we need to know how many columns do we have here. -> based on block size which is what we're chunking along and 
    col_offsets = tl.arange(0,block_size) # tells how many particular chunks of the row to process 
    input_pointers = row_start_prt + col_offsets # <- full access to the entire row.

    row_mask = col_offsets < num_cols # 

    # pick up the row from HBM and move it to sram
    row = tl.load(input_pointers, mask = row_mask, other = float("-inf"))

    # start softmax
    safe_row = row - tl.max(row, axis = 0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator,axis = 0)
    softmax_out = numerator / denominator

    # Pointer arithmetic to set things up and write back to the output buffer
    # moving data from SRAM to HBM or global memory.
    output_ptr_row = output_ptr + (row_index * stride_output_row)
    output_pointers = output_ptr_row + col_offsets
    tl.store(output_pointers, softmax_out,mask = row_mask)
    


def softmax(x:torch.Tensor) -> torch.Tensor:
    """ Triton impl of Softmax, fwd pass only """
    # 여기서 포인터 연산을 사용한다. 
    # 이를 통해서 섬세하고 자세한 메모리 접근을 하고 이를 통해 좋은 성능을 발휘
    rows, cols = x.shape
    assert x.dim() == 2, f"only accepts 2D tensors for now" # 아직 batch는 고려하지 않기 때문에 2차원으로 고정
    
    # get in meta related informations -> how we'll gonna set up the kerenl for this particular kernel. parallelize along the rows. 
    # So each row will become its own kernel instance that will get passed in into our kerenl 
    
    # block size is related to the chunk so we are parallelizing along the rows but we need to then chunk out how we are going to handle the row 
    block_size = triton.next_power_of_2(cols)
    # We would end up having last chunk that has additional space to handle.
    # So we need to mask the kernel to ensure we're actually only accessing valid data not the entire we've allocated here in the block size. 
    num_warps = 4 # * 32 threads
    if block_size > 2047: # 2048
        num_warps = 8
    if block_size > 4095: # 4096
        num_warps = 16

    grid = (rows,)

    # allocate output buffer
    softmax_out = torch.empty_like(x) # duplicate copy of x

    # we need to pass stride to acknowledge kernel
    _softmax_fwd_kernel[grid](
        softmax_out, # output
        # stride -> everything in memory is not the logical representation. e.g. 2 x 2 tensor is actuall laid out as 1D
        # and stride to move between rows is basically the width of one column that would get you to the next row.
        softmax_out.stride(0),
        x, # input
        x.stride(0),
        cols, # mask along the number of cols
        block_size = block_size,
        num_warps = num_warps,
        )
    
    return softmax_out

sample = torch.tensor([[1,2,3,4,5],[5,4,3,2,1]], dtype = torch.float32, device='cuda')
reference_out = F.softmax(sample, dim=1) # row-wise
print(f"{reference_out=}")

eager_out = naive_softmax(sample)
print(f"{eager_out=}")

triton_out = softmax(sample)
print(f"{triton_out=}")