import torch
import triton
import triton.language as tl
from torch import Tensor
from triton import cdiv

#att = F.softmax(att, dim=-1)

@triton.jit
def softmax_kernel(input_ptr, output_ptr, 
                   input_row_stride, output_row_stride, 
                   n_cols, BLOCK_SIZE: tl.constexpr
                   ):
    
    batch_idx = tl.program_id(0)

    batch_start_ptr = input_ptr + batch_idx * input_row_stride
    col_offsets = tl.arange(0,BLOCK_SIZE)
    input_ptrs = batch_start_ptr + col_offsets
    
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other = -float('inf')) 
    row_minus_max = row - tl.max(row, axis = 0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis = 0)
    softmax_output = numerator / denominator

    # 최종 결과값을 DRAM에 Write back
    output_row_start_ptr = output_ptr + batch_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets

    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(input : Tensor) -> Tensor:
    
    reshaped_input = input.unsqueeze(0) if input.ndim == 1 else input
    reshaped_input = reshaped_input.flatten(0, -2)# reshaped_input.flatten(1)
    #print(f"{input.shape}")
    # if input.ndim == 4:
    #     reshaped_input = input.reshape(-1, input.size(2) * input.size(3))
    # elif input.ndim == 3:
    #     print("ndim=3")
    #     reshaped_input = input.reshape(-1,input.size(1)*input.size(2))
    # else:
    #     reshaped_input = input
    #reshaped_input = input.reshape(-1, input.size(-2) * input.size(-1))

    batch_dim, feat_dim = reshaped_input.shape
    # reshape 하는데 앞에 두개를 reshape 해서 kernel에 넣어주는 방식으로
    BLOCK_SIZE = triton.next_power_of_2(feat_dim)
    num_warps = 8
    # if BLOCK_SIZE >= 2048:
    #     num_warps = 8
    # if BLOCK_SIZE >= 4096: # 2 ** 12
    #     num_warps = 16

    output = torch.empty_like(reshaped_input)

    softmax_kernel[(batch_dim, )](reshaped_input, output, reshaped_input.stride(0),output.stride(0),
                                  feat_dim,num_warps=num_warps,BLOCK_SIZE=BLOCK_SIZE)

    return output.view_as(input)