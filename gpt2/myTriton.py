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
    
    flattened_input = input.unsqueeze(0) if input.ndim == 1 else input
    flattened_input = flattened_input.flatten(0, -2)
    batch_dim, feat_dim = flattened_input.shape
    # reshape 하는데 앞에 두개를 reshape 해서 kernel에 넣어주는 방식으로
    BLOCK_SIZE = triton.next_power_of_2(feat_dim)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096: # 2 ** 12
        num_warps = 16

    output = torch.empty_like(flattened_input)

    softmax_kernel[(batch_dim, )](flattened_input, output, flattened_input.stride(0),output.stride(0),
                                  feat_dim,num_warps=num_warps,BLOCK_SIZE=BLOCK_SIZE)

    return output.view_as(input)
    # n_a, n_b, n_c, n_d = x.shape
    # # x.shape=torch.Size([1, 12, 4, 4])
    # print(n_a, n_b, n_c, n_d)
    # print("====\n\n")
    # BLOCK_SIZE = triton.next_power_of_2(n_c * n_d)

    # num_warps = 4
    # if BLOCK_SIZE >= 2048:
    #     num_warps = 8
    # if BLOCK_SIZE >= 4096:
    #     num_warps = 16
    
    # y = torch.empty_like(x)

    # softmax_kernel[(n_a, )](y,x,x.stride(0),y.stride(0),n_c,num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE)
    # return y