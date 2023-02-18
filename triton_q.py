from torch_int._CUDA import linear_a8_w8_b32_o32

from matmul_q import matmul_quant
import torch
torch.random.manual_seed(123)
seq_len = 3
c1 = 3
c2 = 3

dummy_input = torch.randint(-127, 127, (seq_len, c1), dtype=torch.int8).cuda()
weight = torch.randint(-127, 127, (c1, c2), dtype=torch.int8).cuda()
bias = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (c2,), dtype=torch.int32).cuda()
print(matmul_quant(dummy_input, weight, bias))

print(dummy_input.float() @ weight.float() + bias.float())

assert torch.allclose(dummy_input.float() @ weight.float() + bias.float(), matmul_quant(dummy_input, weight, bias).float(), atol=1e-2)

B, M, N = 128, 512, 1024
weight = torch.randint(-128, 127, (N, M), dtype=torch.int8, device="cuda")
bias = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, (N,), dtype=torch.int32, device="cuda")
x = torch.randint(-128, 127, (B, M), dtype=torch.int8, device="cuda")
linear = torch.nn.Linear(M, N, bias=True)
linear.weight.data = weight.float()
linear.bias.data = bias.float()
y_gt = linear(x.float())
y = linear_a8_w8_b32_o32(x, weight, bias)
assert torch.allclose(y_gt, y.float(), atol=1e-3)
y_triton = matmul_quant(x, weight.t(), bias)
assert torch.allclose(y_triton.float(), y.float(), atol=1e-3)
