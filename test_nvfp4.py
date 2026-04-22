import torch
from nvfp4_eval import quantize_nvfp4

def test_quantize_nvfp4():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32).cuda()
    block_size = 4
    q_x = quantize_nvfp4(x, block_size)
    print(f'Quantized NVFP4 x: {q_x}')
    assert q_x.shape == x.shape
    print('NVFP4 test passed!')

if __name__ == '__main__':
    test_quantize_nvfp4()
