import warnings
import torch

def setup():
    torch.set_default_device('cuda')
    torch.set_float32_matmul_precision('medium')
    warnings.filterwarnings('ignore', message='Not enough SMs to use max_autotune_gemm mode')