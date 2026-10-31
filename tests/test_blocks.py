import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để import được src/

import torch
from src.models.blocks import ConvBNReLU, SEBlock, ResidualBlock, init_weights

def n_params(m):
    return sum(p.numel() for p in m.parameters())

def check(name, module, x, expected_shape):
    module.apply(init_weights)
    y = module(x)
    ok = tuple(y.shape) == expected_shape
    print(f"[{'OK' if ok else 'LỖI'}] {name:<38} {tuple(x.shape)} -> {tuple(y.shape)}"
          f"   ({n_params(module):,} tham số)")
    assert ok, f"{name}: mong đợi {expected_shape}, nhận {tuple(y.shape)}"
    return y

torch.manual_seed(0)
B = 2  
print("=" * 80)

x = torch.randn(B, 4, 128, 128)
check("ConvBNReLU 4->32, stride 2 (Stem)", ConvBNReLU(4, 32, 3, 2), x, (B, 32, 64, 64))
check("ConvBNReLU 4->32, stride 1", ConvBNReLU(4, 32, 3, 1), x, (B, 32, 128, 128))

x = torch.randn(B, 64, 32, 32)
se = SEBlock(64)
y = check("SEBlock 64 kênh", se, x, (B, 64, 32, 32))
w = se.fc(se.pool(x))
print(f"      trọng số SE nằm trong [{w.min():.3f}, {w.max():.3f}] (phải trong 0..1)")
assert 0 <= w.min() and w.max() <= 1

x = torch.randn(B, 32, 64, 64)
check("ResidualBlock 32->32, stride 1", ResidualBlock(32, 32, 1), x, (B, 32, 64, 64))

check("ResidualBlock 32->64, stride 2", ResidualBlock(32, 64, 2), x, (B, 64, 32, 32))

blk = ResidualBlock(32, 64, 2)
x = torch.randn(B, 32, 64, 64, requires_grad=True)
blk(x).mean().backward()
ok = x.grad is not None and x.grad.abs().sum() > 0
print(f"[{'OK' if ok else 'LỖI'}] Gradient lan truyền ngược qua ResidualBlock")
assert ok

print("=" * 80)
print("Tất cả khối hoạt động đúng!")