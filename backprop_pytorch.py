"""
PyTorch 自动求导对账验证
用同样的数据和参数，让 autograd 算梯度，和 numpy 手写版对比
"""

import numpy as np
import torch

np.random.seed(42)

# ==================== 数据（和 numpy 版完全一致）====================
X_np = np.array([[0.5, 0.8], [0.2, 0.3], [0.9, 0.7], [0.1, 0.4]])
y_np = np.array([[1], [0], [1], [0]])

W1_np = np.random.randn(2, 3) * 0.1
b1_np = np.zeros((1, 3))
W2_np = np.random.randn(3, 1) * 0.1
b2_np = np.zeros((1, 1))

# 转成 PyTorch 张量（参数需要梯度）
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)
W1 = torch.tensor(W1_np, dtype=torch.float32, requires_grad=True)
b1 = torch.tensor(b1_np, dtype=torch.float32, requires_grad=True)
W2 = torch.tensor(W2_np, dtype=torch.float32, requires_grad=True)
b2 = torch.tensor(b2_np, dtype=torch.float32, requires_grad=True)

# ==================== 前向传播 ====================
Z1 = X @ W1 + b1
A1 = torch.relu(Z1)
Z2 = A1 @ W2 + b2
A2 = torch.sigmoid(Z2)

loss = -torch.mean(y * torch.log(A2 + 1e-15) + (1 - y) * torch.log(1 - A2 + 1e-15))
print(f"PyTorch Loss: {loss.item():.6f}")

# ==================== 自动求导 ====================
loss.backward()

print("\n===== PyTorch 自动求导 =====")
print(f"W2.grad:\n{W2.grad.numpy()}")
print(f"W1.grad:\n{W1.grad.numpy()}")
