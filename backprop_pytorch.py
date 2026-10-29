"""
PyTorch autograd 对账验证
目标：用 PyTorch 重跑同样的网络，让 autograd 自动算梯度，和 numpy 手写版对比
"""

import numpy as np
import torch

np.random.seed(42)

# ==================== 数据（和 numpy 版完全一致，seed 相同）====================
X_np = np.array([[0.5, 0.8], [0.2, 0.3], [0.9, 0.7], [0.1, 0.4]])
y_np = np.array([[1], [0], [1], [0]])

W1_np = np.random.randn(2, 3) * 0.1
b1_np = np.zeros((1, 3))
W2_np = np.random.randn(3, 1) * 0.1
b2_np = np.zeros((1, 1))

# ==================== 转成 PyTorch 张量 ====================
# 数据不需要梯度，参数需要 requires_grad=True，autograd 才会追踪它们
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


# ==================== 自动求导 ====================
loss.backward()

# ==================== 打印结果 ====================
print("="*50)
print(f"Loss: {loss.item():.6f}")
print()
print("PyTorch autograd 梯度：")
print(f"W2.grad:\n{W2.grad.numpy()}")
print(f"b2.grad: {b2.grad.numpy()}")
print(f"W1.grad:\n{W1.grad.numpy()}")
print(f"b1.grad: {b1.grad.numpy()}")


# ==================== 对账：和 numpy 版比较 ====================
# 复跑 numpy 手写梯度
n = X_np.shape[0]
Z1_np = X_np @ W1_np + b1_np
A1_np = np.maximum(0, Z1_np)
Z2_np = A1_np @ W2_np + b2_np
A2_np = 1 / (1 + np.exp(-Z2_np))
dZ2 = (A2_np - y_np) / n
dW2_np = A1_np.T @ dZ2
db2_np = np.sum(dZ2, axis=0, keepdims=True)
dA1 = dZ2 @ W2_np.T
dZ1 = dA1 * (Z1_np > 0).astype(float)
dW1_np = X_np.T @ dZ1
db1_np = np.sum(dZ1, axis=0, keepdims=True)

print()
print("="*50)
print("对账验证（numpy vs PyTorch）：")
diffs = {
    "W2": np.abs(dW2_np - W2.grad.numpy()).max(),
    "b2": np.abs(db2_np - b2.grad.numpy()).max(),
    "W1": np.abs(dW1_np - W1.grad.numpy()).max(),
    "b1": np.abs(db1_np - b1.grad.numpy()).max(),
}
for name, diff in diffs.items():
    print(f"d{name} 最大差异: {diff:.2e}")

if max(diffs.values()) < 1e-6:
    print("\n[SUCCESS] 对账通过！手写反向传播完全正确。")
else:
    print("\n[FAILED] 对账失败，检查梯度公式。")
