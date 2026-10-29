"""
可视化：反向传播的梯度流动
画出 numpy 手写 vs PyTorch autograd 的梯度对比，证明两者一致
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.array([[0.5, 0.8], [0.2, 0.3], [0.9, 0.7], [0.1, 0.4]])
y = np.array([[1], [0], [1], [0]])
n = X.shape[0]

W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros((1, 1))

# numpy 前向 + 反向
Z1 = X @ W1 + b1
A1 = np.maximum(0, Z1)
Z2 = A1 @ W2 + b2
A2 = 1 / (1 + np.exp(-Z2))

dZ2 = (A2 - y) / n
dW2_np = A1.T @ dZ2
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0).astype(float)
dW1_np = X.T @ dZ1

# PyTorch 对账
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
W1_t = torch.tensor(W1, dtype=torch.float32, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float32, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

Z1_t = X_t @ W1_t + b1_t
A1_t = torch.relu(Z1_t)
Z2_t = A1_t @ W2_t + b2_t
A2_t = torch.sigmoid(Z2_t)
loss_t = -torch.mean(y_t * torch.log(A2_t + 1e-15) + (1 - y_t) * torch.log(1 - A2_t + 1e-15))
loss_t.backward()

dW1_torch = W1_t.grad.numpy()
dW2_torch = W2_t.grad.numpy()

# ==================== 图1：梯度对比 ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# dW1 对比
ax = axes[0]
x = np.arange(dW1_np.size)
width = 0.35
ax.bar(x - width/2, dW1_np.flatten(), width, label='numpy 手写', color='steelblue')
ax.bar(x + width/2, dW1_torch.flatten(), width, label='PyTorch autograd', color='coral')
ax.set_title('dW1 梯度对比（2x3=6个）')
ax.set_xlabel('参数编号')
ax.set_ylabel('梯度值')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# dW2 对比
ax = axes[1]
x = np.arange(dW2_np.size)
ax.bar(x - width/2, dW2_np.flatten(), width, label='numpy 手写', color='steelblue')
ax.bar(x + width/2, dW2_torch.flatten(), width, label='PyTorch autograd', color='coral')
ax.set_title('dW2 梯度对比（3x1=3个）')
ax.set_xlabel('参数编号')
ax.set_ylabel('梯度值')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/gradient_comparison.png', dpi=100, bbox_inches='tight')
print("已保存 images/gradient_comparison.png")

# ==================== 图2：差异验证 ====================
fig, ax = plt.subplots(figsize=(8, 4))
diffs = {
    'dW1': np.abs(dW1_np - dW1_torch).max(),
    'dW2': np.abs(dW2_np - dW2_torch).max(),
}
bars = ax.bar(diffs.keys(), diffs.values(), color=['steelblue', 'coral'])
ax.axhline(y=1e-6, color='r', linestyle='--', label='1e-6 阈值')
ax.set_title('numpy vs PyTorch 梯度最大差异')
ax.set_ylabel('最大绝对差异')
ax.set_yscale('log')
ax.legend()
for bar, v in zip(bars, diffs.values()):
    ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.2e}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('images/diff_verification.png', dpi=100, bbox_inches='tight')
print("已保存 images/diff_verification.png")
