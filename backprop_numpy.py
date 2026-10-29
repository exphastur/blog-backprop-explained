"""
从零理解反向传播：numpy 手写实现
网络结构：输入2 → 隐藏层3（ReLU）→ 输出1（Sigmoid）
"""

import numpy as np

np.random.seed(42)

# ==================== 数据 ====================
X = np.array([[0.5, 0.8], [0.2, 0.3], [0.9, 0.7], [0.1, 0.4]])  # (4, 2)
y = np.array([[1], [0], [1], [0]])  # (4, 1)
n = X.shape[0]

# ==================== 参数初始化 ====================
W1 = np.random.randn(2, 3) * 0.1  # (2, 3)
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.1  # (3, 1)
b2 = np.zeros((1, 1))


# ==================== 激活函数 ====================
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ==================== 前向传播 ====================
Z1 = X @ W1 + b1    # (4, 3)
A1 = relu(Z1)       # (4, 3)
Z2 = A1 @ W2 + b2   # (4, 1)
A2 = sigmoid(Z2)    # (4, 1)

eps = 1e-15
loss = -np.mean(y * np.log(A2 + eps) + (1 - y) * np.log(1 - A2 + eps))

print(f"Loss: {loss:.6f}")


# ==================== 反向传播 ====================
# 输出层
dZ2 = (A2 - y) / n
dW2 = A1.T @ dZ2
db2 = np.sum(dZ2, axis=0, keepdims=True)

# 隐藏层
dA1 = dZ2 @ W2.T
dZ1 = dA1 * (Z1 > 0).astype(float)  # ReLU导数
dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0, keepdims=True)


print("\n===== numpy 梯度 =====")
print(f"dW2:\n{dW2}")
print(f"dW1:\n{dW1}")
