# 从零理解反向传播：numpy vs autograd 数值对比

> 手写反向传播，和 PyTorch 自动求导对账验证（梯度差异 < 1e-9）

## 📖 完整博客

**👉 [阅读完整技术博客](blog.md)**

## 🎯 这个项目做了什么

- ✅ 纯 numpy 手写 2 层神经网络的反向传播
- ✅ PyTorch autograd 逐层对账验证
- ✅ 可视化梯度对比图表
- ✅ 完整的数学推导 + 代码实现

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行代码

```bash
# numpy 手写版
python backprop_numpy.py

# PyTorch 对账版
python backprop_pytorch.py

# 生成可视化图表
python visualize_gradients.py
```

### 3. 查看结果

图表保存在 `images/` 目录：
- `gradient_comparison.png` —— numpy vs PyTorch 梯度对比
- `diff_verification.png` —— 差异验证（< 1e-9）

## 📊 关键结果

| 参数 | numpy 梯度 | PyTorch 梯度 | 最大差异 |
|-----|-----------|-------------|---------|
| W1  | 手写计算   | autograd   | < 1e-9  |
| W2  | 手写计算   | autograd   | < 1e-9  |

**✅ 验证通过**：手写实现和 PyTorch 完全一致！

## 💡 核心洞察

1. **链式法则 = 逐层复用**：反向传播把 O(5^300) 降到 O(300)
2. **ReLU 导数 = 0/1 掩码**：`(Z > 0)` 就是导数
3. **矩阵维度推导**：`A1.T @ dZ2` 自动匹配 `dW2` 的 shape

## 🗂️ 项目结构

```
.
├── blog.md                    # 完整技术博客（核心）
├── backprop_numpy.py          # numpy 手写实现
├── backprop_pytorch.py        # PyTorch 对账验证
├── visualize_gradients.py     # 可视化脚本
├── requirements.txt           # 依赖
└── images/                    # 生成的图表
```

## 🎓 学习路线

这是 [ML From Scratch](https://github.com/你的用户名/ml-from-scratch) 系列的一部分：

1. ✅ 逻辑回归（手写梯度下降）
2. ✅ K-means 聚类
3. ✅ PCA 降维
4. ✅ 决策树
5. ✅ **反向传播（本项目）** ← 你在这里
6. 🔄 卷积神经网络（进行中）

## 📚 参考资料

- [CS231n: Backpropagation](http://cs231n.stanford.edu/slides/2024/lecture_4.pdf)
- [PyTorch Autograd 文档](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

## 📝 License

MIT

---

**作者**：Java 工程师转型 AI | [GitHub](https://github.com/你的用户名) | [博客](你的博客链接)
