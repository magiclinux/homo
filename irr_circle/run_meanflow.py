#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ==========================================
# 1. 数据生成部分 (irr_circle: 中心高斯 → 不规则环)
# ==========================================

VAR = 0.3  # 方差
D_1 = 10.0  # 平均环半径
M = D_1 + 5  # 绘图范围
COMP = 200  # 不规则环上的高斯分布数量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 源分布 (π_0): 单一高斯分布 (中心)
source_mean = torch.tensor([0.0, 0.0])
source_cov = VAR * torch.eye(2)
initial_model = MultivariateNormal(source_mean, source_cov)
samples_0 = initial_model.sample([1000])

# 生成不规则环的均值
np.random.seed(42)  # 固定随机种子，保证可重复性
angles = [k * (2 * np.pi / COMP) for k in range(COMP)]
radii = [D_1 + np.sin(3 * theta) * 2 + np.random.uniform(-1.5, 1.5) for theta in angles]
vertices_1 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(radii, angles)]

# 目标分布 (π_1): 不规则环
target_mix = Categorical(torch.tensor([1 / COMP for _ in range(COMP)]))
target_comp = MultivariateNormal(torch.tensor(vertices_1).float(),
                                  VAR * torch.stack([torch.eye(2) for _ in range(COMP)]))
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([1000])

print('Shape of the samples:', samples_0.shape, samples_1.shape)

# 绘制 Ground Truth
plt.figure(figsize=(4, 4))
plt.xlim(-M, M)
plt.ylim(-M, M)
plt.title(r'Samples from $\pi_0$ and $\pi_1$', fontsize=19)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.6, c='#BD8253', label=r'$\pi_0$')
plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.6, c='#2E59A7', label=r'$\pi_1$')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 12})
plt.tight_layout()
plt.show()


# ==========================================
# 2. 模型定义 (Mean Flow Net)
# ==========================================

class MeanFlowNet(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        # Mean Flow One-Step 模型只需要输入 x, 不需要 t
        # 这是一个直接的 map: x_0 -> x_1
        # 使用2个隐藏层以匹配其他 baseline 的参数量
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_num),
            nn.Tanh(),
            nn.Linear(hidden_num, hidden_num),
            nn.Tanh(),
            nn.Linear(hidden_num, input_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 训练逻辑 (OT Matching + Regression)
# ==========================================

def get_ot_pairs(x0, x1):
    r"""
    计算 Minibatch Optimal Transport 配对。
    解决 Assignment Problem: 找到一个排列 sigma，使得 \sum ||x0[i] - x1[sigma[i]]||^2 最小。
    """
    # 1. 计算代价矩阵 (Euclidean distance squared is fine for OT map learning)
    # [Batch, Batch]
    with torch.no_grad():
        dists = torch.cdist(x0, x1)  # L2 distance
        
        # 将 tensor 转为 numpy 以使用 scipy
        dists_np = dists.cpu().numpy()
        
        # 2. 使用匈牙利算法 (Linear Sum Assignment) 求解最优匹配
        # row_idx 通常是 [0, 1, ..., B-1]
        # col_idx 是重新排序后的索引
        row_idx, col_idx = linear_sum_assignment(dists_np)
        
        # 3. 根据 Optimal Assignment 重新排列 x1
        x1_paired = x1[col_idx]
        
    return x0, x1_paired

def train_mean_flow(model, optimizer, z0_sampler, z1_sampler, batchsize, iterations):
    loss_curve = []
    
    for i in tqdm(range(iterations + 1)):
        optimizer.zero_grad()
        
        # 1. 独立采样
        x0 = z0_sampler.sample([batchsize]).to(device)
        x1 = z1_sampler.sample([batchsize]).to(device)
        
        # 2. [Mean Flow Core] Optimal Transport Matching
        # 如果不配对直接 MSE，模型会学习到 E[x1]，即所有 Mode 的平均值（原点）。
        # 配对后，模型学习的是 Optimal Transport Map。
        x0_paired, x1_paired = get_ot_pairs(x0, x1)
        
        # 3. 预测
        pred_x1 = model(x0_paired)
        
        # 4. Loss: Simple MSE
        loss = (pred_x1 - x1_paired).pow(2).mean()
        
        loss.backward()
        optimizer.step()
        
        loss_curve.append(loss.item())
        
    return model, loss_curve

@torch.no_grad()
def draw_plot(model, z0, z1):
    # One-step generation
    generated_z1 = model(z0)

    distances = torch.cdist(generated_z1, torch.tensor(vertices_1).float().to(generated_z1.device))
    min_distances, _ = torch.min(distances, dim=1)
    average_min_distance = min_distances.mean().item()
    print("aver_dist", average_min_distance)

    plt.figure(figsize=(4, 4))
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), c='#BD8253', label=r'$\pi_0$', alpha=0.6)
    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), c='#2E59A7', label=r'$\pi_1$', alpha=0.6)
    plt.scatter(generated_z1[:, 0].cpu().numpy(), generated_z1[:, 1].cpu().numpy(), c='#D9A0B3', label='Mean Flow', alpha=0.6)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    plt.title('Mean Flow (One-Step)\n(via OT Matching)', fontsize=19)
    plt.tight_layout()
    plt.savefig('irr_circle_meanflow_output.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# ==========================================
# 4. 执行训练
# ==========================================

iterations = 2000
batchsize = 256  # 减小batch size，匈牙利算法O(n³)复杂度，1024太慢
input_dim = 2

# 初始化模型
# 结构保持与其他 baseline 相似 (Hidden 100, 2层)
model = MeanFlowNet(input_dim, hidden_num=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# 训练
# 注意：这里直接传入 sampler，因为 Mean Flow 需要每次迭代重新采样并配对
model, loss_curve = train_mean_flow(model, optimizer, initial_model, target_model, batchsize, iterations)

# 绘制 Loss
plt.figure(figsize=(6, 3))
plt.plot(loss_curve)
plt.title('Mean Flow Loss (OT-MSE)')
plt.show()

# 绘图验证
test_z0 = initial_model.sample([500]).to(device)
test_z1 = target_model.sample([500]).to(device)
draw_plot(model, test_z0, test_z1)

# 验证几何特性
# Mean Flow 生成的映射通常非常"直"，且保留几何结构
print("Evaluation Complete.")
