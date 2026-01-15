#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm

# ==========================================
# 1. 数据生成与目标分布定义
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 分布参数
D_0 = 6.
D_1 = 13.
M = D_1 + 5
VAR = 0.3
COMP = 5

# --- 定义目标分布 π_1 (GMM) 用于 Distribution Matching ---
# 正五边形的顶点坐标 (π_1)
angles = [k * (2 * np.pi / COMP) for k in range(COMP)]
rotation_angle = 2 * np.pi / 10
vertices_1 = [[D_1 * np.cos(theta + rotation_angle), D_1 * np.sin(theta + rotation_angle)] for theta in angles]
vertices_1_tensor = torch.tensor(vertices_1).float().to(device)

# 目标分布（π_1）的混合模型
target_mix = Categorical(torch.tensor([1/COMP for _ in range(COMP)]).to(device))
target_comp = MultivariateNormal(vertices_1_tensor, VAR * torch.stack([torch.eye(2) for _ in range(COMP)]).to(device))
target_gmm = MixtureSameFamily(target_mix, target_comp)

# 源分布 π_0 (用于可视化)
vertices_0 = [[D_0 * np.cos(theta), D_0 * np.sin(theta)] for theta in angles]
initial_mix = Categorical(torch.tensor([1/COMP for _ in range(COMP)]).to(device))
initial_comp = MultivariateNormal(torch.tensor(vertices_0).float().to(device), VAR * torch.stack([torch.eye(2) for _ in range(COMP)]).to(device))
initial_model = MixtureSameFamily(initial_mix, initial_comp)

# 采样用于绘图
samples_0 = initial_model.sample([1000])
samples_1 = target_gmm.sample([1000])

# 绘制 Ground Truth
plt.figure(figsize=(4, 4))
plt.xlim(-M, M)
plt.ylim(-M, M)
plt.title(r'Samples from $\pi_0$ and $\pi_1$', fontsize=19)
plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.6, c='#BD8253', label=r'$\pi_0$')
plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.6, c='#2E59A7', label=r'$\pi_1$')
plt.legend(loc='upper right', prop={'size': 12})
plt.tight_layout()
plt.show()

# ==========================================
# 2. 教师模型 (Teacher) - 基于 ODE 的轨迹生成器
# ==========================================
class TeacherODESolver:
    """
    Simulates a pre-trained Diffusion/Flow Teacher.
    For this toy case, we use nearest-neighbor matching to simulate optimal transport.
    """
    def __init__(self, noise_scale=1.0):
        # noise_scale 控制 Teacher 输出的噪声大小
        # 1.0 = 标准噪声，>1.0 = 更大噪声（更不准确的 Teacher）
        self.noise_scale = noise_scale

    @torch.no_grad()
    def generate_targets(self, z0):
        """
        Generates the target x1 for a given z0.
        Heuristic for Toy Data: Map z0 to the closest mode in pi_1.
        """
        # 计算 z0 到所有 5 个目标中心的距离
        # z0: [B, 2], vertices: [5, 2] -> dist: [B, 5]
        dists = torch.cdist(z0, vertices_1_tensor)
        min_idx = torch.argmin(dists, dim=1) # [B]
        
        # 获取对应的目标中心
        z1_centers = vertices_1_tensor[min_idx]
        
        # 添加随机扰动以模拟真实分布的方差（增加 noise_scale 使 Teacher 更不准确）
        z1_target = z1_centers + torch.randn_like(z0) * np.sqrt(VAR) * self.noise_scale
        
        return z1_target

# ==========================================
# 3. 学生模型 (Student) - One-Step Generator
# ==========================================
class DMDStudent(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        # One-step generator: z0 -> z1 directly
        # 3层结构，与 shortcut MLP 参数量相近 (~10,802)
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
# 4. DMD 训练逻辑
# ==========================================
def train_dmd(student, teacher, optimizer, z0_batch, iterations, lambda_dist=0.1):
    loss_curve = []
    
    for i in tqdm(range(iterations + 1)):
        optimizer.zero_grad()
        
        # 1. Sampling
        # 随机采样噪声 z0 (来自 pi_0)
        idx = torch.randperm(len(z0_batch))[:2048]
        z_in = z0_batch[idx]
        
        # 2. Teacher Generation (Regression Target)
        # 教师告诉学生：对于这个 z_in，你应该去哪里 (y_target)
        with torch.no_grad():
            y_target = teacher.generate_targets(z_in)
            
        # 3. Student Prediction
        y_pred = student(z_in)
        
        # 4. Loss Calculation
        
        # (A) Regression Loss (MSE) - "Mode Covering"
        # 确保学生生成的输出在几何上接近教师的轨迹终点
        loss_reg = (y_pred - y_target).pow(2).mean()
        
        # (B) Distribution Matching Loss (NLL) - "Mode Sharpening" [Cite: DMD Paper]
        # DMD Paper 使用 score difference。对于已知密度的 Toy Data，
        # 最小化 -log p(x) 等价于利用真实 Score 进行引导 (Score Distillation)。
        # 这会给模型一个梯度信号，使其输出推向高密度区域。
        log_prob = target_gmm.log_prob(y_pred)
        loss_dist = -log_prob.mean()
        
        # Total Loss
        loss = loss_reg + lambda_dist * loss_dist
        
        loss.backward()
        optimizer.step()
        
        loss_curve.append(loss.item())
        
    return loss_curve

# ==========================================
# 5. 执行训练
# ==========================================

# 准备数据
x_0 = samples_0.detach().clone().to(device)

# 初始化
student = DMDStudent().to(device)
teacher = TeacherODESolver(noise_scale=30.0)  # 增加 Teacher 噪声，使其更不准确
optimizer = torch.optim.Adam(student.parameters(), lr=5e-3)

# 打印参数量
num_params = sum(p.numel() for p in student.parameters())
print(f'Model parameters: {num_params:,}')

# 训练 DMD
# lambda_dist=0: 不使用 Distribution Matching Loss（不利用目标分布的密度信息）
loss_history = train_dmd(student, teacher, optimizer, x_0, iterations=1000, lambda_dist=0.0)

# 绘制 Loss
plt.figure(figsize=(6, 3))
plt.plot(loss_history)
plt.title('DMD Loss (Regression + Distribution Matching)')
plt.show()

# ==========================================
# 6. 评估与绘图
# ==========================================
@torch.no_grad()
def evaluate_dmd(student, z0, z1):
    # One-step generation
    generated_z1 = student(z0)
    
    # 计算 aver_dist（与 shortcut 代码一致）
    distances = torch.cdist(generated_z1, vertices_1_tensor)
    min_distances, _ = torch.min(distances, dim=1)
    average_min_distance = min_distances.mean().item()
    print("aver_dist", average_min_distance)
    
    plt.figure(figsize=(4,4))
    plt.xlim(-M,M)
    plt.ylim(-M,M)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 绘制背景真值
    plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), c='#BD8253', label=r'$\pi_0$', alpha=0.6)
    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), c='#2E59A7', label=r'$\pi_1$', alpha=0.6)
    
    # 绘制生成结果
    plt.scatter(generated_z1[:, 0].cpu().numpy(), generated_z1[:, 1].cpu().numpy(), c='#D9A0B3', label='Generated', alpha=0.6)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    plt.title('DMD One-Step\nGeneration', fontsize=19)
    plt.tight_layout()
    plt.savefig('5_dmd_output.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# 测试
test_z0 = initial_model.sample([500]).to(device)
test_z1 = target_gmm.sample([500]).to(device)
evaluate_dmd(student, test_z0, test_z1)