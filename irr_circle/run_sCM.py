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
# 2. 模型定义 (Consistency Model)
# ==========================================

class ConsistencyModel(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = nn.Tanh()

    def forward(self, x_input, t):
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
            
        inputs = torch.cat([x_input, t], dim=1)
        
        h = self.fc1(inputs)
        h = self.act(h)
        h = self.fc2(h)
        h = self.act(h)
        out = self.fc3(h)
        
        # 直接输出预测值（不使用 Skip Connection）
        # 因为源分布和目标分布的尺度差异大（中心 vs 半径10环）
        return out 

# ==========================================
# 3. 训练逻辑 (sCM: Continuous-Time Consistency Training)
# ==========================================

class sCMTraining():
    def __init__(self, model, delta=0.01):
        """
        Args:
            delta: The simplified small time step interval for continuous training.
        """
        self.model = model
        self.target_model = deepcopy(model)  # EMA Model
        self.delta = delta 
        
        # Freezing the target model
        for param in self.target_model.parameters():
            param.requires_grad = False
            
    def update_target_model(self, decay=0.95):
        """Exponential Moving Average update - 使用更快的更新速度"""
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def get_ode_state(self, z0, z1, t):
        """
        Analytical Ground Truth Solver (VP-ODE style) for training data generation.
        """
        a = 19.9
        b = 0.1
        
        alpha_t = torch.exp(- (0.25) * a * (1-t)**2 - (0.5) * b * (1-t))
        beta_t = torch.sqrt(1 - alpha_t**2 + 1e-8)

        z_t = alpha_t * z1 + beta_t * z0
        return z_t

    def pseudo_huber_loss(self, input, target, c=1.0):
        """
        sCM & Improved CM both recommend Pseudo-Huber loss for stability.
        c=1.0 使损失函数更接近 MSE，更稳定
        """
        loss = torch.sqrt((input - target).pow(2).sum(dim=1) + c**2) - c
        return loss.mean()

    def get_consistency_loss(self, z0, z1):
        batch_size = z0.shape[0]
        
        # [sCM Key Change 1] Continuous Time Sampling
        t = torch.rand((batch_size, 1)).to(z0.device)
        
        # [sCM Key Change 2] Fixed small delta step
        t_next = torch.clamp(t + self.delta, max=1.0)
        
        # 1. 生成成对的训练数据 (z_t, z_next)
        z_t = self.get_ode_state(z0, z1, t)
        z_next = self.get_ode_state(z0, z1, t_next)
        
        # 2. Student Predicts from z_t
        pred_student = self.model(z_t, t)
        
        # 3. Teacher (EMA) Predicts from z_next
        with torch.no_grad():
            pred_teacher = self.target_model(z_next, t_next)
            
        # 4. Consistency Loss
        consistency_loss = self.pseudo_huber_loss(pred_student, pred_teacher)
        
        # 5. Supervision Loss (关键！告诉模型正确的目标是 z1)
        t_random = torch.rand((batch_size, 1)).to(z0.device)
        z_random = self.get_ode_state(z0, z1, t_random)
        pred_z1 = self.model(z_random, t_random)
        supervision_loss = self.pseudo_huber_loss(pred_z1, z1)
        
        # Combined loss: 增加 supervision 权重
        loss = 0.5 * consistency_loss + 1.5 * supervision_loss
        
        return loss

    @torch.no_grad()
    def sample_one_step(self, z0):
        # t=0 represents the start distribution
        t = torch.zeros((z0.shape[0], 1)).to(z0.device)
        pred_z1 = self.model(z0, t)
        return pred_z1

# ==========================================
# 4. 训练过程
# ==========================================

def train_scm(trainer, optimizer, pairs, batchsize, iterations):
    loss_curve = []
    
    for i in tqdm(range(iterations + 1)):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].to(device)
        z1 = batch[:, 1].to(device)
        
        loss = trainer.get_consistency_loss(z0, z1)
        loss.backward()
        optimizer.step()
        
        # Update EMA model
        trainer.update_target_model()
        
        loss_curve.append(loss.item())
        
    return trainer, loss_curve

@torch.no_grad()
def draw_plot(trainer, z0, z1):
    generated_z1 = trainer.sample_one_step(z0)

    distances = torch.cdist(generated_z1, torch.tensor(vertices_1).float().to(generated_z1.device))
    min_distances, _ = torch.min(distances, dim=1)
    average_min_distance = min_distances.mean().item()
    print("aver_dist", average_min_distance)

    plt.figure(figsize=(4,4))
    plt.xlim(-M,M)
    plt.ylim(-M,M)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), c='#BD8253' , label=r'$\pi_0$', alpha=0.6)
    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), c='#2E59A7', label=r'$\pi_1$', alpha=0.6)
    plt.scatter(generated_z1[:, 0].cpu().numpy(), generated_z1[:, 1].cpu().numpy(), c='#D9A0B3' , label='sCM Generated', alpha=0.6)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    plt.title('sCM (Continuous)\n(One-Step Generation)', fontsize=19)
    plt.tight_layout()
    plt.savefig('irr_circle_scm_output.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# ==========================================
# 5. 执行训练
# ==========================================

# 准备数据对 (z0, z1) - 最近邻配对
# 对于 irr_circle，源分布是中心高斯，目标是不规则环
# 我们将每个 z0 映射到环上最近的点
def create_paired_data(z0_samples):
    z0 = z0_samples.to(device)
    dists = torch.cdist(z0, torch.tensor(vertices_1).float().to(device))
    min_idx = torch.argmin(dists, dim=1)
    z1_centers = torch.tensor(vertices_1).float().to(device)[min_idx]
    z1 = z1_centers + torch.randn_like(z0) * np.sqrt(VAR)
    return z0, z1

x_0, x_1 = create_paired_data(samples_0)
z_pairs = torch.stack([x_0, x_1], dim=1).to(device)

iterations = 2000 
batchsize = 2048
input_dim = 2

# 初始化模型
model = ConsistencyModel(input_dim, hidden_num=100).to(device)

# sCM 训练器
scm_trainer = sCMTraining(model, delta=0.02) 

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# 训练
scm_trainer, loss_curve = train_scm(scm_trainer, optimizer, z_pairs, batchsize, iterations)

# 绘制 Loss
plt.figure(figsize=(6, 3))
plt.plot(loss_curve)
plt.title('sCM Loss (Continuous Time)')
plt.show()

# 绘图验证
draw_plot(scm_trainer, z0=initial_model.sample([500]).to(device), z1=target_model.sample([500]).to(device))

# 验证一致性 (Consistency Check)
print("Testing Consistency across time steps...")
test_z0_samples = initial_model.sample([10]).to(device)
test_z0, test_z1 = create_paired_data(test_z0_samples)
t_mid = torch.ones(10, 1).to(device) * 0.5
z_mid = scm_trainer.get_ode_state(test_z0, test_z1, t_mid)

pred_from_start = scm_trainer.model(test_z0, torch.zeros_like(t_mid))
pred_from_mid = scm_trainer.model(z_mid, t_mid)

diff = (pred_from_start - pred_from_mid).pow(2).sum().item()
print(f"Difference between prediction from t=0 and t=0.5: {diff:.6f} (Should be small)")
