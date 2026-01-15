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
# 1. 数据生成部分 (与 13_irr_circle.py 一致)
# ==========================================

# 分布参数
VAR = 0.3  # 方差
D_1 = 10.0  # 平均环半径
M = D_1 + 5  # 绘图范围
COMP = 200  # 不规则环上的高斯分布数量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 源分布 (π_0): 单一高斯分布
source_mean = torch.tensor([0.0, 0.0])
source_cov = VAR * torch.eye(2)
initial_model = MultivariateNormal(source_mean, source_cov)
samples_0 = initial_model.sample([600])

# 生成不规则环的均值 (固定随机种子以保证可复现)
np.random.seed(42)
angles = [k * (2 * np.pi / COMP) for k in range(COMP)]
radii = [D_1 + np.sin(3 * theta) * 2 + np.random.uniform(-1.5, 1.5) for theta in angles]
vertices_1 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(radii, angles)]

# 构建目标分布 (π_1)
target_mix = Categorical(torch.tensor([1 / COMP for _ in range(COMP)]))
target_comp = MultivariateNormal(torch.tensor(vertices_1).float(),
                                  VAR * torch.stack([torch.eye(2) for _ in range(COMP)]))
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([600])

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
# 2. 模型定义 (Consistency Model)
# ==========================================

class ConsistencyModel(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        # 输入维度: x(2) + t(1) = 3
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
        # 因为源分布和目标分布的尺度差异大（中心点 vs 环形）
        return out


# ==========================================
# 3. 改进的训练逻辑 (Improved Consistency Training)
# ==========================================

class ImprovedConsistencyTraining():
    def __init__(self, model, num_steps=20, rho=7.0):
        self.model = model
        self.target_model = deepcopy(model)  # EMA Model
        self.rho = rho  # Karras Schedule 参数
        
        # Freezing the target model
        for param in self.target_model.parameters():
            param.requires_grad = False
            
        self.N = num_steps 
        
    def update_target_model(self, decay=0.999):
        """Exponential Moving Average update"""
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def get_karras_schedule(self, n, N):
        r"""
        [Improved Technique] Karras Time Schedule
        将离散的时间步 n 映射到连续时间 t，更密集地采样低噪声区域
        t \in [0, 1]
        """
        t = (n / (N - 1)) ** self.rho
        return t

    def get_ode_state_and_derivative(self, z0, z1, t):
        """Analytical Solver (VP-ODE style interpolation)"""
        a = 19.9
        b = 0.1
        
        alpha_t = torch.exp(- (0.25) * a * (1-t)**2 - (0.5) * b * (1-t))
        beta_t = torch.sqrt(1 - alpha_t**2 + 1e-8)
        z_t = alpha_t * z1 + beta_t * z0
        return z_t

    def pseudo_huber_loss(self, input, target, c=1.0):
        """
        [Improved Technique] Pseudo-Huber Loss
        Loss = sqrt(|x-y|^2 + c^2) - c
        """
        loss = torch.sqrt((input - target).pow(2).sum(dim=1) + c**2) - c
        return loss.mean()

    def get_consistency_loss(self, z0, z1):
        batch_size = z0.shape[0]
        
        # 1. Sample discrete time steps n ~ U[0, N-2]
        n = torch.randint(0, self.N - 1, (batch_size, 1)).to(z0.device)
        
        # 2. Convert to continuous time
        t_current = n / self.N
        t_next = (n + 1) / self.N
        
        # 3. Generate data points on the trajectory
        z_current = self.get_ode_state_and_derivative(z0, z1, t_current)
        z_next = self.get_ode_state_and_derivative(z0, z1, t_next)
        
        # 4. Student predicts from 'next'
        pred_next = self.model(z_next, t_next)
        
        # 5. Teacher (EMA) predicts from 'current'
        with torch.no_grad():
            target_current = self.target_model(z_current, t_current)
            
        # 6. Pseudo-Huber Consistency Loss
        consistency_loss = self.pseudo_huber_loss(pred_next, target_current)
        
        # 7. Supervision Loss (关键！告诉模型正确的目标是 z1)
        t_random = torch.rand((batch_size, 1)).to(z0.device)
        z_t = self.get_ode_state_and_derivative(z0, z1, t_random)
        pred_z1 = self.model(z_t, t_random)
        supervision_loss = self.pseudo_huber_loss(pred_z1, z1)
        
        # 同时使用 Consistency Loss 和 Supervision Loss
        loss = consistency_loss + supervision_loss
        
        return loss

    @torch.no_grad()
    def sample_one_step(self, z0):
        t = torch.zeros((z0.shape[0], 1)).to(z0.device)
        pred_z1 = self.model(z0, t)
        return pred_z1


# ==========================================
# 4. 训练过程
# ==========================================

def train_consistency_model(cm_trainer, optimizer, pairs, batchsize, iterations):
    loss_curve = []
    
    for i in tqdm(range(iterations + 1)):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].to(device)
        z1 = batch[:, 1].to(device)
        
        loss = cm_trainer.get_consistency_loss(z0, z1)
        loss.backward()
        optimizer.step()
        
        # Update EMA model
        cm_trainer.update_target_model()
        
        loss_curve.append(loss.item())
        
    return cm_trainer, loss_curve


@torch.no_grad()
def draw_plot(cm_trainer, z0, z1):
    generated_z1 = cm_trainer.sample_one_step(z0)

    # 计算 aver_dist
    vertices_1_tensor = torch.tensor(vertices_1).float().to(generated_z1.device)
    distances = torch.cdist(generated_z1, vertices_1_tensor)
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
    plt.scatter(generated_z1[:, 0].cpu().numpy(), generated_z1[:, 1].cpu().numpy(), c='#D9A0B3', label='Improved CM', alpha=0.6)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    plt.title('Improved Consistency\nModel', fontsize=19)
    plt.tight_layout()
    plt.savefig('irr_circle_improved_cm_output.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# ==========================================
# 5. 执行训练
# ==========================================

# 准备数据对 (z0, z1) - 最近邻配对
vertices_1_tensor = torch.tensor(vertices_1).float().to(device)

def create_paired_data(z0_samples):
    z0 = z0_samples.to(device)
    dists = torch.cdist(z0, vertices_1_tensor)
    min_idx = torch.argmin(dists, dim=1)
    z1_centers = vertices_1_tensor[min_idx]
    z1 = z1_centers + torch.randn_like(z0) * np.sqrt(VAR)
    return z0, z1

x_0, x_1 = create_paired_data(samples_0)
z_pairs = torch.stack([x_0, x_1], dim=1).to(device)

print('z_pairs shape:', z_pairs.shape)

# 训练参数
iterations = 1000  # 与 shortcut 一致
batchsize = 2048
input_dim = 2

# 初始化模型
model = ConsistencyModel(input_dim, hidden_num=100).to(device)
cm_trainer = ImprovedConsistencyTraining(model, num_steps=40)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# 打印参数量
num_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {num_params:,}')

# 训练
cm_trainer, loss_curve = train_consistency_model(cm_trainer, optimizer, z_pairs, batchsize, iterations)

# 绘制 Loss
plt.figure(figsize=(6, 3))
plt.plot(loss_curve)
plt.title('Improved Consistency Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# 绘图验证
draw_plot(cm_trainer, z0=initial_model.sample([600]).to(device), z1=target_model.sample([600]).to(device))
