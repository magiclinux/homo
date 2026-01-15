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
# 1. 数据生成部分 (保持不变)
# ==========================================

# 分布的半径
D_0 = 6.  # π_0 的半径（较小）
D_1 = 13.  # π_1 的半径（较大）
M = D_1 + 5  # 绘图范围
VAR = 0.3  # 方差
COMP = 5  # 组件数目（正五边形的5个顶点）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动选择设备

# 正五边形的顶点坐标 (π_0)
angles = [k * (2 * np.pi / COMP) for k in range(COMP)]  # 5个顶点的角度
vertices_0 = [[D_0 * np.cos(theta), D_0 * np.sin(theta)] for theta in angles]  # π_0 的顶点

# 源分布（π_0）的混合模型
initial_mix = Categorical(torch.tensor([1/COMP for _ in range(COMP)]))  # 均匀权重
initial_comp = MultivariateNormal(torch.tensor(vertices_0).float(),  # π_0 的顶点作为均值
                                  VAR * torch.stack([torch.eye(2) for _ in range(COMP)]))  # 协方差矩阵
initial_model = MixtureSameFamily(initial_mix, initial_comp)
samples_0 = initial_model.sample([1000])

# 正五边形的顶点坐标 (π_1)
rotation_angle = 2 * np.pi / 10  # 顺时针旋转 2π/10
vertices_1 = [[D_1 * np.cos(theta + rotation_angle), D_1 * np.sin(theta + rotation_angle)] for theta in angles]  # π_1 的顶点

# 目标分布（π_1）的混合模型
target_mix = Categorical(torch.tensor([1/COMP for _ in range(COMP)]))  # 均匀权重
target_comp = MultivariateNormal(torch.tensor(vertices_1).float(),  # π_1 的顶点作为均值
                                  VAR * torch.stack([torch.eye(2) for _ in range(COMP)]))  # 协方差矩阵
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([1000])

print('Shape of the samples:', samples_0.shape, samples_1.shape)

# 绘制样本点
plt.figure(figsize=(4, 4))
plt.xlim(-M, M)
plt.ylim(-M, M)
plt.title(r'Samples from $\pi_0$ and $\pi_1$', fontsize=19)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.6, c = '#BD8253', label=r'$\pi_0$')
plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.6, c = '#2E59A7', label=r'$\pi_1$')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 12})
plt.tight_layout()
plt.show()


# ==========================================
# 2. 模型定义 (Consistency Model)
# ==========================================

class ConsistencyModel(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        # 输入维度: x(2) + t(1) = 3 (去掉了d)
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = nn.Tanh()

    def forward(self, x_input, t):
        # t needs to be shaped [batch, 1]
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
            
        inputs = torch.cat([x_input, t], dim=1)
        
        h = self.fc1(inputs)
        h = self.act(h)
        h = self.fc2(h)
        h = self.act(h)
        out = self.fc3(h)
        
        return x_input + (1.0 - t) * out 

# ==========================================
# 3. 训练逻辑 (Consistency Training / Distillation)
# ==========================================

class ConsistencyTraining():
    def __init__(self, model, num_steps=20):
        self.model = model
        self.target_model = deepcopy(model) # EMA Model
        
        # Freezing the target model
        for param in self.target_model.parameters():
            param.requires_grad = False
            
        self.N = num_steps # Discretization steps for training
        
    def update_target_model(self, decay=0.999):
        """Exponential Moving Average update for target model"""
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def get_ode_state_and_derivative(self, z0, z1, t):
        """
        Calculates z_t and dz_t/dt based on the analytical schedule (VP-ODE style).
        Used as the 'Teacher' solver.
        """
        a = 19.9
        b = 0.1
        
        # alpha_t = e^{(-1/4 a (1-t)^2-1/2 b(1-t))}
        alpha_t = torch.exp(- (0.25) * a * (1-t)**2 - (0.5) * b * (1-t))

        # d alpha_t / dt
        d_alpha_dt = alpha_t * 0.5 * (a * (1-t) + b)

        # beta_t = sqrt{1-alpha^2}
        beta_t = torch.sqrt(1 - alpha_t**2 + 1e-8)
        
        # d beta_t / dt
        # Avoid division by zero at t=1 (where beta=0)
        d_beta_dt = torch.zeros_like(beta_t)
        mask = beta_t > 1e-4
        d_beta_dt[mask] = (- alpha_t[mask] / beta_t[mask]) * d_alpha_dt[mask]

        z_t = alpha_t * z1 + beta_t * z0
        
        # Velocity v_t = d z_t / dt
        velocity_t = d_alpha_dt * z1 + d_beta_dt * z0
        
        return z_t, velocity_t

    def get_consistency_loss(self, z0, z1):
        r"""
        Consistency Loss with Supervision:
        1. Consistency loss: F(z_{n+1}, t_{n+1}) ≈ F(z_n, t_n) (self-consistency)
        2. Supervision loss: F(z_t, t) ≈ z1 (target supervision)
        """
        batch_size = z0.shape[0]
        
        n = torch.randint(0, self.N - 1, (batch_size, 1)).to(z0.device)
        
        t_current = n / self.N       # t_n
        t_next = (n + 1) / self.N    # t_{n+1}
        
        # 1. Generate data at t_{n} (Current) and t_{n+1} (Next/Future in flow direction)
        z_current, _ = self.get_ode_state_and_derivative(z0, z1, t_current)
        z_next, _ = self.get_ode_state_and_derivative(z0, z1, t_next)
        
        # 2. Consistency Loss: F(z_next, t_next) ≈ F(z_current, t_current)
        pred_next = self.model(z_next, t_next)
        
        with torch.no_grad():
            target_current = self.target_model(z_current, t_current)
            
        consistency_loss = (pred_next - target_current).pow(2).mean()
        
        # 3. Supervision Loss: F(z_t, t) should predict z1 (the target)
        # 这是关键！Consistency Model 应该把轨迹上的点映射到终点 z1
        t_random = torch.rand((batch_size, 1)).to(z0.device)
        z_t, _ = self.get_ode_state_and_derivative(z0, z1, t_random)
        pred_z1 = self.model(z_t, t_random)
        supervision_loss = (pred_z1 - z1).pow(2).mean()
        
        # Combined loss
        loss = consistency_loss + supervision_loss
        
        return loss

    @torch.no_grad()
    def sample_one_step(self, z0):
        """
        Consistency Models allow single-step generation.
        Input z0 (from pi_0), predict z1 (pi_1).
        """
        # t=0 represents the start distribution
        t = torch.zeros((z0.shape[0], 1)).to(z0.device)
        pred_z1 = self.model(z0, t)
        return pred_z1

    @torch.no_grad()
    def sample_multistep(self, z0, steps=10):
        """
        Chained consistency sampling (optional, improves quality).
        Algo: z -> denoise -> add noise -> denoise ...
        Here implemented as simple Euler for comparison with original code behavior.
        """
        dt = 1.0 / steps
        traj = [z0.detach().clone()]
        z = z0.detach().clone()
        
        # This model predicts z1 directly.
        # If we want to check intermediate steps, we should simply use the consistency property.
        # But CM maps x_t -> x_1.
        # To generate trajectory, we usually just interpolate or use the predictor.
        
        # For visualization, let's just show the one-step jump at different t
        # (Just linear interpolation visualizer essentially)
        pred_final = self.model(z0, torch.zeros_like(z0[:, :1]))
        
        # Linear interp for visualization
        for i in range(steps):
             alpha = i / steps
             z_interp = (1-alpha) * z0 + alpha * pred_final
             traj.append(z_interp)
             
        traj.append(pred_final)
        return traj

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
    # One-step generation
    generated_z1 = cm_trainer.sample_one_step(z0)

    # 计算 aver_dist（与 shortcut 代码一致）
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
    plt.scatter(generated_z1[:, 0].cpu().numpy(), generated_z1[:, 1].cpu().numpy(), c='#D9A0B3' , label='Generated', alpha=0.6)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    plt.title('Consistency Model\n(One-Step Generation)', fontsize=19)
    plt.tight_layout()
    plt.savefig('5_consistency_output.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# ==========================================
# 5. 执行训练
# ==========================================

# 准备数据对 (z0, z1) - 使用最近邻配对（与 DMD 一致，公平对比）
def create_paired_data(z0_samples):
    """使用最近邻匹配创建配对数据"""
    z0 = z0_samples.to(device)
    # 计算到所有目标中心的距离
    dists = torch.cdist(z0, torch.tensor(vertices_1).float().to(device))
    min_idx = torch.argmin(dists, dim=1)
    # 获取对应的目标中心并添加噪声
    z1_centers = torch.tensor(vertices_1).float().to(device)[min_idx]
    z1 = z1_centers + torch.randn_like(z0) * np.sqrt(VAR)
    return z0, z1

x_0, x_1 = create_paired_data(samples_0)
z_pairs = torch.stack([x_0, x_1], dim=1).to(device)

iterations = 1000  # 与 shortcut 一致
batchsize = 2048
input_dim = 2

# 初始化模型和训练器
model = ConsistencyModel(input_dim, hidden_num=100).to(device)
cm_trainer = ConsistencyTraining(model, num_steps=20) # 离散化为20步进行一致性学习
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# 训练
cm_trainer, loss_curve = train_consistency_model(cm_trainer, optimizer, z_pairs, batchsize, iterations)

# 绘制 Loss 曲线
plt.figure(figsize=(6, 3))
plt.plot(loss_curve)
plt.title('Consistency Loss')
plt.show()

# 绘图验证
draw_plot(cm_trainer, z0=initial_model.sample([500]).to(device), z1=target_model.sample([500]).to(device))

print("Testing Consistency across time steps...")
test_z0 = initial_model.sample([10]).to(device)
test_z1 = target_model.sample([10]).to(device)
t_mid = torch.ones(10, 1).to(device) * 0.5
z_mid, _ = cm_trainer.get_ode_state_and_derivative(test_z0, test_z1, t_mid)

pred_from_start = cm_trainer.model(test_z0, torch.zeros_like(t_mid))
pred_from_mid = cm_trainer.model(z_mid, t_mid)

diff = (pred_from_start - pred_from_mid).pow(2).sum().item()
print(f"Difference between prediction from t=0 and t=0.5: {diff:.6f} (Should be small)")