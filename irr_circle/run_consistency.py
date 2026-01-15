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
source_mean = torch.tensor([0.0, 0.0])  # 中心位置
source_cov = VAR * torch.eye(2)  # 方差矩阵
initial_model = MultivariateNormal(source_mean, source_cov)
samples_0 = initial_model.sample([600])  # 从单一高斯分布中采样

# 生成不规则环的均值 (固定随机种子以保证可复现)
np.random.seed(42)
angles = [k * (2 * np.pi / COMP) for k in range(COMP)]  # 环上的角度
radii = [D_1 + np.sin(3 * theta) * 2 + np.random.uniform(-1.5, 1.5) for theta in angles]  # 半径引入扰动
vertices_1 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(radii, angles)]  # 各高斯分布的均值

# 构建目标分布 (π_1)
target_mix = Categorical(torch.tensor([1 / COMP for _ in range(COMP)]))  # 均匀权重
target_comp = MultivariateNormal(torch.tensor(vertices_1).float(),  # 各高斯分布的均值
                                  VAR * torch.stack([torch.eye(2) for _ in range(COMP)]))  # 方差矩阵
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([600])  # 从不规则环分布中采样

print('Shape of the samples:', samples_0.shape, samples_1.shape)

# 绘制样本点
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
        # t needs to be shaped [batch, 1]
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
            
        inputs = torch.cat([x_input, t], dim=1)
        
        h = self.fc1(inputs)
        h = self.act(h)
        h = self.fc2(h)
        h = self.act(h)
        out = self.fc3(h)
        
        # Skip Connection: F(x, t) = x + (1-t) * Net(x, t)
        # 当 t=1 时，F(x, 1) = x (恒等映射)
        # 当 t=0 时，F(x, 0) = x + Net(x, 0)
        return x_input + (1.0 - t) * out


# ==========================================
# 3. 训练逻辑 (Consistency Training)
# ==========================================

class ConsistencyTraining():
    def __init__(self, model, num_steps=20):
        self.model = model
        self.target_model = deepcopy(model)  # EMA Model
        
        # Freezing the target model
        for param in self.target_model.parameters():
            param.requires_grad = False
            
        self.N = num_steps  # Discretization steps for training
        
    def update_target_model(self, decay=0.999):
        """Exponential Moving Average update for target model"""
        with torch.no_grad():
            for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def get_ode_state_and_derivative(self, z0, z1, t):
        """
        Calculates z_t based on the analytical schedule (VP-ODE style).
        与 shortcut 代码使用相同的插值公式。
        """
        a = 19.9
        b = 0.1
        
        # alpha_t = e^{(-1/4 a (1-t)^2-1/2 b(1-t))}
        alpha_t = torch.exp(- (0.25) * a * (1-t)**2 - (0.5) * b * (1-t))

        # beta_t = sqrt{1-alpha^2}
        beta_t = torch.sqrt(1 - alpha_t**2 + 1e-8)

        z_t = alpha_t * z1 + beta_t * z0
        
        return z_t

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
        
        # 1. Generate data at t_{n} and t_{n+1}
        z_current = self.get_ode_state_and_derivative(z0, z1, t_current)
        z_next = self.get_ode_state_and_derivative(z0, z1, t_next)
        
        # 2. Consistency Loss: F(z_next, t_next) ≈ F(z_current, t_current)
        pred_next = self.model(z_next, t_next)
        
        with torch.no_grad():
            target_current = self.target_model(z_current, t_current)
            
        consistency_loss = (pred_next - target_current).pow(2).mean()
        
        # 3. Supervision Loss: F(z_t, t) should predict z1 (the target)
        t_random = torch.rand((batch_size, 1)).to(z0.device)
        z_t = self.get_ode_state_and_derivative(z0, z1, t_random)
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
    # One-step generation
    generated_z1 = cm_trainer.sample_one_step(z0)

    # 计算 aver_dist（与 shortcut 代码一致）
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
    plt.scatter(generated_z1[:, 0].cpu().numpy(), generated_z1[:, 1].cpu().numpy(), c='#D9A0B3', label='Generated', alpha=0.6)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
    plt.title('Consistency Model\n(One-Step Generation)', fontsize=19)
    plt.tight_layout()
    plt.savefig('irr_circle_consistency_output.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# ==========================================
# 5. 执行训练
# ==========================================

# 准备数据对 (z0, z1) - 使用最近邻配对（与 DMD 一致，公平对比）
vertices_1_tensor = torch.tensor(vertices_1).float().to(device)

def create_paired_data(z0_samples):
    """使用最近邻匹配创建配对数据"""
    z0 = z0_samples.to(device)
    # 计算到所有目标中心的距离
    dists = torch.cdist(z0, vertices_1_tensor)
    min_idx = torch.argmin(dists, dim=1)
    # 获取对应的目标中心并添加噪声
    z1_centers = vertices_1_tensor[min_idx]
    z1 = z1_centers + torch.randn_like(z0) * np.sqrt(VAR)
    return z0, z1

x_0, x_1 = create_paired_data(samples_0)
z_pairs = torch.stack([x_0, x_1], dim=1).to(device)

print('z_pairs shape:', z_pairs.shape)

# 训练参数 (与 shortcut 保持一致)
iterations = 1000
batchsize = 2048
input_dim = 2

# 初始化模型和训练器
# hidden_num=100 使参数量与 shortcut MLP(hidden_num=100) 相近
model = ConsistencyModel(input_dim, hidden_num=100).to(device)
cm_trainer = ConsistencyTraining(model, num_steps=20)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

# 打印参数量
num_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {num_params:,}')

# 训练
cm_trainer, loss_curve = train_consistency_model(cm_trainer, optimizer, z_pairs, batchsize, iterations)

# 绘制 Loss 曲线
plt.figure(figsize=(6, 3))
plt.plot(loss_curve)
plt.title('Consistency Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# 绘图验证
draw_plot(cm_trainer, z0=initial_model.sample([600]).to(device), z1=target_model.sample([600]).to(device))
