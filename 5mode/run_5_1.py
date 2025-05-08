#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn as nn

# 分布的半径
D_0 = 6.  # π_0 的半径（较小）
D_1 = 13.  # π_1 的半径（较大）
M = D_1 + 5  # 绘图范围
VAR = 0.3  # 方差
COMP = 5  # 组件数目（正五边形的5个顶点）
device = torch.device('cpu')

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

# 设置标题和坐标轴标签的字体大小
plt.title(r'Samples from $\pi_0$ and $\pi_1$', fontsize=19)

# 设置坐标轴刻度的字体大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.6, c = '#BD8253', label=r'$\pi_0$')
plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.6, c = '#2E59A7', label=r'$\pi_1$')

plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 12})

plt.tight_layout()
plt.savefig('5_dataset.pdf', format='pdf', bbox_inches='tight')
plt.show()


# In[2]:


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x


# In[3]:


class RectifiedFlow():
  def __init__(self, model=None, num_steps=1000):
    self.model = model
    self.N = num_steps

  def get_train_tuple(self, z0=None, z1=None):

 
    # Mingda: I compute the first and second order derivative of alpha_t and beta_t manually, and implement them as follows
    ############################

    # we need to exclude 1, since 1 will make first order beta to be inf
    t = (torch.rand((z1.shape[0], 1)) / (1 + 1e-6)).to(device)
    a = 19.9
    b = 0.1

    # alpha_t = e^{(-1/4 a (1-t)^2-1/2 b(1-t))}
    alpha_t = torch.exp(- (1/4) * a * (1-t)**2 - (1/2) * b * (1-t))

    # first order alpha:
    # d alpha_t / dt = alpha_t * 1/2 * (a (1-t) + b)
    first_order_alpha = alpha_t * (1/2) * (a * (1-t) + b)

    # second order alpha:
    # d^2 alpha_t / dt^2 = 1/2 * (alpha_t * (a(1-x)+b)^2 - a alpha_t)
    second_order_alpha = (1/2) * (alpha_t * (a * (1-t) + b)**2 - a * alpha_t)

    # beta_t = sqrt{1-alpha^2}
    beta_t = torch.sqrt(1 - alpha_t**2)
    # first order beta
    # d beta_t / dt = (- alpha  / sqrt{1 - alpha^2}) * (d alpha / dt)
    first_order_beta = (- alpha_t / torch.sqrt(1 - alpha_t**2)) * first_order_alpha
    # second order beta
    # d^2 beta_t / dt^2 = (- 1  / (1 - alpha^2) sqrt (1 - x^2)) * (d alpha / dt) + (- alpha  / sqrt{1 - alpha^2}) * (d^2 alpha / dt^2)
    second_order_beta = (- 1 / ((1 - alpha_t**2) * torch.sqrt(1 - alpha_t**2))) * first_order_alpha + first_order_beta * second_order_alpha


    z_t = alpha_t * z1 + beta_t * z0
    first_order_gt = first_order_alpha * z1 + first_order_beta * z0
    second_order_gt = second_order_alpha * z1 + second_order_beta * z0

    return z_t, t, first_order_gt, second_order_gt


  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]

    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1)) * i / N
      pred = self.model(z, t)
      z = z.detach().clone() + pred * dt

      traj.append(z.detach().clone())
    distances = torch.cdist(z, torch.tensor(vertices_1).float())

    min_distances, _ = torch.min(distances, dim=1) 

    average_min_distance = min_distances.mean().item()

    print("aver_dist", average_min_distance)

    return traj


# In[4]:


def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
  loss_curve = []
  for i in range(inner_iters+1):
    optimizer.zero_grad()
    indices = torch.randperm(len(pairs))[:batchsize]
    batch = pairs[indices]
    z0 = batch[:, 0].detach().clone()
    z1 = batch[:, 1].detach().clone()
    z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)

    pred = rectified_flow.model(z_t, t)
    loss = abs((target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1))
    loss = loss.mean()
    loss.backward()

    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return rectified_flow, loss_curve


# In[5]:


@torch.no_grad()
def draw_plot(rectified_flow, z0, z1, N=None):
  traj = rectified_flow.sample_ode(z0=z0, N=N)

  plt.figure(figsize=(4,4))
  plt.xlim(-M,M)
  plt.ylim(-M,M)

  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)

  plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), c='#BD8253' , label=r'$\pi_0$', alpha=0.6)
  plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), c='#2E59A7', label=r'$\pi_1$', alpha=0.6)
  plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), c='#D9A0B3' , label='Generated', alpha=0.6)
  plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), prop={'size': 9})
  plt.title('HOMO optimized\n with M1 loss', fontsize=19)
  plt.tight_layout()
  plt.savefig('5_1_output.pdf', format='pdf', bbox_inches='tight') # Should change

 

  # traj_particles = torch.stack(traj)
  # plt.figure(figsize=(4,4))
  # plt.xlim(-M,M)
  # plt.ylim(-M,M)
  # plt.axis('equal')
  # for i in range(30):
  #   plt.plot(traj_particles[:, i, 0], traj_particles[:, i, 1])
  # plt.title('Transport Trajectory')
  # plt.tight_layout()
  # # plt.savefig('8_2_output_traj.pdf', format='pdf', bbox_inches='tight')


# In[6]:


x_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))]
x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
x_pairs = torch.stack([x_0, x_1], dim=1)
print(x_pairs.shape)


# In[7]:


iterations = 100
batchsize = 2048
input_dim = 2

rectified_flow_1 = RectifiedFlow(model=MLP(input_dim, hidden_num=100), num_steps=100)
optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr=5e-3)

rectified_flow_1, loss_curve = train_rectified_flow(rectified_flow_1, optimizer, x_pairs, batchsize, iterations)
# plt.plot(np.linspace(0, iterations, iterations+1), loss_curve[:(iterations+1)])
# plt.title('Training Loss Curve')


# In[8]:


z10 = samples_0.detach().clone()
traj = rectified_flow_1.sample_ode(z0=z10.detach().clone(), N=100)
z11 = traj[-1].detach().clone()
z_pairs = torch.stack([z10, z11], dim=1)
# print(z_pairs.shape)


# In[9]:


reflow_iterations = 1000

rectified_flow_2 = RectifiedFlow(model=MLP(input_dim, hidden_num=100), num_steps=100)
import copy
rectified_flow_2.net = copy.deepcopy(rectified_flow_1) # we fine-tune the model from 1-Rectified Flow for faster training.
optimizer = torch.optim.Adam(rectified_flow_2.model.parameters(), lr=5e-3)

rectified_flow_2, loss_curve = train_rectified_flow(rectified_flow_2, optimizer, z_pairs, batchsize, reflow_iterations)
# plt.plot(np.linspace(0, reflow_iterations, reflow_iterations+1), loss_curve[:(reflow_iterations+1)])


# In[10]:


draw_plot(rectified_flow_2, z0=initial_model.sample([500]), z1=target_model.sample([500]).detach().clone())

