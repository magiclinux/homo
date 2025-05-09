#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import numpy as np
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn as nn

# 分布参数
VAR = 0.3  # 方差
M = 500  # 绘图范围
COMP = 300  # 外环的高斯分布数量
device = torch.device('cpu')

# 定义螺旋线函数
def spiral_function(theta):
    """r(θ) = 5 * (θ*5 + 10)^1.01"""
    return 5 * pow(theta * 5 + 10, 1.01)

# 设置螺旋线的圈数
num_turns = 2  # 螺旋线绕的圈数

# 生成螺旋分布（π_1）的均值
angles = [k * (2 * np.pi / COMP) for k in range(COMP * num_turns)]  # 螺旋线的角度范围
radii = [spiral_function(theta) for theta in angles]  # 使用函数生成半径
vertices_1 = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(radii, angles)]  # 各高斯分布的均值
spiral_means = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(radii, angles)]  # 螺旋线分布的均值

# 定义螺旋分布 π_1
spiral_mix = Categorical(torch.tensor([1 / len(spiral_means) for _ in range(len(spiral_means))]))  # 均匀权重
spiral_comp = MultivariateNormal(torch.tensor(spiral_means).float(),  # 螺旋线分布的均值
                                  VAR * torch.stack([torch.eye(2) for _ in range(len(spiral_means))]))  # 协方差
spiral_model = MixtureSameFamily(spiral_mix, spiral_comp)

# 定义中心分布 π_0_1
center_mean = torch.tensor([0.0, 0.0])
center_cov = VAR * torch.eye(2)
initial_model = MultivariateNormal(center_mean, center_cov)

# 定义外环分布 π_0_2
outer_radius = max(radii) + 50  # 外环的半径
outer_ring_angles = torch.rand(COMP) * 2 * np.pi  # 均匀分布的角度
outer_ring_radii = torch.normal(mean=outer_radius, std=5.0, size=(COMP,))  # 外环半径增加随机扰动
outer_ring_means = torch.stack([
    outer_ring_radii * torch.cos(outer_ring_angles),
    outer_ring_radii * torch.sin(outer_ring_angles)
], dim=1)
outer_covs = VAR * torch.eye(2).repeat(COMP, 1, 1)
outer_mix = Categorical(torch.tensor([1 / COMP for _ in range(COMP)]))  # 均匀权重
outer_comp = MultivariateNormal(outer_ring_means, outer_covs)
outer_model = MixtureSameFamily(outer_mix, outer_comp)

# 采样 π_0_1 和 π_0_2，然后拼接成 samples_0
samples_0_1 = initial_model.sample([300])  # 从中心分布采样 300 个点
samples_0_2 = outer_model.sample([300])  # 从外环分布采样 300 个点
samples_0 = torch.cat([samples_0_1, samples_0_2], dim=0)  # 合并为源分布样本点

# 从 π_1（螺旋分布）采样 600 个点
samples_1 = spiral_model.sample([600])

# 打印数据形状
print('Shape of the samples:')
print('π_0:', samples_0.shape)  # 源分布（中心 + 外环）
print('π_1:', samples_1.shape)  # 目标分布（螺旋）

# 绘制样本点
plt.figure(figsize=(4, 4))
plt.xlim(-M, M)
plt.ylim(-M, M)

# 设置标题和坐标轴标签的字体大小
plt.title(r'Samples from $\pi_0$ and $\pi_1$', fontsize=19)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.5, c='#BD8253', label=r'$\pi_0$')
plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.5, c='#2E59A7', label=r'$\pi_1$')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 12})

plt.tight_layout()
# plt.savefig('../figures/dotpluscircle_dataset.pdf', format='pdf', bbox_inches='tight')
plt.show()


# In[10]:


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t, d):
        inputs = torch.cat([x_input, t, d], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x

############################
# Mingda: I add the following model class
############################
class MLP_2nd_order(nn.Module):
    def __init__(self, input_dim=2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + input_dim + 1 + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, first_order_input, x_input, t, d):
        inputs = torch.cat([first_order_input, x_input, t, d], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x


# In[11]:


class RectifiedFlow2():
  def __init__(self, model = None, num_steps=1000):
    self.model = model
    self.first_order_model = self.model[0]
    self.second_order_model = self.model[1]
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

  def frist_and_second_order_predict(self, z_t, t, d):
    tmpd = d.clone()
    tmpd[tmpd < (1 / 128)] = 0
    first_order_pred = self.first_order_model(z_t, t, tmpd)
    second_order_pred = self.second_order_model(first_order_pred, z_t, t, tmpd)
    return first_order_pred, second_order_pred

  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]

    ############################
    # Mingda: I rewrite the inference code to second order version
    ############################

    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize,1)) * i / N
    #   pred = self.model(z, t)
      zero = torch.zeros_like(t)
      first_order_pred, second_order_pred = self.frist_and_second_order_predict(z, t, zero)
    #   z = z.detach().clone() + pred * dt
      z = z.detach().clone() + first_order_pred * dt + 0.5 * second_order_pred * dt**2

      traj.append(z.detach().clone())
    distances = torch.cdist(z, torch.tensor(vertices_1).float())

    min_distances, _ = torch.min(distances, dim=1) 

    average_min_distance = min_distances.mean().item()

    print("aver_dist", average_min_distance)

    return traj

  @torch.no_grad()
  def new_gt(self, first_order_gt, second_order_gt, z_t, t, d, flag): 
    tmpd = d.clone() / 2
    f_t = self.first_order_model(z_t, t, tmpd)
    s_t = self.second_order_model(f_t, z_t, t, tmpd)
    z_tpd = z_t + tmpd * f_t + 0.5 * tmpd**2 * s_t
    f_tpd = self.first_order_model(z_tpd, t + tmpd, tmpd)
    s_tpd = self.second_order_model(f_tpd, z_tpd, t + tmpd, tmpd)
    # for i in range(len(flag)):
    #   if flag[i] == 1:
    #     first_order_gt[i] = ( f_t[i] + f_tpd[i] ) / 2
    #     second_order_gt[i] = ( s_t[i] + s_tpd[i] ) / 2
    mask = (flag == 1).squeeze()
    first_order_gt[mask] = ( f_t[mask] + f_tpd[mask] ) / 2
    second_order_gt[mask] = ( s_t[mask] + s_tpd[mask] ) / 2

    return first_order_gt, second_order_gt


# In[12]:


from tqdm import tqdm

second_order_loss_scale = 1e-7
first_order_loss_scale = 1 - second_order_loss_scale

def get_gradient_norm(model):
  grad_norm_dict = {}
  grad_norm_sum = 0.0
  # Iterate through model parameters and compute gradient norm
  for name, param in model.named_parameters():
      if param.grad is not None:  # Check if gradients exist for the parameter
          grad_norm = param.grad.data.norm(2)  # L2 norm of the gradient
          grad_norm_dict[name] = grad_norm.item()
          # print(f"Gradient norm for {name}: {grad_norm.item()}")
          grad_norm_sum += grad_norm.item()

  return grad_norm_dict, grad_norm_sum

def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
  loss_curve = []
  for i in tqdm(range(inner_iters+1)):
    optimizer.zero_grad()
    indices = torch.randperm(len(pairs))[:batchsize]
    batch = pairs[indices]
    z0 = batch[:, 0].detach().clone().to(device)
    z1 = batch[:, 1].detach().clone().to(device)

    # z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)

    z_t, t, first_order_gt, second_order_gt = rectified_flow.get_train_tuple(z0=z0, z1=z1)

    # zt shape: [bs, 2]
    # t shape: [bs, 1]
    # d shape: [bs, 1]

    ############################    
    # Mingda: I randomly select 1 / 4 of the data to compute the Self-consistency loss at each iteration
    ############################
    d = torch.zeros_like(t)
    # randomly 1 / 4 flag are 1, others are 0
    flag = torch.zeros_like(t, dtype=torch.int)
    num_elements = t.numel()  # the number of elements in t
    num_ones = num_elements // 1  # the number of 1s
    indices = torch.randperm(num_elements)[:num_ones]  # Randomly select num_ones indices
    # Set these indices to 1
    flag[indices] = 1
    # When flag is 1, then set d \in (1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1)
    d[flag == 1] = 1 / 2**torch.randint(0, 8, (num_ones,)).to(device)
    # Change first,order_gt and second_order_gt to the flag == 1 part
    first_order_gt, second_order_gt = rectified_flow.new_gt(first_order_gt, second_order_gt, z_t, t, d, flag)
    
    
       

    first_order_pred, second_order_pred = rectified_flow.frist_and_second_order_predict(z_t, t, d)



    ############################
    # Mingda: I changed the loss term here
    ############################

    first_order_loss = (first_order_gt - first_order_pred).abs().pow(2).sum(dim=1)
    second_order_loss = (second_order_gt - second_order_pred).abs().pow(2).sum(dim=1)
    first_order_loss_mean = first_order_loss.mean()
    second_order_loss_mean = second_order_loss.mean()

    first_order_loss_mean = first_order_loss_scale * first_order_loss_mean
    second_order_loss_mean = second_order_loss_scale * second_order_loss_mean

    second_order_loss_mean = 0
    loss = first_order_loss_mean + second_order_loss_mean

    loss.backward()

    # first_order_grad_norm_dict, first_order_grad_norm_sum = get_gradient_norm(rectified_flow.first_order_model)
    # second_order_grad_norm_dict, second_order_grad_norm_sum = get_gradient_norm(rectified_flow.second_order_model)

    # print("first order grad norm is:", max(first_order_grad_norm_dict.values()))
    # print("second order grad norm is:", max(second_order_grad_norm_dict.values()))


    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return rectified_flow, loss_curve


# In[13]:


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
  plt.title('HOMO optimized\n with SC loss', fontsize=19)
  plt.tight_layout()
  plt.savefig('../figures/2_3_dotpluscircle_output.pdf', format='pdf', bbox_inches='tight') # Should change


  # traj_particles = torch.stack(traj)
  # plt.figure(figsize=(4,4))
  # plt.xlim(-M,M)
  # plt.ylim(-M,M)
  # plt.axis('equal')
  # for i in range(30):
  #   plt.plot(traj_particles[:, i, 0], traj_particles[:, i, 1])
  # plt.title('Transport Trajectory')
  # plt.tight_layout()
  # # plt.savefig("8_123_output_traj.pdf", format="pdf", bbox_inches="tight")


# ## 2-Rectified Flow
# Now let's use the *reflow* procedure to get a straightened rectified flow,
# denoted as 2-Rectified Flow, by repeating the same procedure on with $(X_0,X_1)$ replaced by  $(Z_0^1, Z_1^1)$, where   $(Z_0^1, Z_1^1)$ is the coupling simulated from 1-Rectified Flow.  
# Specifically, we randomly sample 10000 $Z_0^1$  and generate their corresponding  $Z_1^1$ by simulating 1-Rectified Flow.

# In[14]:


# ##############################
# Mingda: Hi guys, I only updated code above this remark. The code after this remark is untouched
# ##############################

iterations = 10000
batchsize = 2048
input_dim = 2
# z10 = samples_0.detach().clone()
# traj = rectified_flow_1.sample_ode(z0=z10.detach().clone(), N=100)
# z11 = traj[-1].detach().clone()
# z_pairs = torch.stack([z10, z11], dim=1)

x_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))].to(device)
x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))].to(device)
z_pairs = torch.stack([x_0, x_1], dim=1).to(device)

print(z_pairs.shape)


# In[15]:


iterations = 10000
batchsize = 2048
input_dim = 2

# reflow_iterations = 50
reflow_iterations = 180
model1 = MLP(input_dim, hidden_num=100).to(device)
model2 = MLP_2nd_order(input_dim, hidden_num=100).to(device)
modellist = nn.ModuleList([model1, model2]).to(device)
rectified_flow_2 = RectifiedFlow2(modellist, num_steps=100)
# import copy
# rectified_flow_2.net = copy.deepcopy(rectified_flow_1) # we fine-tune the model from 1-Rectified Flow for faster training.
optimizer = torch.optim.Adam(rectified_flow_2.model.parameters(), lr=5e-3)

rectified_flow_2, loss_curve = train_rectified_flow(rectified_flow_2, optimizer, z_pairs, batchsize, reflow_iterations)
# plt.plot(np.linspace(0, reflow_iterations, reflow_iterations+1), loss_curve[:(reflow_iterations+1)])


# In[16]:


# 采样 π_0_1 和 π_0_2，然后拼接成 samples_0
samples_0_1 = initial_model.sample([300])  # 从中心分布采样 300 个点
samples_0_2 = outer_model.sample([300])  # 从外环分布采样 300 个点
samples_0 = torch.cat([samples_0_1, samples_0_2], dim=0)  # 合并为源分布样本点

# 从 π_1（螺旋分布）采样 600 个点
samples_1 = spiral_model.sample([600])
draw_plot(rectified_flow_2, z0=samples_0, z1=samples_1.detach().clone())

