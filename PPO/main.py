import torch
import torch.nn.functional as F
import rl_utils
import gym
import matplotlib.pyplot as plt

# 策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))

# 更新一个连续策略网络
class PolicyNetContinue(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinue, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

# 价值网络    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))
    
class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, 
                 actor_lr, critic_lr, device, gamma, lmbda, alpha, 
                 circle_num, theta, env_name):
        # 初始化策略网络
        if env_name == 'CartPole-v0':
            self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        else:
            self.actor = PolicyNetContinue(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), actor_lr)

        self.env_name = env_name

        # 初始化价值网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), critic_lr)

        self.device = device
        self.gamma = gamma
        self.circle_num = circle_num
        self.theta = theta  # 截断因子

        # 广义优势函数求解过程
        self.alpha = alpha
        self.lmbda = lmbda

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        if self.env_name == 'CartPole-v0':
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
        else:
            mu, std = self.actor(state)
            action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]
    
    def update(self, transitions):
        states = torch.tensor(transitions['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transitions['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 对reward进行一定的变形
        rewards = (rewards + 8.0) / 8.0
        # 求出优势函数
        target_qsa = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        qsa = self.critic(states)
        td_delta = target_qsa - qsa
        advantage = rl_utils.compute_advantage(self.alpha, self.lmbda, td_delta.cpu()).to(self.device)

        # 求出重要性采样的权重中的旧策略
        if self.env_name == 'CartPole-v0':
            old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        else:
            mu, std = self.actor(states)
            action_dist = torch.distributions.Normal(mu.detach(), std.detach())
            old_log_probs = action_dist.log_prob(actions).detach()
        for _ in range(self.circle_num):
            if self.env_name == 'CartPole-v0':
                new_log_probs = torch.log(self.actor(states).gather(1, actions))
            else:
                mu, std = self.actor(states)
                action_dist = torch.distributions.Normal(mu, std)
                new_log_probs = action_dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            # 截断
            actor_loss = torch.mean(-torch.min(ratio * advantage.detach(),
                                   torch.clamp(ratio, 1 - self.theta, 1 + self.theta) * advantage.detach()))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), target_qsa.detach()))

            # 梯度优化
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

# 创建环境
# env_name = 'CartPole-v0'
env_name = 'Pendulum-v1'
env = gym.make(env_name)

# 固定随机种子数
env.seed(0)
torch.manual_seed(0)

if env_name == 'CartPole-v0':
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
else:
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.shape[0]
# 不同的环境对应不同的参数
if env_name == 'CartPole-v0':
    actor_lr = 1e-3
    critic_lr = 1e-2
    gamma = 0.98
    lmbda = 0.95
    alpha = 0.98
    circle_num = 10
    theta = 0.2
else:
    actor_lr = 1e-4
    critic_lr = 5e-3
    gamma = 0.9
    lmbda = 0.9
    alpha = 0.9
    circle_num = 10
    theta = 0.2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
agent = PPO(state_dim, hidden_dim, action_dim, 
            actor_lr, critic_lr, device, gamma, lmbda, alpha, 
            circle_num, theta, env_name)
# 训练次数
num_episode = 2000
return_list = rl_utils.train_on_policy_agent(env, agent, num_episode)

# 画图
return_list_len = list(range(len(return_list)))
plt.plot(return_list_len, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('PPO on {}'.format(env_name))
plt.show()

return_mean = rl_utils.moving_average(return_list, 9)
plt.plot(return_list_len, return_mean)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('PPO on {}'.format(env_name))
plt.show()