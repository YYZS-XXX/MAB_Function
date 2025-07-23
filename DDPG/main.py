import torch
import torch.nn.functional as F
import rl_utils
import gym
import numpy as np
import matplotlib.pyplot as plt

# 创建策略函数类
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

# 创建价值函数类    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cout = torch.nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        cat = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.cout(x)
    
class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, device, 
                 sigma, gamma, tua, action_bound):
        # 创建价值网络以及其目标函数
        self.critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 创建策略网络以及其目标函数
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 确定优化参数
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.device = device
        self.sigma = sigma
        self.action_dim = action_dim
        self.gamma = gamma
        self.tua = tua

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        action = self.tua * np.random.randn(self.action_dim) + action
        return action
    
    def soft_update(self, net, target_net):
        for params, target_params in zip(net.parameters(), target_net.parameters()):
            target_params.data.copy_(self.sigma * params + (1.0 - self.sigma) * target_params)

    def update(self, transitions):
        # 将参数张量化
        states = torch.tensor(transitions['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transitions['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 策略价值函数的更新
        next_q_value = self.target_critic(next_states, self.target_actor(next_states))
        target_qsa = rewards + self.gamma * next_q_value * (1 - dones)
        qsa = self.critic(states, actions)
        critic_loss = torch.mean(F.mse_loss(qsa, target_qsa))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # 动作价值函数的更新
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)  # 软更新目标策略函数

# 创建环境
env_name = 'Pendulum-v1'
env = gym.make(env_name)

# 确定随机种子
env.seed(0)
np.random.seed(0)
torch.manual_seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 64
actor_lr = 3e-4
critic_lr = 3e-3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
sigma = 0.005
gamma = 0.98
tua = 0.01
action_bound = env.action_space.high[0]
agent = DDPG(state_dim, hidden_dim, action_dim,  actor_lr, critic_lr, device, sigma, gamma, tua, action_bound)

capacity = 10000
replay_buffer = rl_utils.ReplayBuffer(capacity)
num_episode = 200
minimal_size = 1000
batch_size = 64
return_list = rl_utils.train_off_policy_agent(env, agent, num_episode, replay_buffer, minimal_size, batch_size)

return_list_len = list(range(len(return_list)))
plt.plot(return_list_len, return_list)
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.title('DDPG on {}'.format(env_name))
plt.show()

return_list_mean = rl_utils.moving_average(return_list, 9)
plt.plot(return_list_len, return_list_mean)
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.title('DDPG on {}'.format(env_name))
plt.show()