import torch
import torch.nn.functional as F
import gym
import rl_utils
import matplotlib.pyplot as plt

"""
策略网络，用来改善策略
"""
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

"""
价值网络，用于减小Q(s, a)的方差，提高策略梯度算法的精度
"""    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ActorCritic:
    def __init__(self, gamma, device, 
                 state_dim, hidden_dim, action_dim, 
                 critic_lr, actor_lr):
        self.gamma = gamma
        self.device = device
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transitions):
        # 将变量转化为张量, 方便神经网络的计算
        states = torch.tensor(transitions['states'], 
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions['rewards'], 
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transitions['next_states'],
                                  dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions['dones'], 
                             dtype=torch.float).view(-1, 1).to(self.device)

        target_qsa = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        delta = target_qsa - self.critic(states)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), target_qsa.detach()))

        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * delta.detach())

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.step()

# 创建环境
env_name = 'CartPole-v0'
env = gym.make(env_name)

# 创建随机种子
env.seed(0)
torch.manual_seed(0)

# 创建智能体
gamma = 0.98
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n
critic_lr = 1e-2
actor_lr = 1e-3
agent = ActorCritic(gamma, device, state_dim, hidden_dim, action_dim, critic_lr, actor_lr)

# 开始训练
num_episode = 1000
return_list = rl_utils.train_on_policy_agent(env, agent, num_episode)

return_list_len = list(range(len(return_list)))
plt.plot(return_list_len, return_list)
plt.xlabel('episode')
plt.ylabel('return')
plt.title('ActorCritic on {}'.format('env_name'))
plt.show()

mean_return_list = rl_utils.moving_average(return_list, 9)
plt.plot(return_list_len, mean_return_list)
plt.xlabel('episode')
plt.ylabel('return')
plt.title('ActorCritic on {}'.format('env_name'))
plt.show()