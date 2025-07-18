import collections
import random
import numpy as np
import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import rl_utils

"""
经验回放，用于解决MDP过程不独立的问题，同时提高样本的速率
"""
class ReplayBuffer:
    def __init__(self, capacity):  # 创建队列
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):  # 向队列中添加新的成员
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, bitch_size):  # 从经验回放池中进行采样
        transitions = random.sample(self.buffer, bitch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

"""
创建Q网络
"""
class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQN:
    def __init__(self, epsilon, state_dim, hidden_dim, 
                 action_dim, device, gamma, learning_rate, target_update):
        """
        Q_Learning的一些参数
        """
        self.epsilon = epsilon
        self.gamma = gamma

        """
        神经网络的一些参数
        """
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.count = 0
        self.target_update = target_update

        # 创建网络
        self.qnet = Qnet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_qnet = Qnet(state_dim, hidden_dim, action_dim).to(self.device)

        # 设置优化器
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype = torch.float).to(self.device)
            action = self.qnet(state).argmax().item()
        return action
    
    def update(self, transition_dict):
        """
        将变量转化为张量的形式
        """
        states = torch.tensor(transition_dict['state'], 
                              dtype = torch.float).to(self.device)
        actions = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['reward'],
                               dtype = torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_state'],
                                    dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['done'], 
                             dtype=torch.float).view(-1, 1).to(self.device)
        
        qsa = self.qnet(states).gather(1, actions)
        qsa_max = self.target_qnet(next_states).max(1)[0].view(-1, 1)
        target_qsa = rewards + self.gamma * qsa_max * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(qsa, target_qsa))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_qnet.load_state_dict(
                self.qnet.state_dict()
            )
        
        self.count += 1

# 创建环境
env_name = 'CartPole-v0'
env = gym.make(env_name)
# 创建随机变量种子
np.random.seed(0)
random.seed(0)
env.seed(0)
torch.manual_seed(0)
# 创建智能体
epsilon = 0.01
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128
device = 'cuda'
gamma = 0.98
learning_rate = 2e-3
target_update = 10
agent = DQN(epsilon, state_dim, hidden_dim, action_dim,
            device, gamma, learning_rate, target_update)
# 创建经验回放队列
buffer_size = 10000
minimal_size = 500
bitch_size = 64
buffer = ReplayBuffer(buffer_size)
# 关于训练的一些变量
episode_num = 500
episode_return = []
for i in range(10):
    with tqdm(total = int(episode_num/10), desc = 'Iration: %d' % i) as pbar:
        for episode_i in range(int(episode_num/10)):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                buffer.add(state, action, reward, next_state, done)  # 向队列中添加采样的成员
                state = next_state
                total_reward += reward
                if buffer.size() > minimal_size:
                    s_, a_, r_, n_, d_ = buffer.sample(bitch_size)
                    transition_dict = {
                        'state': s_,
                        'action': a_,
                        'reward': r_,
                        'next_state': n_,
                        'done': d_,
                    }
                    agent.update(transition_dict)
            episode_return.append(total_reward)
            if (episode_i + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (i * (episode_num/10) + episode_i + 1),
                    'return':
                    '%.3f' % np.mean(episode_return[-10:])
                })
            pbar.update(1)

episode_return_len = range(len(episode_return))
plt.plot(episode_return_len, episode_return)
plt.xlabel('episode')
plt.ylabel('return')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(episode_return, 9)
plt.plot(episode_return_len, mv_return)
plt.xlabel('episode_return_len')
plt.ylabel('mv_return')
plt.title('DQN on {}'.format(env_name))
plt.show()