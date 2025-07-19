import collections
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gym
import matplotlib.pylab as plt
import rl_utils

"""
经验回放池
"""
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, bitch_size):
        transitions = random.sample(self.buffer, bitch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)

"""
Q函数网络
"""    
class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

"""
Dueling_DQN网络配置
"""
class VAnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        V = self.fc_V(F.relu(self.fc1(x)))
        A = self.fc_A(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q

"""
DQN函数的主要实现过程
"""    
class DQN:
    def __init__(self, epsilon, device, state_dim, 
                 hidden_dim, action_dim, learning_rate, gamma, 
                 bitch_size, dtype='VanillaDQN'):
        self.epsilon = epsilon
        self.device = device
        self.action_dim = action_dim
        if dtype == 'DuelingDQN':
            self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(self.device)
            self.target_q_net = VAnet(state_dim, hidden_dim, action_dim).to(self.device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(self.device)
            self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(self.device)
        self.learning_rate = learning_rate
        self.bitch_size = bitch_size
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), self.learning_rate)
        self.dtype = dtype
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transitions):
        # 将变量都转化为张量的形式，便于神经网络的计算
        states = torch.tensor(transitions['state'], 
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions['action']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions['reward'], 
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transitions['next_state'], 
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions['done'], 
                             dtype=torch.float).view(-1, 1).to(self.device)

        qsa = self.q_net(states).gather(1, actions)
        # if self.dtype == 'DoubleDQN':
        actions_index = self.q_net(next_states).max(1)[1].view(-1, 1)
        Gt = self.target_q_net(next_states).gather(1, actions_index)
        # else:
        #     Gt = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        qsa_target = rewards + self.gamma * Gt * (1 - dones)

        # 损失函数的创建以及目标函数的拟合
        qsa_loss = torch.mean(F.mse_loss(qsa, qsa_target))
        self.optimizer.zero_grad()
        qsa_loss.backward()
        self.optimizer.step()

        if self.count % self.bitch_size == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

def dis_to_con(discrete_action, env, action_dim):
    low_bound = env.action_space.low[0]
    high_bound = env.action_space.high[0]
    return low_bound + (discrete_action / (action_dim - 1)) * (high_bound - low_bound)

def DQN_Fun(episode_num, env, agent, action_dim, buffer, 
        capacity, bitch_size):
    total_reward_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(episode_num/10), desc='Iration: %d' % (i + 1)) as pbar:
            for episode_i in range(int(episode_num/10)):
                total_reward = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    action_continuous = dis_to_con(action, env, action_dim)
                    next_state, reward, done, _ = env.step([action_continuous])
                    max_q_value = 0.005 * agent.max_q_value(state) + 0.995 * max_q_value
                    max_q_value_list.append(max_q_value)
                    buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    if buffer.size() > capacity:
                        s_, a_, r_, ns_, d_ = buffer.sample(bitch_size)
                        transition = {
                            'state': s_,
                            'action': a_,
                            'reward': r_,
                            'next_state': ns_,
                            'done': d_
                        }
                        agent.update(transition)
                total_reward_list.append(total_reward)
                pbar.update(1)
            
                if (episode_i + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (i * (episode_num / 10) + episode_i),
                        'return':
                        '%.3f' % np.mean(total_reward_list[-10:])
                    })

    return total_reward_list, max_q_value_list

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
episode_num = 200

env_name = 'Pendulum-v1'
env = gym.make(env_name)
env.seed(0)

epsilon = 0.01
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = 11
learning_rate = 1e-2
gamma = 0.98
target_update = 50
# agent = DQN(epsilon, device, state_dim, hidden_dim, action_dim, learning_rate, gamma, target_update)
# agent = DQN(epsilon, device, state_dim, hidden_dim, action_dim, 
#             learning_rate, gamma, target_update, dtype='DoubleDQN')
agent = DQN(epsilon, device, state_dim, hidden_dim, action_dim, 
            learning_rate, gamma, target_update, dtype='DuelingDQN')


capacity = 5000
buffer = ReplayBuffer(capacity)
minimal_size = 1000
bitch_size = 64

total_reward_list, max_q_value_list = DQN_Fun(episode_num, env, agent, action_dim, buffer, minimal_size, bitch_size)

# 绘图
total_reward_list_len = range(len(total_reward_list))
mv_return = rl_utils.moving_average(total_reward_list, 5)
plt.plot(total_reward_list_len, mv_return)
plt.xlabel('episode')
plt.ylabel('return')
plt.title('DQN on {}'.format(env_name))
plt.show()

max_q_value_list_len = range(len(max_q_value_list))
plt.plot(max_q_value_list_len, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('DQN on {}'.format(env_name))
plt.show()

