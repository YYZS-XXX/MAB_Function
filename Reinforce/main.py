import torch
import torch.nn.functional as F
import gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class RFnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(RFnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class Reinforce:
    def __init__(self, state_dim, hidden_dim, action_dim, device, 
                 gamma, learning_rate):
        self.gamma = gamma

        self.device = device
        self.policy_net = RFnet(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        # 归一化概率分布
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transitions):
        states = transitions['states']
        actions = transitions['actions']
        rewards = transitions['rewards']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(states))):
            state = torch.tensor([states[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([actions[i]]).view(-1, 1).to(self.device)
            reward = rewards[i]

            G = self.gamma * G + reward
            log_pi = torch.log(self.policy_net(state).gather(1, action))
            loss = -G * log_pi
            loss.backward()
        self.optimizer.step()

# 创建环境
env_name = 'CartPole-v0'
env = gym.make(env_name)

# 创建一些种子数，保证实验结果的确定性
env.seed(0)
torch.manual_seed(0)

# 创建智能体
state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
gamma = 0.98
learning_rate = 1e-3
agent = Reinforce(state_dim, hidden_dim, action_dim, device, 
                  gamma, learning_rate)

# 关于训练的一些参数
episode_num = 1000
total_reward_list = []

# 开始训练
for i in range(10):
    with tqdm(total=int(episode_num / 10), desc='%d' % (i + 1)) as pbar:
        for episode_i in range(int(episode_num / 10)):
            state = env.reset()
            total_reward = 0
            done = False
            transitions = {
                'states': [],
                'actions': [],
                'rewards': []
            }
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                transitions['states'].append(state)
                transitions['actions'].append(action)
                transitions['rewards'].append(reward)
                state = next_state
            total_reward_list.append(total_reward)
            agent.update(transitions)
            if ((episode_i + 1) % 10) == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (i * int(episode_num / 10) + episode_i + 1),
                    'return':
                    '%.3f' % np.mean(total_reward_list[-10:])
                })
            pbar.update(1)

# 绘图
total_reward_list_len = range(len(total_reward_list))
plt.plot(total_reward_list_len, total_reward_list)
plt.xlabel('episode')
plt.ylabel('return')
plt.title('Reinforce on {}'.format(env_name))
plt.show()

mean_reward = rl_utils.moving_average(total_reward_list, 9)
mean_reward_len = range(len(mean_reward))
plt.plot(mean_reward_len, mean_reward)
plt.xlabel('episode')
plt.ylabel('mean_return')
plt.title('Reinforce on {}'.format(env_name))
plt.show()