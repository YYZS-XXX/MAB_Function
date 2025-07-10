import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class CliffWalking():
    def __init__(self, nrow = 4, ncol = 12):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0
        self.y = self.nrow - 1

    def step(self, action):
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        done = False
        reward = -1
        if self.x > 0 and self.y == (self.nrow - 1):
            done = True
            if self.x != (self.ncol - 1):
                reward = -100
        return next_state, done, reward
    
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

    
class Sarsa:
    def __init__(self, gamma, ncol, nrow, epsilon, alpha, n_action = 4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_action = n_action

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update_Q(self, s0, a0, r, s1, a1):
        err = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * err

class nstep_Sarsa:
    def __init__(self, n, epsilon, ncol, nrow, n_action, gamma, alpha):
        self.n = n
        self.n_action = n_action
        self.Q_table = np.zeros([ncol * nrow, n_action])
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.state_list = []
        self.action_list = []
        self.reward_list = []

    def take_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.Q_table[state])
        else:
            action = np.random.randint(self.n_action)
        return action
    
    def best_action(self, state):
        max_a = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == max_a:
                a[i] = 1
        return a
    
    def update(self, n, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == n:
            G = self.Q_table[s1, a1]
            for i in reversed(range(n)):
                G = self.gamma * G + self.reward_list[i]
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            r = self.reward_list.pop(0)
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []

class QLearning:
    def __init__(self, gamma, alpha, epsilon, n_action, ncol, nrow):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_action = n_action
        self.nrow = nrow
        self.ncol = ncol
        self.Q_table = np.zeros([ncol * nrow, self.n_action])

    def take_action(self, state):
        if np.random.random() > epsilon:
            action = np.argmax(self.Q_table[state])
        else:
            action = np.random.randint(self.n_action)
        return action
    
    def best_action(self, state):
        a = [0 for _ in range(self.n_action)]
        a_max = max(self.Q_table[state])
        for i in range(self.n_action):
            if a_max == self.Q_table[state][i]:
                a[i] = 1
        return a
    
    def update(self, s0, a0, r, s1):
        G = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0][a0]
        self.Q_table[s0][a0] += self.alpha * G

def print_agent(Sarsa, ncol, nrow, action_meaning, disaster = [], end = []):
    for i in range(nrow):
        for j in range(ncol):
            location = i * ncol + j
            if location in disaster:
                print("****", end=' ')
            elif location in end:
                print("EEEE", end=' ')
            else:
                a = Sarsa.best_action(location)
                pi_str = ''
                for o in range(len(action_meaning)):
                    pi_str += action_meaning[o] if a[o] > 0 else 'o'
                print(pi_str, end=' ')
        print()
"""
单步Sarsa算法
"""
# 创建环境
# gamma = 0.9
# ncol = 12
# nrow = 4
# epsilon = 0.1
# alpha = 0.1
# env = CliffWalking(nrow, ncol)
# np.random.seed(0)
# algorithm = Sarsa(gamma, ncol, nrow, epsilon, alpha)
# practice_num = 500
# epsidoes_list = []
# for i in range(10):
#     with tqdm(total=int(practice_num / 10), desc="Iteration %d" % i) as pdar:
#         for num in range(int(practice_num / 10)):
#             state = env.reset()
#             action = algorithm.take_action(state)
#             epsiodes_return = 0
#             done = False
#             while not done:
#                 next_state, done, reward = env.step(action)
#                 next_action = algorithm.take_action(next_state)  # 获取下一次的动作
#                 algorithm.update_Q(state, action, reward, next_state, next_action)
#                 state = next_state
#                 action = next_action
#                 epsiodes_return += reward
#             epsidoes_list.append(epsiodes_return)
#             if (num + 1) % 10 == 0:
#                 pdar.set_postfix({
#                     'episode':
#                     '%d' % (practice_num / 10 * i + num + 1), 
#                     'return':
#                     '%.3f' % np.mean(epsidoes_list[-10:])
#                 })
#             pdar.update(1)

# return_len = list(range(len(epsidoes_list)))
# plt.plot(return_len, epsidoes_list)
# plt.xlabel("episodes")
# plt.ylabel("returns")
# plt.title("Sarsa on {}" .format("cliffwalking"))
# plt.show()

# action_meaning = ['^', 'v', '<', '>']                
# print('Sarsa算法最终收敛得到的策略为：')                
# print_agent(algorithm, ncol, nrow, action_meaning, list(range(37, 47)), [47])

"""
多步Sarsa算法
"""
# np.random.seed(0)
# n = 5  # 5步Sarsa算法
# ncol = 12
# nrow = 4
# env = CliffWalking()
# epsilon = 0.1
# n_action = 4
# gamma = 0.9
# alpha = 0.1
# agent = nstep_Sarsa(n, epsilon, ncol, nrow, n_action, gamma, alpha)  # 创建智能体
# episode_num = 500  # 训练轮数为500轮
# episode_reward_list = []
# for i in range(int(episode_num / 50)):
#     with tqdm(total = int(episode_num / 10), desc = 'Iteration: %d' % i) as pbar:
#         for i_episode in range(int(episode_num / 10)):
#             state = env.reset()
#             action = agent.take_action(state)
#             done = False
#             episode_reward = 0
#             while not done:
#                 next_state, done, reward = env.step(action)
#                 next_action = agent.take_action(next_state)
#                 episode_reward += reward
#                 agent.update(n, state, action, reward, next_state, next_action, done)
#                 state = next_state
#                 action = next_action
#             episode_reward_list.append(episode_reward)
#             if i_episode % 10 == 0:
#                 pbar.set_postfix({
#                     'episode':
#                     '%d' % (i * int(episode_num / 10) + i_episode), 
#                     'return':
#                     '%.3f' % np.mean(episode_reward_list[-10:])
#                 })
#             pbar.update(1)

# # 绘制奖励图片
# reward_len = range(len(episode_reward_list))
# plt.plot(reward_len, episode_reward_list)
# plt.xlabel('episodes')
# plt.ylabel('Returns')
# plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
# plt.show()

# action_meaning = ['^', 'v', '<', '>']
# print('5步Sarsa算法最终收敛得到的策略为：')
# print_agent(agent, ncol, nrow, action_meaning, list(range(37, 47)), [47])

"""
Q_Learning算法
"""
np.random.seed(0)
nrow = 4
ncol = 12
env = CliffWalking()
gamma = 0.9
alpha = 0.1
epsilon = 0.1
n_action = 4
agent = QLearning(gamma, alpha, epsilon, n_action, ncol, nrow)
episode_num = 500
episode_reward_list = []
for i in range(10):
    with tqdm(total = int(episode_num / 10), desc = 'Iration: %d' % i) as pbar:
        for j in range(int(episode_num / 10)):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, done, reward = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                episode_reward += reward
            episode_reward_list.append(episode_reward)
            if len(episode_reward_list) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (i * int(episode_num/10) + j),
                    'return':
                    '%.3f' % np.mean(episode_reward_list[-10:])
                })
            pbar.update(1)
                
return_len = range(len(episode_reward_list))
plt.plot(return_len, episode_reward_list)
plt.xlabel('episode')
plt.ylabel('return')
plt.title('Q_Learning on {}'.format('cliff walking'))
plt.show()

action_meaning = ['^', 'v', '<', '>']
print('Q_Learning算法最终收敛得到的策略为：')
print_agent(agent, ncol, nrow, action_meaning, list(range(37, 47)), [47])