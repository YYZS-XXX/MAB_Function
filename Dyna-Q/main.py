import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time

class CliffWalking:
    def __init__(self, nrow = 4, ncol = 12):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0
        self.y = self.nrow - 1

    def step(self, action):
        action_total = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + action_total[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + action_total[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        
        return next_state, done, reward
    
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
    
class DynaQ:
    def __init__(self, gamma, alpha, epsilon, ncol, nrow, n_action, n_qplanning):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_action = n_action
        self.Q_table = np.zeros([ncol * nrow, n_action])
        self.model = dict()
        self.n_qplanning = n_qplanning

    def take_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.Q_table[state])
        else:
            action = np.random.randint(self.n_action)
        return action
    
    def best_action(self, state):
        a_max = max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == a_max:
                a[i] = 1
        return a
    
    def q_learning(self, s0, a0, r, s1):
        G = r + self.gamma * self.Q_table[s1].max()
        self.Q_table[s0, a0] += self.alpha * (G - self.Q_table[s0, a0])

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1
        for _ in range(self.n_qplanning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)

def Dynaq_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = CliffWalking()
    gamma = 0.9
    alpha = 0.1
    epsilon = 0.01
    n_action = 4
    n_qplanning = n_planning
    agent = DynaQ(gamma, alpha, epsilon, ncol, nrow, n_action, n_qplanning)
    episode_num = 300
    episode_reward_list = []
    for i in range(10):
        with tqdm(total = int(episode_num / 10), desc = 'Iration: %d' % i) as pbar:
            for j in range(int(episode_num / 10)):
                state = env.reset()
                episdoe_reward = 0
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, done, reward = env.step(action)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    episdoe_reward += reward
                episode_reward_list.append(episdoe_reward)
                if len(episode_reward_list) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (i * (episode_num / 10) + j + 1),
                        'return':
                        '%.3f' % np.mean(episode_reward_list[-10:])
                    })
                pbar.update(1)
    
    return episode_reward_list

np.random.seed(0)
random.seed(0)
dynaq_step_list = [0, 2, 20]
for dynaq_step in dynaq_step_list:
    print('Q_Planning步数: %d' % dynaq_step)
    time.sleep(0.5)
    episode_return = Dynaq_CliffWalking(dynaq_step)
    episode_len = list(range(len(episode_return)))
    plt.plot(episode_len, episode_return, label = str(dynaq_step) + ' planning steps')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Dyna_q on {}'.format('Cliff Walking'))
plt.show()