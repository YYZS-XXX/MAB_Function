import copy
import gym

class CliffWalking():
    def __init__(self, nrow=4, ncol=12):
        # 对该问题进行数学建模
        self.nrow = nrow
        self.ncol = ncol
        self.P = self.create_cliff()

    def create_cliff(self):
        # 生成每一步的信息载体
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        P = [[[] for i in range(4)] for j in range(self.nrow * self.ncol)]
        # 执行相应的动作 change[0]:上 change[1]:下 change[2]:左 change[3]:右
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    done = 0
                    # 当当前位置为悬崖或者目标点时
                    if i == self.nrow - 1 and j > 0:
                        P[j + i * self.ncol][a] = [(1, j + i * self.ncol, 0, True)]
                        continue
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    reward = -1
                    done = False
                    # 当下一步的位置为悬崖或者目标点时
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != (self.ncol - 1):
                            reward = -100
                    P[j + i * self.ncol][a] = [(1, next_x + next_y * self.ncol, reward, done)]
        return P

class ValueIteration():
    def __init__(self, env, gamma, theta):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value = [0] * self.env.ncol * self.env.nrow
        self.pi = [[None for i in range(4)]
                   for j in range(self.env.ncol * self.env.nrow)]

    def value_access(self):
        cnt = 1
        while 1:
            new_v = [0] * self.env.ncol * self.env.nrow
            diff_max = 0
            for s in range(self.env.ncol * self.env.nrow):
                value_list = []
                for a in range(4):
                    value = 0
                    for res in self.env.P[s][a]:
                        p, next_state, reward, done = res
                        value += p * (reward + self.gamma * self.value[next_state] * (1 - done))
                    value_list.append(value)
                new_v[s] = max(value_list)
                diff_max = max(diff_max, abs(new_v[s] - self.value[s]))
            self.value = new_v
            if diff_max < self.theta: break
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.policy_improvement()

    def policy_improvement(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_max = 0
            count = 0
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, reward, done = res
                    qsa += (reward + self.gamma * p * self.value[next_state] * (1 - done))
                qsa_list.append(qsa)
            qsa_max = max(qsa_list)
            count = qsa_list.count(qsa_max)
            self.pi[s] = [1/count if q == qsa_max else 0 for q in qsa_list]
                    

class PolicyIteration():
    def __init__(self, gamma, env, theta):
        self.gamma = gamma
        self.env = env
        self.pi =[[0.25, 0.25, 0.25, 0.25] for i in range(self.env.nrow * self.env.ncol)]
        self.value = [0] * self.env.nrow * self.env.ncol
        self.theta = theta

    """
    策略评估
    """
    def policy_access(self):
        cnt = 1
        while 1:
            max_diff = 0
            value_sum = [0] * self.env.nrow * self.env.ncol
            for s in range(self.env.nrow * self.env.ncol):
                value_part_list = []
                for a in range(4):
                    value_part = 0
                    for res in self.env.P[s][a]:
                        p, next_state, reward, done = res
                        value_part += p * (reward + self.gamma * self.value[next_state] * (1 - done))
                    value_part_list.append(self.pi[s][a] * value_part)
                value_sum[s] = sum(value_part_list)
                max_diff = max(max_diff, abs(value_sum[s] - self.value[s]))
            self.value = value_sum
            if max_diff < self.theta: break
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)
    
    """
    策略提升
    """
    def policy_improvement(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_part_list = []
            for a in range(4):
                qsa_part = 0
                for res in self.env.P[s][a]:
                    p, next_state, reward, done = res
                    qsa_part += p * (reward + self.gamma * self.value[next_state] * (1 - done))
                qsa_part_list.append(qsa_part)
            qsa_max = max(qsa_part_list)
            count = qsa_part_list.count(qsa_max)
            self.pi[s] = [1/count if q == qsa_max else 0 for q in qsa_part_list]
        print("策略提升完成")
        return self.pi
    
    """
    策略迭代
    """
    def policy_iteration(self):
        while 1:
            self.policy_access()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break

def print_agent(agent, action_meaning, disaster = [], end = []):
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.value[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print("****", end = " ")
            elif (i * agent.env.ncol + j) in end:
                print("EEEE", end = " ")
            else:
                string = ''
                for a in range(len(action_meaning)):
                    if agent.pi[i * agent.env.ncol + j][a] > 0:
                        string += action_meaning[a]
                    else:
                        string += 'o'
                print(string, end = ' ')
        print()

"""
策略迭代
"""
# env = CliffWalking()
# gamma = 0.9
# theta = 0.001
# action_meaning = ['^', 'v', '<', '>']
# agent = PolicyIteration(gamma, env, theta)
# agent.policy_iteration()
# print_agent(agent, action_meaning, list(range(37, 47)), [47])

"""
价值迭代
"""
# env = CliffWalking()
# gamma = 0.9
# theta = 0.001
# action_meaning = ['^', 'v', '<', '>']
# agent = ValueIteration(env, gamma, theta)
# agent.value_access()
# print_agent(agent, action_meaning, list(range(37, 47)), [47])

"""
调用gym库，应用其冰湖环境进行策略迭代测试
"""
env = gym.make("FrozenLake-v1")
env = env.unwrapped
env.render()

gamma = 0.9
theta = 1e-5
action_meaning = ['<', 'v', '>', '^']
agent = PolicyIteration(gamma, env, theta)
agent.policy_iteration()
print("策略迭代")
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

agent = ValueIteration(env, gamma, theta)
print("\n价值迭代")
agent.value_access()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

"""
测试
"""
# P = CliffWalking()
# for res in P.P[1][1]:
#     print(res, "\n")                        