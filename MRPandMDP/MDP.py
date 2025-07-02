import numpy as np

S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s2", "前往s1", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移概率
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}

gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}

# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}

"""
将MDP转化为MRP,以此来计算value的值
"""
mrp_p = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0]
]

mrp_r = [-0.5, -1.5, -1, 5.5, 0.0]

def join(str1, str2):
    return str1 + '-' + str2

"""
贝尔曼方程求解
"""
def compute(rewards, gamma, P, state_num):
    rewards = np.array(rewards).reshape(-1, 1)
    Value = np.dot(np.linalg.inv(np.eye(state_num, state_num) - gamma * P),
                   rewards)
    return Value

"""
蒙特卡洛采样法
"""
def sample(MDP, Times, max_step, Pi):
    S, A, P, R, gamma = MDP
    # 用来存储程序遍历的路径
    episodes = []
    # Times为中最大遍历步长
    for _ in range(Times):
        episode = []
        s = S[np.random.randint(4)]
        time_count = 0
        """
        整个过程可以包含以下两步
        1、由策略确定动作
        2、由动作和当前状态确定下一步的状态
        """
        while s != "s5" and time_count <= max_step:
            time_count += 1
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if rand < temp:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break

            rand, temp = np.random.rand(), 0
            for s_opt in S:
               temp += P.get(join(join(s, a), s_opt), 0)
               if rand < temp:
                   s_next = s_opt
                   break
            episode.append((s, a, r, s_next))   
            s = s_next
        episodes.append(episode)
    return episodes

"""
通过蒙特卡洛的方式求解每个状态的价值函数
相当于利用穷举法计算出了价值函数（大树定理）
"""
def MC(episodes, gamma, V, N):
    for epsiode in episodes:
        G = 0
        for i in range(len(epsiode)-1, -1, -1):
            (s, a, r, s_next) = epsiode[i]
            G = gamma*G + r
            N[s] += 1
            V[s] = V[s] + (G - V[s])/N[s]

"""
求解占用度量
主要是为了求解在相同时刻，出现某种状态的概率，以此来求解占用度量
"""
def occupancy(max_step, epsiodes, s, a, gamma):
    result = 0
    total_times = np.zeros(max_step)
    occupy_times = np.zeros(max_step)
    for epsiode in epsiodes:
        for i in range(len(epsiode)):
            (s_opt, a_opt, r, s_next) = epsiode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occupy_times[i] += 1
    for i in reversed(range(max_step)):
        if total_times[i]:
            result += gamma ** i * occupy_times[i] / total_times[i]
    return (1 - gamma) * result

"""
利用公式法求解贝尔曼方程
"""
mrp_p = np.array(mrp_p)    
value = compute(mrp_r, gamma, mrp_p, 5)
print("MDP中每个状态的价值函数为\n", value)

"""
利用蒙特卡洛法求解价值函数
"""
episodes = sample(MDP, 1000, 20, Pi_1)
print("第一条序列\n", episodes[0])
print("第二条序列\n", episodes[1])
print("第五条序列\n", episodes[4])
gamma = 0.5
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
MC(episodes, gamma, V, N)
print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)

"""
求解占用度量
"""
gamma = 0.5
max_step = 1000

episodes_1 = sample(MDP, max_step, 1000, Pi_1)
episodes_2 = sample(MDP, max_step, 1000, Pi_2)
rho_1 = occupancy(max_step, episodes_1, "s4", "概率前往", gamma)
rho_2 = occupancy(max_step, episodes_2, "s4", "概率前往", gamma)
print("两种策略对于s4状态的占用度量分别为%f和%f" % (rho_1, rho_2))