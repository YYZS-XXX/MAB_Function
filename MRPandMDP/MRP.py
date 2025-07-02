import numpy as np
np.random.seed(0)
# 定义状态转移矩阵
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]  # 定义奖励
gamma = 0.5  # 定义衰减因子

# 计算回报Rt
def Compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = G * gamma + rewards[chain[i]-1]
    return G

# 计算价值Value
def compute(rewards, gamma, P, state_num):
    rewards = np.array(rewards).reshape(-1, 1)
    Value = np.dot(np.linalg.inv(np.eye(state_num, state_num) - gamma * P), 
                   rewards)
    return Value

# 一个状态序列,s1-s2-s3-s6
chain = [1, 2, 3, 6]
start_index = 0
# 计算回报
G = Compute_return(start_index, chain, gamma)
print("本序列得到的回报是：%s。" % G)
V = compute(rewards, gamma, P, 6)
print("MRP中每个状态的价值为：%s" % V)