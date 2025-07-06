import gym

env = gym.make("FrozenLake-v1")  # 创建环境
env = env.unwrapped  # 解封访问
env.render()

holes = set()
ends = set()
for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])

holes = holes - ends

print("冰洞的索引为:", holes)
print("终点的索引为:", ends)

for a in env.P[14]:
    print(env.P[14][a])