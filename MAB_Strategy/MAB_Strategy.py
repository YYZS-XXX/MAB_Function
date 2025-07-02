import numpy as np
import matplotlib.pyplot as plt

"""
生成一个伯努利分布的类
"""
class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)
        self.best_id = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_id]
        self.K = K

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

"""
生成MAB的操作类Solver
"""
class solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.count = np.zeros(bandit.K)  # 记录拉下老虎机的次数
        self.regret = 0  # 记录当前步骤的懊悔值
        self.regrets = []  # 记录每一步的懊悔值
        self.actions = []  # 记录每一步所执行的动作

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplemented
    
    def run(self, num_step):
        for _ in range(num_step):
            k = self.run_one_step()
            self.count[k] += 1
            self.update_regret(k)
            self.actions.append(k)

"""
贪婪算法
"""
# class EpsilonGreedy(solver):
#     def __init__(self, bandit, init_prob = 1.0):
#         super().__init__(bandit)
#         self.estimate = np.array([init_prob] * bandit.K)
#         self.counts = 0

#     def run_one_step(self):
#         self.counts += 1
#         if np.random.random() > 1/self.counts:
#             k = np.argmax(self.estimate)
#         else:
#             k = np.random.randint(0, self.bandit.K)
#         r = self.bandit.step(k)
#         self.estimate[k] += 1./(self.count[k]+1)*(r - self.estimate[k])
#         return k

"""
上置信界算法
"""
# class UCB(solver):
#     def __init__(self, bandit, coef, init_prob=1.0):
#         super().__init__(bandit)
#         self.coef = coef
#         self.estimates = np.array([init_prob] * bandit.K)
#         self.total_count = 0

#     def run_one_step(self):
#         self.total_count += 1
#         self.ucb = self.estimates + self.coef*np.sqrt(np.log(self.total_count)/(2*(self.count+1)))
#         k = np.argmax(self.ucb)
#         r = self.bandit.step(k)
#         self.estimates[k] += 1. / (self.count[k] + 1) * (r - self.estimates[k])

#         return k

"""
汤普森采样法
"""
class Tompsonsampling(solver):
    def __init__(self, bandit):
        super().__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)

    def run_one_step(self):
        self.sample = np.random.beta(self._a, self._b)
        k = np.argmax(self.sample)
        r = self.bandit.step(k)

        self._a[k] = self._a[k] + r
        self._b[k] = self._b[k] + (1-r)
        return k

def plot_result(solvers, solver_name):
    for idx, solver in enumerate(solvers):
        # i = 1
        # print(idx, type(solver))
        time_list = range(len(solver.regrets))
        # print("时间跨度", time_list)
        plt.plot(time_list, solver.regrets, label = solver_name[idx])
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

# MAB的验证程序        
np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bandit_10_arm.best_id, bandit_10_arm.best_prob))

# 单个epsilon贪婪算法
np.random.seed(1)
epsilon_greedy = Tompsonsampling(bandit_10_arm)
epsilon_greedy.run(5000)
print("epsilon-贪婪算法的累积懊悔为：", epsilon_greedy.regret)
plot_result([epsilon_greedy], ["EpsilonGreedy"])

# # 多个epsilon贪婪算法
# np.random.seed(0)
# epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]  # 贪婪算法的Epsilon值
# # 根据Epsilon的值产生新的贪婪算法
# epsilon_greedy_list = [EpsilonGreedy(bandit_10_arm, Epsilon=e) for e in epsilons]
# # 绘图时对不同曲线标注的名称
# Epsilon_Name_List = ["Epsilon = {}".format(e) for e in epsilons]
# for solver in epsilon_greedy_list:
#     solver.run(5000)
# plot_result(epsilon_greedy_list, Epsilon_Name_List)