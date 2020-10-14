import numpy as np


class pendulum:
    def __init__(self):
        # state参数范围设置
        self.theta_high = 2 * np.pi
        self.theta_low = 0
        self.theta_dot_high = 8.0
        self.theta_dot_low = -8.0

        # 参数预设省的下面有波浪线
        self.theta = 0
        self.theta_dot = 0
        self.u = None
        self.done = False

        # action范围设置
        self.u_high = 2.0
        self.u_low = -2.0

        # 仿真参数设置
        self.g = 10.0
        self.dt = 0.05  # 仿真步长
        self.m = 1  # 质量
        self.l = 1  # 长度

        # reward权重参数
        self.w1 = 1
        self.w2 = 0.1
        self.w3 = 0.001

        # 初始化状态
        self.reset()

    # 随机位置
    def reset(self):
        self.theta = np.random.uniform(self.theta_low, self.theta_high)
        self.theta_dot = np.random.uniform(self.theta_dot_low, self.theta_dot_high)
        self.u = None
        return np.array([self.theta, self.theta_dot])

    def reward(self, theta, theta_dot, u):
        R = -(self.w1 * (theta - np.pi / 2) ** 2 + self.w2 * theta_dot ** 2 + self.w3 * u ** 2)
        return R

    def step(self, u):
        # 初始化运算参数
        self.done = False
        # 获取k时间状态
        theta = self.theta
        theta_dot = self.theta_dot

        # 计算reward
        reward = self.reward(theta, theta_dot, u)
        new_theta_dot = theta_dot + (
                -(3 * self.g * np.cos(theta)) / (self.l * theta) + 3 / (self.m * self.l ** 2) * u) * self.dt

        new_theta_dot = np.clip(new_theta_dot, self.theta_dot_low, self.theta_dot_high)
        new_theta = theta + new_theta_dot * self.dt





        # 圆整角度

        while new_theta < 0:
            new_theta += 2 * np.pi

        while new_theta - 2 * np.pi > 0:
            new_theta -= 2 * np.pi

        # 状态空间参数
        A = np.array([
            [0, 1],
            [-(3 * self.g * np.cos(theta)) / (self.l * theta), 0]
        ])

        X_dot = np.array([theta_dot,-(3 * self.g * np.cos(theta)) / (self.l * theta) + 3 / (self.m * self.l ** 2) * u])

        self.theta = new_theta
        self.theta_dot = new_theta_dot

        # 判断当前状态是否结束
        f0 = self.m * self.g * np.cos(theta) / theta
        if np.abs(u - f0) < 1e-1:  # 控制精度为百分之九十
            # or theta_dot > self.theta_dot_high or theta_dot < self.theta_dot_low:
            self.done = True
        return np.array([self.theta, self.theta_dot]), reward, self.done, A, X_dot
