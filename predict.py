import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from Pendulum import pendulum

env = pendulum()
s_dims = 2
a_dims = 1
upper_bound = env.u_high
lower_bound = env.u_low

# 初始化权值
last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

inputs = layers.Input(shape=(s_dims,))
out = layers.Dense(512, activation="relu")(inputs)
out = layers.BatchNormalization()(out)
out = layers.Dense(512, activation="relu")(out)
out = layers.BatchNormalization()(out)
outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

outputs = outputs * upper_bound
model = tf.keras.Model(inputs, outputs)

check_point = "pendulum_online_actor.h5"
model.load_weights(check_point)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


def uniform_random():
    no = np.random.uniform(0.2*lower_bound, 0.2*upper_bound)
    #print(no)
    return no

def Gaussian():
    no = np.random.normal(0,0.1)
    return no


std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))


def action_policy(state):
    sample_actions = tf.squeeze(model(state))
    #no = ou_noise.__call__()
    no = Gaussian()
    #no = uniform_random()
    # no = 0
    # print(no)
    sample_actions = sample_actions.numpy() + no
    sample_actions = np.clip(sample_actions, lower_bound, upper_bound)

    return sample_actions


total_step = []
total_action = []
step = 0
prev_state = env.reset()
while step < 1000:
    tf_pre_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
    action = action_policy(tf_pre_state)
    total_action.append(action)
    state, reward, done = env.step(np.mean(action))
    total_step.append(state)
    prev_state = state
    step += 1

total_step = np.array(total_step)
total_action = np.array(total_action)

plt.subplot(221)
plt.plot(total_step[:, 0])
plt.xlabel("theta")
plt.subplot(222)
plt.plot(total_step[:, 1])
plt.xlabel("theta_dot")
plt.subplot(223)
plt.plot(total_action)
plt.xlabel("action")
plt.show()
