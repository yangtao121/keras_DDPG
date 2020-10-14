from Pendulum import pendulum
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_DDPG import DDPG

env = pendulum()

s_dims = 2
a_dims = 1
upper_bound = env.u_high
lower_bound = env.u_low

total_episodes = 100

RL = DDPG(upper_bound, lower_bound, s_dims, a_dims)

ep_reward_list = []
avg_reward_list = []

for ep in range(total_episodes):
    prev_state = env.reset()
    episodic_reward = 0
    step = 0

    while True:
        tf_pre_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = RL.action_policy(tf_pre_state)
        # print(action)
        state, reward, done = env.step(np.mean(action))
        # print(state)

        RL.record(prev_state, action, reward, state)
        episodic_reward += reward

        RL.learn()
        RL.soft_update()

        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-40:])
    # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

RL.save_model()

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()
