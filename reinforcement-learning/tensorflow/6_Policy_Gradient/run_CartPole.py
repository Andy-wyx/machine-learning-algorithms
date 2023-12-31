"""
Policy Gradient, Reinforcement Learning.

The cart pole example

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
import os

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering slows down learning speed, we default it to false and render after a while

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(3000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            #根据刚才这一次完整的游戏经历，估算Q
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: 
                print('Over Threshold!')
                RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn() #和之前算法不同，现在是回合更新：一回合只学一次

            if i_episode == 1000:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                print(os.path.dirname(__file__)) # get directory name
                print(os.path.abspath(__file__)) # get absolute path of current py file
                print(os.path.dirname(os.path.dirname(__file__))) # get path of directory of directory
                dirname=os.path.dirname(__file__)
                plt.savefig(dirname+'/CartPole_output.png')
                plt.show()
            break

        observation = observation_

