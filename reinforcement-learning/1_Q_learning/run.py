import pandas as pd
import numpy as np

df=pd.read_csv('q_table_output.csv',index_col=0)
#df=pd.read_csv('q_table_output.csv') 默认读取时会自动设置新的index，但是这里我们需要的是将第一列(state)设置为index

"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
"""

from maze_env import Maze
from RL_brain import QLearningTable
import time

epoches = 50


def update():
    count_success = 0
    count_steps = 0
    for episode in range(epoches):
        start = time.time()
        # initial observation
        observation = env.reset()

        # step counter
        steps = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            print(1)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            print(1)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            print(1)

            # swap observation
            observation = observation_

            steps += 1

            print(1)

            # break while loop when end of this episode
            if done:
                end = time.time()
                if reward == 1:
                    count_success += 1
                count_steps += steps
                print('Episode ' + str(episode) + ', total steps: ' + str(steps) + ', Time Consumed: ' + str(
                    round(end - start)) + 's, reward: ', end='')
                print('1') if reward == 1 else print('-1')
                break

    # end of game
    print('game over, AvgSteps= ' + str(round(count_steps / epoches)) + ', success rate= ' + str(
        "%.2f%%" % (count_success / epoches * 100)))
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    # RL = QLearningTable(actions=list(range(env.n_actions)))
    RL = QLearningTable(actions=list(range(env.n_actions)), trained_q_table=df)

    env.after(epoches, update)
    env.mainloop()
    print('\r\nQ-table:\n')
    print(RL.q_table)