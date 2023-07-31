import gym
import torch
from DQN import DQN
import matplotlib.pyplot as plt
import os, time
import numpy as np


######################################## Hyper Parameters ###########################################
MAX_EPISODES=400
MAX_EP_STEPS=3000
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0                 # greedy policy (eploration vs exploitation)
EPSILON_INCR=0.001
EPSILON_MAX=0.9
GAMMA = 0.9                 # reward discount rate
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_FEATURES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

######################################################################################################
# Get cpu, gpu or mps device for training.
'''
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
'''
device="cpu"

########################################## run CartPole-V0 ###########################################
def run_CartPoleV0():
    print('\nCollecting experience...')
    t1 = time.time()
    for i_episode in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0
        ep_step_counter=0
        while True:
            env.render()
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            # some reward engineering
            # there are a x_threshold and a theta_threshold for gameover
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            ep_step_counter +=1
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done or ep_step_counter>MAX_EP_STEPS:
                    print('Ep: ', str(i_episode).rjust(4),
                        '| Ep_r: ', str(round(ep_r, 2)).rjust(7),
                        '| Steps: ', str(ep_step_counter).rjust(4),
                        '| Epsilon: ', round(dqn.get_epsilon(),4))

            if done or ep_step_counter>MAX_EP_STEPS:
                dqn.store_ep_reward(ep_r)
                break
            s = s_

    print('Running time: ', round(time.time() - t1,2),'s')

    dirname=os.path.dirname(__file__)
    dqn.plot(dirname)

if __name__ == '__main__':
	dqn = DQN(N_FEATURES, N_ACTIONS, MEMORY_CAPACITY,
            EPSILON, EPSILON_INCR, EPSILON_MAX,
            ENV_A_SHAPE, TARGET_REPLACE_ITER, BATCH_SIZE, 
            GAMMA, LR, device)
	run_CartPoleV0()

