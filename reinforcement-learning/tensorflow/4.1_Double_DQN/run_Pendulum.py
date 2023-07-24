"""
Double DQN & Natural DQN comparison,
The Pendulum example.

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


env = gym.make('Pendulum-v0')
env = env.unwrapped
print(env.action_space)
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11 #!

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        if total_steps - MEMORY_SIZE > 4000: env.render()      #8000

        action = RL.choose_action(observation)
        #print('observation: '+str(observation)) observation是三行数组，1:x:cos(theta) 2.y:sin(theta) 3:angle velocity [-8,8]
        #print('action: '+str(action))     #action index 0～10

        #该环境的动作是[-2,2]连续的,DQN需要离散的actionspace,等分成11个动作
        #if this the best way to deal with continous space? 有其他处理方法吗？

        #convert to [-2 ~ 2] float actions
        #Refer to offical description https://gymnasium.farama.org/environments/classic_control/pendulum/ and check the pendulum.py file
        #For simplicity,this double DQN still wants to use a discrete action-space, so we define action_space=11 so actions are indexed from 0 to 10
        #and we build l2 in NN based on action_space==11, so in the choose_action(obs), an index (0~10) is returned
        # however, it is still better to store and train the network with the orginial scale: torque range [-2,2]
        # so here we use the following line to turn discrete space [0,1,2,...,9,10] into [-2,2].i.e.[-2,-1.6,-1.2,-0.8,-0.4,...,1.6,2] 
        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   
        #print('f_action: '+str(f_action))
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. 
        # Referring to the final result, we see that natural DQN overestimate Q, while double DQN doesn't.

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:   # learning
            RL.learn()

        if total_steps - MEMORY_SIZE > 10000:   # stop game 20000
            break

        observation = observation_
        total_steps += 1
    return RL.q

q_natural = train(natural_DQN)
q_double = train(double_DQN)

plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
dirname=os.path.dirname(__file__)
plt.savefig(dirname+'/output.png')
plt.show()
