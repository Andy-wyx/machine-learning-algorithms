"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import os
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 10000

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
    )

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    accumulated_steps = []
    episodes = []
    steps= []
    learn_counters=[]
    for i_episode in range(20):
        observation = env.reset()
        epi_steps=0
        learn_counter=0
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done: reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()
                learn_counter+=1

            if done:
                print('episode ', i_episode, ' finished')
                accumulated_steps.append(total_steps)
                episodes.append(i_episode)
                steps.append(epi_steps)
                learn_counters.append(learn_counter)
                break

            observation = observation_
            total_steps += 1
            epi_steps += 1
    return np.vstack((episodes, accumulated_steps,steps,learn_counters))     #episode:[0,1,2,...,19]

print('----------------------trainning natural DQN----------------------')
his_natural = train(RL_natural)
print('\n----------------------trainning Prioritized Replay DQN----------------------')
his_prio = train(RL_prio)




# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(1, 3)

# subplot1
# compare based on first success 这是两个模型 accumulated trainning steps 随episode变化的对比折线图
# 下面去掉episode0的trainning steps是因为主要比较的是从开始学习之后，曲线趋于横线的快慢，episode0跑的steps数量是由随机种子决定的，起点的高低并没有什么意思，所以各自减去第一次的steps是是有一定合理性的比较策略
#plt.plot(his_natural[0, :], his_natural[1, :]- his_natural[1, 0] , c='b', label='natural DQN') #x-axis=episodes，y-axix=accumulated-steps color=blue
#plt.plot(his_prio[0, :], his_prio[1, :]- his_prio[1, 0] , c='r', label='DQN with prioritized replay') # color=red
axis[0].plot(his_natural[0, :], his_natural[1, :] , c='b', label='natural DQN')   
axis[0].plot(his_prio[0, :], his_prio[1, :] , c='r', label='DQN with prioritized replay') 
axis[0].legend(loc='best') #设置图例
axis[0].set_ylabel('total training time')
axis[0].set_xlabel('episode')
axis[0].grid() #设置网格线
axis[0].set_title("Comparison1")

# subplot2
# 类似图一，画一个每个episode中的steps折线变化图(non-accumlated)，
# 并且可以适当提高训练epoches数，看看大概第几轮时两个模型都基本达到了最优，
axis[1].plot(his_natural[0, :], his_natural[2, :] , c='b', label='natural DQN')   
axis[1].plot(his_prio[0, :], his_prio[2, :], c='r', label='DQN with prioritized replay') 
axis[1].legend(loc='best') #设置图例
axis[1].set_ylabel('training time of each episode')
axis[1].set_xlabel('episode')
axis[1].grid() #设置网格线
axis[1].set_title("Comparison2")


# subplot3
#感觉这个例子不好，明显是prioritized replay DQN在第一步是走了更多的步数(3w多步，而normalDQN只有1w+)这样在episode0时，prio网络learn了更多次
#无法证明曲线斜率低时算法更优导致的，
axis[2].plot(his_natural[0, :], his_natural[3, :] , 'o', color='blue', label='natural DQN')   
axis[2].plot(his_prio[0, :], his_prio[3, :],'o', color='red', label='DQN with prioritized replay') 
axis[2].legend(loc='best') #设置图例
axis[2].set_ylabel('learning counters')
axis[2].set_xlabel('episode')
axis[2].grid() #设置网格线
axis[2].set_title("Comparison3")

# Combine all the operations and display
dirname=os.path.dirname(__file__)
plt.savefig(dirname+'/output.png')
plt.show()   

'''
# compare based on first success 这是两个模型 accumulated trainning steps 随episode变化的对比折线图
# 下面去掉episode0的trainning steps是因为主要比较的是从开始学习之后，曲线趋于横线的快慢，episode0跑的steps数量是由随机种子决定的，起点的高低并没有什么意思，所以各自减去第一次的steps是是有一定合理性的比较策略
#plt.plot(his_natural[0, :], his_natural[1, :]- his_natural[1, 0] , c='b', label='natural DQN') #x-axis=episodes，y-axix=accumulated-steps color=blue
#plt.plot(his_prio[0, :], his_prio[1, :]- his_prio[1, 0] , c='r', label='DQN with prioritized replay') # color=red

plt.figure(0)
plt.plot(his_natural[0, :], his_natural[1, :] , c='b', label='natural DQN') #x-axis=episodes，y-axix=accumulated-steps color=blue
plt.plot(his_prio[0, :], his_prio[1, :] , c='r', label='DQN with prioritized replay') # color=red
plt.legend(loc='best') #设置图例
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid() #设置网格线
dirname=os.path.dirname(__file__)
plt.savefig(dirname+'/accumulated_trainning_time.png')


# 类似图一，画一个每个episode中的steps折线变化图(non-accumlated)，
# 并且可以适当提高训练epoches数，看看大概第几轮时两个模型都基本达到了最优，

plt.figure(1)
plt.plot(his_natural[0, :], his_natural[2, :] , c='b', label='natural DQN') #x-axis=episodes，y-axix=steps color=blue
plt.plot(his_prio[0, :], his_prio[2, :], c='r', label='DQN with prioritized replay') # color=red
plt.legend(loc='best') #设置图例
plt.ylabel('training time of each episode')
plt.xlabel('episode')
plt.grid() #设置网格线
dirname=os.path.dirname(__file__)
plt.savefig(dirname+'/trainning_time_each_episode.png')


#感觉这个例子不好，明显是prioritized replay DQN在第一步是走了更多的步数(3w多步，而normalDQN只有1w+)这样在episode0时，prio网络learn了更多次
#无法证明曲线斜率低时算法更优导致的，

plt.figure(2)
plt.plot(his_natural[0, :], his_natural[3, :] , 'o', color='blue', label='natural DQN') 
plt.plot(his_prio[0, :], his_prio[3, :],'o', color='red', label='DQN with prioritized replay') 
plt.legend(loc='best') #设置图例
plt.ylabel('learning counters')
plt.xlabel('episode')
plt.grid() #设置网格线
dirname=os.path.dirname(__file__)
plt.savefig(dirname+'/learning_counters.png')
plt.show()   
'''
