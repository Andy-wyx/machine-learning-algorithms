"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain import DeepQNetwork

#创建cartpole这个环境
env = gym.make('CartPole-v0') 
env = env.unwrapped

#打印一些关于该环境的信息
print(env.action_space) #向左和向右两个动作
print(env.observation_space) 
print(env.observation_space.high)  #observation's upper limit
print(env.observation_space.low)   #obs's lower limit

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0


for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        #由于第四哥output没什么用，可以忽略掉 observation_, reward, done, _ = env.step(action)

        #该环境中的机制为：1.杆子偏离度数达到一定值，2.物体移动到了边界    
        # 以上任一条件发生时game over，game over前reward一直为1，gameover时无reward
        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8     #位置偏离20%以内为正奖励，[-0.8,0.2] 貌似还有可能低于-0.8吧
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5  #角度偏离max radian的50%以内为正奖励，上限0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000: #1000步以后，每步都learn
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 4))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost('CartPole_cost')
