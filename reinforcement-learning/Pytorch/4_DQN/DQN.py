import torch
import torch.nn as nn # from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Module): #nn.Module: Base class for all neural network modules.
    def __init__(
            self,
            N_FEATURES,
            N_ACTIONS
    ):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(N_FEATURES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

        '''perhapes another way：
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N_FEATURES,50)
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS)
        )
        '''

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# DQN base version （Experience Replay + Fixed Q target）
class DQN(object):
    def __init__(
            self,
            N_FEATURES,
            N_ACTIONS,
            MEMORY_CAPACITY, 
            EPSILON, 
            EPSILON_INCR,
            EPSILON_MAX,
            ENV_A_SHAPE,
            TARGET_REPLACE_ITER, 
            BATCH_SIZE, 
            GAMMA,
            LR,
            device
    ):
        #pay attention not to use comma here, otherwise a tuple would be assigned
        self.N_FEATURES=N_FEATURES
        self.N_ACTIONS=N_ACTIONS
        self.MEMORY_CAPACITY=MEMORY_CAPACITY
        self.EPSILON=EPSILON
        self.EPSILON_INCR = EPSILON_INCR
        self.EPSILON_MAX = EPSILON_MAX
        self.ENV_A_SHAPE=ENV_A_SHAPE
        self.TARGET_REPLACE_ITER=TARGET_REPLACE_ITER
        self.BATCH_SIZE=BATCH_SIZE
        self.GAMMA=GAMMA
        self.LR=LR
        self.device=device

        
        ##self.eval_net = Net(self.N_FEATURES,self.N_ACTIONS).to(device)

        self.eval_net = Net(self.N_FEATURES,self.N_ACTIONS).to(device)
        self.target_net = Net(self.N_FEATURES,self.N_ACTIONS).to(device)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_FEATURES * 2 + 2))     # initialize memory

        #define loss and optimizer
        #intro to adma optimizer：https://www.jianshu.com/p/aebcaf8af76e， https://cloud.tencent.com/developer/article/1771256
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss() #常见的损失函数有MSE，MAE，Cross-Entropy etc

        self.cost=[]
        self.reward=[]

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0) #add one more dimension. Here: turn [a,b,c,d] to tensor([[a,b,c,d]])
        # input only one sample
        if np.random.uniform() < self.EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            # max(,1)返回的是每行最大值，以及该最大值在tuple中的indice
            # 取[1]后拿到了indice   不过这个indice默认放在一个list里
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def store_ep_reward(self,ep_r):
        if len(self.reward)==0:
            self.reward.append(round(ep_r,2))
        else:
            ep_r=self.reward[-1]*0.95+ep_r*0.05 #smooth the curve
            self.reward.append(round(ep_r,2)) 

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  #比tf方便多了
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE) 
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_FEATURES])
        b_a = torch.LongTensor(b_memory[:, self.N_FEATURES:self.N_FEATURES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_FEATURES+1:self.N_FEATURES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_FEATURES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 
        #b_s:batchsize*N_FEATURES, eval_net(b_s):batchsize*N_ACTIONS, gather(dim=1,index=b_a)也就是挑选每一行的那个action
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)   #（1，batch） reshape to (batch, 1) 
        # view 用于tensor纬度的重构，view(a,b)代表把tensor变成a*b的结构，view(-1)是变成一维，view(m,-1)代表变成m行，列数则自动匹配
        #.max(1) 是返回每一行最大的，返回的第一个参数是每行最大的拼出来的array，第二个参数是他们各自在原array中的indices
        loss = self.loss_func(q_eval, q_target)

        self.EPSILON = self.EPSILON + self.EPSILON_INCR if self.EPSILON < self.EPSILON_MAX else self.EPSILON_MAX

        ##
        self.optimizer.zero_grad() #每次反向传播前先把优化器中之前的梯度先清除掉
        loss.backward() #反向传播，这个过程中过挨个计算涉及到的梯度，并累加成最后的梯度x.grad
        self.optimizer.step() #对参数进行更新，例如SGD中x=x-lr*x.grad

        l=loss.detach().numpy()
        if len(self.cost)==0:
            
            self.cost.append(l)
        else:
            l=self.cost[-1]*0.95+l*0.05 #smooth the curve
            self.cost.append(l) 

    def plot(self,dirname):
        figure, axis = plt.subplots(1, 2)
        axis[0].plot(np.arange(len(self.cost)), self.cost , c='b')   
        axis[0].set_ylabel('Cost')
        axis[0].set_xlabel('Trainning steps')
        axis[0].grid() 
        axis[0].set_title("Cost curve")
        axis[1].plot(np.arange(len(self.reward)), self.reward , c='b')   
        axis[1].set_ylabel('Reward')
        axis[1].set_xlabel('Episode')
        axis[1].grid() 
        axis[1].set_title("Reward curve")
        plt.savefig(dirname+'/plot_CartPole-V0.png')
        plt.show()
    
    def get_epsilon(self):
        return self.EPSILON
    

    ##做PPO，SAC，TD3，TRPO，DDPG，完全体DQN或者完全体DDPG