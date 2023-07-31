import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)

np.random.seed(1)
torch.manual_seed(1)

########################################## hyperparameters ################################################
MAX_EPISODE = 150
MAX_EP_STEPS = 3000   # maximum time steps in one episode
RENDER = True   
GAMMA = 0.9   # reward discount in TD error
LR_A = 0.005   # learning rate for actor
LR_C = 0.01   # learning rete for critic


env = gym.make('CartPole-v0')
env.seed(1)   # reproducible
env = env.unwrapped


N_F = env.observation_space.shape[0]
N_A = env.action_space.n

######################################## Native Actor-Critic Model ###########################################
'''
use loss: sigma[Pi*(r+V'-V)] for updating policy network
use loss: td error of V for updating value network. i.e. r+V'-V
'''
class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output, activate=False):
		super(Net, self).__init__()
		self.l1 = nn.Linear(n_feature, n_hidden)
		self.acts_prob = nn.Linear(n_hidden, n_output)
		self.activate=activate


	def forward(self, x):
		x = self.l1(x)
		x = F.relu(x)
		x = self.acts_prob(x)
		if self.activate:
			x = F.softmax(x)
		return x


class Actor(object):
	def __init__(self, n_features, n_actions, n_hidden=20, lr=0.001):
		self.n_features = n_features
		self.n_actions = n_actions
		self.n_hidden = n_hidden
		self.lr = lr

		self._build_net()


	def _build_net(self):
		self.actor_net = Net(self.n_features, self.n_hidden, self.n_actions, activate=True)  #use softmax to deal with discrete actions, perhapes as a eploration substitue for greedy strategy
		self.optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr)


	def choose_action(self, s):
		s = torch.Tensor(s[np.newaxis, :])
		probs = self.actor_net(s)
		return np.random.choice(np.arange(probs.shape[1]), p=probs.data.numpy().ravel())


	def learn(self, s, a, td):
		s = torch.Tensor(s[np.newaxis, :])
		acts_prob = self.actor_net(s)
		log_prob = torch.log(acts_prob[0, a])
		exp_v = torch.mean(log_prob * td)

		loss = -exp_v
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return exp_v


class Critic(object):
	def __init__(self, n_features, lr=0.01):
		self.n_features = n_features
		self.lr = lr

		self._build_net()

		self.cost=[]
		self.reward=[]


	def _build_net(self):
		self.critic_net = Net(self.n_features, 20, 1)
		self.optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)


	def learn(self, s, r, s_):
		s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
		v, v_ = self.critic_net(s), self.critic_net(s_)
		#critic network only take s as input, so update criterion is based on td error of state value, not advantage.
		td_error = r + GAMMA * v_ - v 
		loss = td_error ** 2

		self.optimizer.zero_grad()
		loss.backward(retain_graph=True) # in our updating process, critic network is updated before actor
		#Both losses, critic_loss and actor_loss use the advantage tensor in their computation.
		#The first critic_loss.backward() call will free the intermediate forward activations stored during the previous forward pass, which will cause actor_loss.backward() to fail since both backward passes depend on the computation graph (and the intermediate activations) attached to advantage.
		#To solve the issue you could use actor_loss.backward(retain_graph=True) or, if it fits your use case, sum both losses together before calling .backward() on the sum.
		self.optimizer.step()

		return td_error
	
	def store_ep_reward(self,ep_r):
		if len(self.reward)==0:
			self.reward.append(ep_r)
		else:
			smoothed_reward=self.reward[-1]*0.95+ep_r*0.05
			self.reward.append(smoothed_reward)

	def store_ep_cost(self,l):
		if len(self.cost)==0:
			self.cost.append(l)
		else:
			l=self.cost[-1]*0.95+l*0.05 #smooth the curve
			self.cost.append(l) 
	
	def plot(self,dirname):
		figure, axis = plt.subplots(1, 2)
		axis[0].plot(np.arange(len(self.cost)), self.cost , c='b')   
		axis[0].set_ylabel('Critic_Cost')
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



############################################# run CartPole-V0 ##############################################

def run_CartPoleV0():
	t1=time.time()
	for i_episode in range(MAX_EPISODE):
		s = env.reset()
		t = 0
		ep_r = 0

		while True:
			if RENDER: env.render()

			a = actor.choose_action(s)

			s_, r, done, info = env.step(a)

			if done: r = -20   ##

			ep_r+=r

			td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
			actor.learn(s, a, td_error)   # true_gradient = grad[logPi(s, a) * td_error]

			#l=td_error.detach().numpy() ** 2
			#critic.store_ep_cost(l ** 2)

			#https://github.com/NVlabs/FUNIT/issues/23

			s = s_
			t += 1

			if done or t>=MAX_EP_STEPS:
				critic.store_ep_reward(ep_r)

				print('Ep: ', str(i_episode).rjust(4),
                        '| Ep_r: ', str(round(ep_r, 2)).rjust(7),
                        '| Steps: ', str(t).rjust(4))
				break

	print('Total running time: ',round(time.time()-t1,2),'s')
	dirname=os.path.dirname(__file__)
	#critic.plot(dirname)


if __name__=='__main__':
	actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
	critic = Critic(n_features=N_F, lr=LR_C)   
	run_CartPoleV0()



