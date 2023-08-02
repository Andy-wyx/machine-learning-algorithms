from collections import namedtuple
from torch.distributions import Categorical
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import matplotlib.pyplot as plt

########################################## hyperparameters ################################################

MAX_EPISODE = 3000
MAX_EP_STEPS = 3000   # maximum time steps in one episode
RENDER = True   
GAMMA = 0.9   # reward discount in TD error
BETA=0.05     # Hyperparameter that controls the influence of entropy loss
#LR_A = 0.005   # learning rate for actor
#LR_C = 0.01   # learning rate for critic
LR = 0.003
PUNISHMENT=10 

###########################################   parameters   ###################################################

env = gym.make('CartPole-v0')
env = env.unwrapped

#np.random.seed(1)
#torch.manual_seed(1)
#env.seed(1)   # reproducible

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

######################################## Native Actor-Critic Model ###########################################
'''
A natural AC implementation with the following loss standard:
log[Pi*(V)] as loss for updating policy network
Td error of V as loss for updating value network. i.e. r+V'-V 
(there are many other loss options for implem Policy Net)
Implement Actor and Critic into one single class makes the code simpler,  but may have to use the same learning rate 
adding two losses together makes retain_graph=True unnecessary thus save our effort from fixing inplace operation as well. 
'''
class ActorCriticNet(nn.Module):
	def __init__(self, n_features, n_actions, n_hidden=32):
		super(ActorCriticNet, self).__init__()
		'''
		an example of Convlayers
		self.net 1 = nn.Sequential(
		nn.Conv2d(n_channels, 32, 8, 4),
		nn.ReLU(),
		nn.Conv2d(32, 64, 4, 2),
		nn.ReLU(),
		nn.Conv2d(64, 32, 3, 1),
		nn.ReLU(),
		)
		'''
		self.fc1 = nn.Linear(n_features, n_hidden)
		self.actor = nn.Linear(n_hidden,n_actions)
		self.critic = nn.Linear(n_hidden, 1)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)
		
		self.policy_cost=[]
		self.reward=[]
		self.value_cost=[]
		self.total_cost=[]

	def forward(self, x):
		x = self.fc1(x)
		x = F.relu(x)
		acts_prob = self.actor(x)
		policy = F.softmax(acts_prob,dim=-1) 
		#m*n*k,dim=-1 or 2 means numbers on the third dimension sums to 1. (same row)
		#dim=0, numbers on the same position across the first dimension sums to 1, dim=1, the 2nd dim sums to one (same column)
		value = self.critic(x)
		return policy, value
	
	def choose_action(self,state):
		'''example of np.random.choice
		#np.random.choice(5, 3) Generate a uniform random sample from np.arange(5) of size 3
		#np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0]) Generate a non-uniform random sample from np.arange(5) of size 3:
		choice=np.random.choice(np.arange(probs.shape[1]), p=probs.data.numpy().ravel())
		'''
		#here is another approach provided by pytorch:https://pytorch.org/docs/stable/distributions.html

		#increase one dimension cuz actor needs input with shape (,n_features); another way is to use state = torch.Tensor(state[np.newaxis, :]) 
		state = torch.from_numpy(state).float()
		probs, state_value= self(state)
		m=Categorical(probs)
		action=m.sample()
		
		return int(action.item()) ## the returned value is indeed an index of action, but hold by a tensor e.g. tensor(3) 
	
	def learn(self,s,a,r,s_):
		s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
		acts_prob,v = self(s) 
		_,v_ = self(s_)
		#critic network only take s as input, so update criterion is based on td error of state value, not advantage.
		td_error = r + GAMMA * v_ - v   # gradient = grad[r + gamma * V(s_) - V(s)]
		value_loss = td_error ** 2

		log_prob = torch.log(acts_prob[0, a])
		exp_v = torch.mean(log_prob * v)  #true_gradient = grad[logPi(s, a) * td_error]
		policy_loss = -exp_v

		entropy_loss = -BETA * sum((acts_prob * torch.log(acts_prob))[0])

		#When you call loss.backward(), all it does is compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True 
		# and store them in parameter.grad attribute for every parameter.
		# optimizer.step() updates all the parameters based on parameter.grad
		#when pdf is not spreaded out, p*logP is close to 0, means go ahead to update strongly when you are confident enough about choosing action, 
		#when the prob of each action is close to each other, entropy_loss is negative and have a large abs value, means don't step too far, update cautiously
		# the introduction of entropy helps with avoiding stucking into local minima  and helps encourage exploration.
		loss=value_loss+policy_loss+entropy_loss 
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		policy_loss=float(policy_loss.detach().numpy()) # cannot numpy on tensor that require grad
		value_loss=float(value_loss.detach().numpy())
		total_loss=float(loss.detach().numpy())
		self.policy_cost.append(policy_loss)
		self.value_cost.append(value_loss)
		self.total_cost.append(total_loss)

	def plot(self,dirname):
		figure, axis = plt.subplots(2, 2)
		axis[0,0].plot(np.arange(len(self.value_cost)), self.value_cost , c='b' , label='original')   
		axis[0,0].plot(np.arange(len(self.value_cost)), get_smoothed(self.value_cost), color='red', label='average of ten') 
		axis[0,0].legend(loc='best') 
		axis[0,0].set_ylabel('Td error of V in Value Network')
		axis[0,0].set_xlabel('Trainning steps')
		axis[0,0].grid() 
		axis[0,0].set_title("Value loss")
		axis[0,1].plot(np.arange(len(self.policy_cost)), self.policy_cost , c='b' , label='original')   
		axis[0,1].plot(np.arange(len(self.policy_cost)), get_smoothed(self.policy_cost), color='red', label='average of ten') 
		axis[0,1].legend(loc='best') 
		axis[0,1].set_ylabel('logPi*V in Policy Network')
		axis[0,1].set_xlabel('Trainning steps')
		axis[0,1].grid() 
		axis[0,1].set_title("Policy loss")
		axis[1,0].plot(np.arange(len(self.reward)), self.reward , c='b', label='original')   
		axis[1,0].plot(np.arange(len(self.reward)), get_smoothed(self.reward), color='red', label='average of ten') 
		axis[1,0].legend(loc='best') 
		axis[1,0].set_ylabel('Reward')
		axis[1,0].set_xlabel('Episode')
		axis[1,0].grid() 
		axis[1,0].set_title("Reward Curve")
		axis[1,1].plot(np.arange(len(self.total_cost)), self.total_cost , c='b', label='original')   
		axis[1,1].plot(np.arange(len(self.total_cost)), get_smoothed(self.total_cost), color='red', label='average of ten') 
		axis[1,1].legend(loc='best') 
		axis[1,1].set_ylabel('policy_loss+value_loss+entropy_loss')
		axis[1,1].set_xlabel('Trainning steps')
		axis[1,1].grid() 
		axis[1,1].set_title("Total Cost")
		figure.tight_layout()
		plt.savefig(dirname+'/AC_v1_CartPole-V0.png')
		plt.show()


def get_smoothed(original):
	smoothed=[sum(original[e-9:e+1])/10 if e>9 else sum(original[:e+1])/(e+1) for e in range(len(original))]
	return smoothed

############################################# run CartPole-V0 ##############################################

def run_CartPoleV0(model):
	t1=time.time()
	for i_episode in range(MAX_EPISODE):
		s = env.reset()
		t = 0
		ep_r = 0

		while True:
			if RENDER: env.render()

			a = model.choose_action(s)

			s_, r, done, info = env.step(a)

			x, x_dot, theta, theta_dot = s_
			r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8 
			r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
			r = r1 + r2

			if done: r = -PUNISHMENT   

			ep_r+=r
		
			model.learn(s,a,r,s_)
			#https://github.com/NVlabs/FUNIT/issues/23

			s = s_
			t += 1

			if done or t>=MAX_EP_STEPS:
				model.reward.append(ep_r)

				print('Ep: ', str(i_episode).rjust(4),
						'| Ep_r: ', str(round(ep_r, 2)).rjust(7),
						'| Steps: ', str(t).rjust(4))
				break

	t=round(time.time()-t1,2)
	print('Total running time: ',t,'s','Avg Time Per Ep: ', t/MAX_EPISODE,'s')
	dirname=os.path.dirname(__file__)
	model.plot(dirname)


if __name__=='__main__':
	AC=ActorCriticNet(n_features=N_F, n_actions=N_A)
	dir_path=os.path.dirname(__file__)
	model_path=dir_path+'/AC.pth'
	if os.path.exists(model_path):
		AC.load_state_dict(torch.load(model_path))
		print("AC.pth Loaded.")

	run_CartPoleV0(model=AC)
	#torch.save(AC.state_dict(), model_path)
	#print("Saved PyTorch Natural AC Model State to AC.pth")