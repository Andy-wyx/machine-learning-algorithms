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

MAX_EPISODE = 150
MAX_EP_STEPS = 3000   # maximum time steps in one episode
RENDER = True   
GAMMA = 0.9   # reward discount in TD error
#LR_A = 0.005   # learning rate for actor
#LR_C = 0.01   # learning rate for critic
LR = 0.01

env = gym.make('CartPole-v0')
env = env.unwrapped

np.random.seed(1)
torch.manual_seed(1)
env.seed(1)   # reproducible

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

######################################## Native Actor-Critic Model ###########################################
'''
A natural AC implementation with the following loss standard:
log[Pi*(r+V'-V)] as loss for updating policy network
Td error of V as loss for updating value network. i.e. r+V'-V (there are many other loss options for implem Value Net)
Implement Actor and Critic into one single class makes the code simpler,  but may have to use the same learning rate
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
		self.save_actions=[]

	def forward(self, x):
		#replace x=self.f1(x)to fix inplace operation error
		x = F.linear(x,self.fc1.weight.clone(),self.fc1.bias)
		x = F.relu(x)
		acts_prob = F.linear(x,self.actor.weight.clone(),self.actor.bias)
		policy = F.softmax(acts_prob,dim=-1) 
		#m*n*k,dim=-1 or 2 means numbers on the third dimension sums to 1. (same row)
		#dim=0, numbers on the same position across the first dimension sums to 1, dim=1, the 2nd dim sums to one (same column)
		value = F.linear(x,self.critic.weight.clone(),self.critic.bias)
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
		self.save_actions.append(SavedAction(m.log_prob(action),state_value))
		
		return int(action.item()) ## the returned value is indeed an index of action, but hold by a tensor e.g. tensor(3) 
	
	def learn(self,s,a,r,s_):
		s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
		acts_prob,v = self(s) 
		_,v_ = self(s_)
		#critic network only take s as input, so update criterion is based on td error of state value, not advantage.
		td_error = r + GAMMA * v_ - v   # gradient = grad[r + gamma * V(s_) - V(s)]
		loss = td_error ** 2

		#critic update
		self.optimizer.zero_grad()
		loss.backward(retain_graph=True) 
		'''in our updating process, critic network is updated before actor
		Both losses, critic_loss and actor_loss use the 'advantage tensor' in their computation.
		The first critic_loss.backward() call will free the intermediate forward activations stored during the previous forward pass, 
		which will cause actor_loss.backward() to fail since both backward passes depend on the computation graph (and the intermediate activations) attached to advantage.
		To solve the issue you could use actor_loss.backward(retain_graph=True) or, if it fits your use case, sum both losses together before calling .backward() on the sum.
		However, retain the graph leads to new trouble of inplace operation for opimizer.step(). When retain_graph=fasle by default, the opimizer.step() uses inplace operation it'ok
		cuz it will create new computation graph anyway, but when retain_graph=true, inplace operation of those parameters could be a problem. 
		As analyzed and suggested by KFrank on this post https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-torch-floattensor-64-1-which-is-output-0-of-asstridedbackward0-is-at-version-3-expected-version-2-instead-hint-the-backtrace-further-a/171826/7
		a solution is to clone related non-activation layers.
		'''
		self.optimizer.step()
		
		#actor update
		log_prob = torch.log(acts_prob[0, a])
		exp_v = torch.mean(log_prob * td_error)  #true_gradient = grad[logPi(s, a) * td_error]
		loss = -exp_v
		#When you call loss.backward(), all it does is compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True 
		# and store them in parameter.grad attribute for every parameter.
		# optimizer.step() updates all the parameters based on parameter.grad
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		policy_cost=float(exp_v.detach().numpy()) # cannot numpy on tensor that require grad
		value_cost=float(td_error.detach().numpy())
		self.policy_cost.append(policy_cost)
		self.value_cost.append(value_cost)

	def plot(self,dirname,t):
		figure, axis = plt.subplots(2, 2)
		axis[0,0].plot(np.arange(len(self.value_cost)), self.value_cost , c='b' , label='original')   
		axis[0,0].plot(np.arange(len(self.value_cost)), get_smoothed(self.value_cost), color='red', label='average of ten') 
		axis[0,0].legend(loc='best') 
		axis[0,0].set_ylabel('Td error of V in Value Network')
		axis[0,0].set_xlabel('Trainning steps')
		axis[0,0].grid() 
		axis[0,0].set_title("Cost curve")
		axis[0,1].plot(np.arange(len(self.policy_cost)), self.policy_cost , c='b' , label='original')   
		axis[0,1].plot(np.arange(len(self.policy_cost)), get_smoothed(self.policy_cost), color='red', label='average of ten') 
		axis[0,1].legend(loc='best') 
		axis[0,1].set_ylabel('logPi*Td(V) in Policy Network')
		axis[0,1].set_xlabel('Trainning steps')
		axis[0,1].grid() 
		axis[0,1].set_title("Cost curve")
		axis[1,0].plot(np.arange(len(self.reward)), self.reward , c='b', label='original')   
		axis[1,0].plot(np.arange(len(self.reward)), get_smoothed(self.reward), color='red', label='average of ten') 
		axis[1,0].legend(loc='best') 
		axis[1,0].set_ylabel('Reward')
		axis[1,0].set_xlabel('Episode')
		axis[1,0].grid() 
		axis[1,0].set_title("Reward Curve")
		axis[1,1].plot(np.arange(len(self.reward)), [t/len(self.reward) for i in range(len(self.reward))] , c='b')   
		axis[1,1].set_ylabel('Average Time')
		axis[1,1].set_xlabel('Episode')
		axis[1,1].grid() 
		axis[1,1].set_title("Average Time Per Ep")
		figure.tight_layout()
		plt.savefig(dirname+'/AC_CartPole-V0.png')
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

			if done: r = -20   ##

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
	print('Total running time: ',t,'s')
	dirname=os.path.dirname(__file__)
	model.plot(dirname, t)


if __name__=='__main__':
	AC=ActorCriticNet(n_features=N_F, n_actions=N_A)
	run_CartPoleV0(model=AC)
	torch.save(AC.state_dict(), "model.pth")
	print("Saved PyTorch Model State to model.pth")
	# AC= ActorCriticNet(n_features=N_F, n_actions=N_A, lr=LR_A)
	#AC.load_state_dict(torch.load("model.pth"))

	##和另一版本再对比一下取其精华，然后去做continous的，然后A2C,DDPG,A3C,CNN,PPO,EA,TD3,LSTM,写实验，