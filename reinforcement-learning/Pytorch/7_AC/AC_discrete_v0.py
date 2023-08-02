import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import matplotlib.pyplot as plt

########################################## hyperparameters ################################################
MAX_EPISODE = 500
MAX_EP_STEPS = 3000   # maximum time steps in one episode
RENDER = True   
GAMMA = 0.9   # reward discount in TD error
LR_A = 0.005   # learning rate for actor
LR_C = 0.01   # learning rete for critic

###########################################  parameters  #################################################
np.random.seed(1)
torch.manual_seed(1)
env = gym.make('CartPole-v0')
env.seed(1)   # reproducible
#torch.manual_seed(1)
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
		#x = self.l1(x)
		x=F.linear(x,self.l1.weight.clone(),self.l1.bias)
		x = F.relu(x)
		#x = self.acts_prob(x)
		x=F.linear(x,self.acts_prob.weight.clone(),self.acts_prob.bias)
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
		#When you call loss.backward(), all it does is compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True 
		# and store them in parameter.grad attribute for every parameter.
		# optimizer.step() updates all the parameters based on parameter.grad
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
		self.actor_cost=[]


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

		return td_error
	
	def store_ep_reward(self,ep_r):
			self.reward.append(ep_r)

	def store_ep_cost(self,l):
			self.cost.append(l) 

	def store_ep_actor_cost(self,l):
		self.actor_cost.append(l)
	
	def get_smoothed(self, original):
		smoothed=[sum(original[e-9:e+1])/10 if e>9 else sum(original[:e+1])/(e+1) for e in range(len(original))]
		return smoothed
	
	def plot(self,dirname,t):
		figure, axis = plt.subplots(2, 2)
		axis[0,0].plot(np.arange(len(self.cost)), self.cost , c='b' , label='original')   
		axis[0,0].plot(np.arange(len(self.cost)), self.get_smoothed(self.cost), color='red', label='average of ten') 
		axis[0,0].legend(loc='best') 
		axis[0,0].set_ylabel('Td error of V in Value Network')
		axis[0,0].set_xlabel('Trainning steps')
		axis[0,0].grid() 
		axis[0,0].set_title("Cost curve")
		axis[0,1].plot(np.arange(len(self.actor_cost)), self.actor_cost , c='b' , label='original')   
		axis[0,1].plot(np.arange(len(self.actor_cost)), self.get_smoothed(self.actor_cost), color='red', label='average of ten') 
		axis[0,1].legend(loc='best') 
		axis[0,1].set_ylabel('logPi*Td(V) in Policy Network')
		axis[0,1].set_xlabel('Trainning steps')
		axis[0,1].grid() 
		axis[0,1].set_title("Cost curve")
		axis[1,0].plot(np.arange(len(self.reward)), self.reward , c='b', label='original')   
		axis[1,0].plot(np.arange(len(self.reward)), self.get_smoothed(self.reward), color='red', label='average of ten') 
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
		plt.savefig(dirname+'/AC_v0_CartPole-V0.png')
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
			td_actor_error=actor.learn(s, a, td_error)   # true_gradient = grad[logPi(s, a) * td_error]

			l=float(td_error.detach().numpy()) # cannot numpy on tensor that require grad
			l_actor=float(td_actor_error.detach().numpy())
			critic.store_ep_cost(l)
			critic.store_ep_actor_cost(l_actor)

			#https://github.com/NVlabs/FUNIT/issues/23

			s = s_
			t += 1

			if done or t>=MAX_EP_STEPS:
				critic.store_ep_reward(ep_r)

				print('Ep: ', str(i_episode).rjust(4),
                        '| Ep_r: ', str(round(ep_r, 2)).rjust(7),
                        '| Steps: ', str(t).rjust(4))
				break

	t=round(time.time()-t1,2)
	print('Total running time: ',t,'s')
	dirname=os.path.dirname(__file__)
	critic.plot(dirname, t)


if __name__=='__main__':
	actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
	critic = Critic(n_features=N_F, lr=LR_C)   
	run_CartPoleV0()
	torch.save(actor.state_dict(), "Amodel.pth")
	torch.save(critic.state_dict(), "Cmodel.pth")
	print("Saved PyTorch Model State to model.pth")
	#actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
	#actor.load_state_dict(torch.load("Amodel.pth"))



