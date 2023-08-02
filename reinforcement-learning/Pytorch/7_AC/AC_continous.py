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
MAX_EP_STEPS = 300   # maximum time steps in one episode
RENDER = True   
GAMMA = 0.9   # reward discount in TD error
LR_A = 0.003   # learning rate for actor
LR_C = 0.01   # learning rete for critic

###########################################  parameters  #################################################
np.random.seed(1)
torch.manual_seed(1)
env = gym.make('Pendulum-v0')
env.seed(1)
env = env.unwrapped

N_S = env.observation_space.shape[0]
A_BOUND = env.action_space.high

################################################ Model ######################################################
class Actor_Net(nn.Module):
	def __init__(self, n_features, n_hidden, n_outputs):
		super(Actor_Net, self).__init__()
		self.l1 = nn.Linear(n_features, n_hidden)
		self.mu = nn.Linear(n_hidden, n_outputs)
		self.sigma = nn.Linear(n_hidden, n_outputs)


	def forward(self, x):
		x = F.linear(x,self.l1.weight.clone(),self.l1.bias)
		x = F.relu(x)
		mu = F.linear(x,self.mu.weight.clone(),self.mu.bias)
		mu = torch.tanh(mu) # (-1,1)
		sigma = F.linear(x,self.sigma.weight.clone(),self.sigma.bias)
		sigma = F.softplus(sigma) #https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html

		return mu, sigma


class Actor(object):
	def __init__(self, n_features, action_bound, n_hidden=30, lr=0.0001):
		self.n_features = n_features
		self.action_bound = action_bound
		self.n_hidden = n_hidden
		self.lr = lr

		self._build_net()

		self.policy_cost=[]


	def _build_net(self):
		self.actor_net = Actor_Net(self.n_features, self.n_hidden, 1)
		self.optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr)

	#in this env with continous action space we use normal distribution to sample our real choice, 
	# the actor network is to provide the best mu and sigma for our normal distribution.
	## why normal distribution
	def normal_dist(self, s):
		s = torch.Tensor(s[np.newaxis, :])
		mu, sigma = self.actor_net(s)
		# in this pendulum env, range of the action (torque) is [-2.0,2.0]
		# from my current understanding this 0.1 makes sure that std is at least 0.1 ensuring some random exploration around the mean.
		mu, sigma = (mu*2).squeeze(),  (sigma+0.1).squeeze()    #squeeze() decrease 1 dim,unsqueeze increase 1 dim
		normal_dist = torch.distributions.Normal(mu, sigma)   # get the normal distribution of average=mu and std=sigma
		return normal_dist


	def choose_action(self, s):
		normal_dist = self.normal_dist(s)
		# sample action according to the distribution and clamp the sampling output into the range of [min,max] if not already.
		self.action = torch.clamp(normal_dist.sample(), self.action_bound[0], self.action_bound[1])   
		return self.action


	def learn(self, s, a, td):
		normal_dist = self.normal_dist(s)
		# log_prob get the probability of action a under the distribution of normal_dist,internally it will compute logarithm of pdf of normal distribution.
		log_prob = normal_dist.log_prob(a)   
		exp_v = log_prob * td.float()   # advantage (TD_error) guided loss
		# Add cross entropy cost to encourage exploration. entropy() is an abstract method in super class distribution. It is implemented in each of the inherited distribution class such as normal
		exp_v += 0.01*normal_dist.entropy()   
		loss = -exp_v   # max(v) = min(-v)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return exp_v


class Critic_Net(nn.Module):
	def __init__(self, n_features, n_hidden, n_outputs):
		super(Critic_Net, self).__init__()
		self.l1 = nn.Linear(n_features, n_hidden)
		self.v = nn.Linear(n_hidden, n_outputs)


	def forward(self, x):
		x =F.linear(x, self.l1.weight.clone(),self.l1.bias)
		x = F.relu(x)
		x = F.linear(x, self.v.weight.clone(),self.v.bias)
		return x


class Critic(object):
	def __init__(self, n_features, n_hidden=30, n_output=1, lr=0.01):
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.n_output = n_output
		self.lr = lr

		self._build_net()

		self.value_cost=[]
		self.reward=[]
		self.total_cost=[]


	def _build_net(self):
		self.critic_net = Critic_Net(self.n_features, self.n_hidden, self.n_output)
		self.optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)


	def learn(self, s, r, s_):
		s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
		v, v_ = self.critic_net(s), self.critic_net(s_)
		td_error = torch.mean(r + GAMMA * v_.double() - v.double())
		loss = td_error ** 2

		self.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		self.optimizer.step()

		return td_error
	
def plot(dirname,t,actor,critic):
		figure, axis = plt.subplots(2, 2)
		axis[0,0].plot(np.arange(len(critic.value_cost)), critic.value_cost , c='b' , label='original')   
		axis[0,0].plot(np.arange(len(critic.value_cost)), get_smoothed(critic.value_cost), color='red', label='average of ten') 
		axis[0,0].legend(loc='best') 
		axis[0,0].set_ylabel('Td error of V in Value Network')
		axis[0,0].set_xlabel('Trainning steps')
		axis[0,0].grid() 
		axis[0,0].set_title("Value loss")
		axis[0,1].plot(np.arange(len(actor.policy_cost)), actor.policy_cost , c='b' , label='original')   
		axis[0,1].plot(np.arange(len(actor.policy_cost)), get_smoothed(actor.policy_cost), color='red', label='average of ten') 
		axis[0,1].legend(loc='best') 
		axis[0,1].set_ylabel('logPi*Td(V) in Policy Network')
		axis[0,1].set_xlabel('Trainning steps')
		axis[0,1].grid() 
		axis[0,1].set_title("Policy loss")
		axis[1,0].plot(np.arange(len(critic.reward)), critic.reward , c='b', label='original')   
		axis[1,0].plot(np.arange(len(critic.reward)), get_smoothed(critic.reward), color='red', label='average of ten') 
		axis[1,0].legend(loc='best') 
		axis[1,0].set_ylabel('Reward')
		axis[1,0].set_xlabel('Episode')
		axis[1,0].grid() 
		axis[1,0].set_title("Reward Curve")
		axis[1,1].plot(np.arange(len(critic.total_cost)), critic.total_cost , c='b', label='original')   
		axis[1,1].plot(np.arange(len(critic.total_cost)), get_smoothed(critic.total_cost), color='red', label='average of ten') 
		axis[1,1].set_ylabel('policy_loss+value_loss+entropy_loss')
		axis[1,1].set_xlabel('Trainning steps')
		axis[1,1].grid() 
		axis[1,1].set_title("Total Cost")
		figure.tight_layout()
		plt.savefig(dirname+'/AC_Continous_Pendulum-V0.png')
		plt.show()

def get_smoothed(original):
	smoothed=[sum(original[e-9:e+1])/10 if e>9 else sum(original[:e+1])/(e+1) for e in range(len(original))]
	return smoothed

######################################################### run Pendulum-V0 ##############################################################
def run_PendulumV0(actor,critic):
	t1=time.time()
	for i_episode in range(MAX_EPISODE):
		s = env.reset()
		t = 0
		ep_r = 0
		while True:
			if RENDER: env.render()
			a_tensor = actor.choose_action(s)  # a tensor

			a=np.array([float(a_tensor)])
			s_, r, done, info = env.step(a)  # require an numpy array [] containning the float action number
			r /= 10
			r+=0.4

			td_error = critic.learn(s, r, s_)   # gradient = grad[r + gamma * V(s_) - V(s)]
			td_actor_error=actor.learn(s, a_tensor, td_error)   # gradient = grad[logPi(s, a) * td_error]

			l=float(td_error.detach().numpy()) # cannot numpy on tensor that require grad
			l_actor=float(td_actor_error.detach().numpy())
			critic.value_cost.append(l)
			actor.policy_cost.append(l_actor)
			critic.total_cost.append(l+l_actor)

			s = s_
			t += 1
			ep_r+=r
			if done or t >= MAX_EP_STEPS:
				critic.reward.append(ep_r)
				print('Ep: ', str(i_episode).rjust(4),
                        '| Ep_r: ', str(round(ep_r, 2)).rjust(7),
                        '| Steps: ', str(t).rjust(4))
				break
	t=round(time.time()-t1,2)
	print('Total running time: ',t,'s','Avg Time Per Ep: ', t/MAX_EPISODE,'s')
	dirname=os.path.dirname(__file__)
	plot(dirname, t,actor,critic)

if __name__=='__main__':
	actor = Actor(n_features=N_S, lr=LR_A, action_bound=[float(-A_BOUND), float(A_BOUND)])
	critic = Critic(n_features=N_S, lr=LR_C)
	dir_path=os.path.dirname(__file__)
	actor_model_path=dir_path+'/actor_continuous.pth'
	critic_model_path=dir_path+'/critic_continuous.pth'
	if os.path.exists(actor_model_path):
		actor.load_state_dict(torch.load(actor_model_path))
	if os.path.exists(critic_model_path):
		critic.load_state_dict(torch.load(critic_model_path))
		print("model Loaded.")
	run_PendulumV0(actor,critic)

	#torch.save(actor.state_dict(), actor_model_path)
	#torch.save(critic.state_dict(), critic_model_path)
	#print("Saved PyTorch Natural AC Model State to actor_continous.pth and critic_continous.pth")

	##这一节主要就是为什么用GaussDist，怎么计算，entropy函数的实现等