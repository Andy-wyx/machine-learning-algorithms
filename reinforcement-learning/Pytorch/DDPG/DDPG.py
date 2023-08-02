import argparse
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

############################################################### Hyperparameters ############################################################

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=50000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=64, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--render', default=False, type=bool) 
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=1000, type=int) # num of games 100000
parser.add_argument('--max_length_of_trajectory', default=2000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=10, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
dir_name=os.path.dirname(__file__)
env = gym.make(args.env_name).unwrapped

if args.seed:
	env.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) # min value, but haven't been used

directory = dir_name+'/logs/'

############################################################# DDPG Model ##################################################################

class Replay_buffer():
	'''
	Code based on:
	https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
	Expects tuples of (state, next_state, action, reward, done)
	'''
	def __init__(self, max_size=args.capacity):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def push(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind:
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim) # action_dim=1

		self.max_action = max_action

	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		#similar to what we used in AC model, here we move the std (noise) out in the main loop, this x can be regarded as the mu.
		x = self.max_action * torch.tanh(self.l3(x)) #[-2,2]
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400 , 300)
		self.l3 = nn.Linear(300, 1)

	def forward(self, s, a):
		s = F.relu(self.l1(torch.cat([s, a], 1)))
		s = F.relu(self.l2(s))
		Q = self.l3(s)
		return Q


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict()) # initialize target network with eval network parameters.
		self.actor_optimizer = optim.Adam(self.actor.parameters(), args.learning_rate)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = optim.Adam(self.critic.parameters(), args.learning_rate)
		self.replay_buffer = Replay_buffer()
		self.writer = SummaryWriter(directory)
		self.num_critic_update_iteration = 0
		self.num_actor_update_iteration = 0
		self.num_training = 0

		self.reward=[]
		self.policy_cost=[]
		self.value_cost=[]
		self.total_cost=[]



	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def update(self):

		for it in range(args.update_iteration):
			# Sample replay buffer
			s, s_, a, r, d = self.replay_buffer.sample(args.batch_size)
			s = torch.FloatTensor(s).to(device)
			a = torch.FloatTensor(a).to(device)
			s_ = torch.FloatTensor(s_).to(device)
			done = torch.FloatTensor(d).to(device)
			r = torch.FloatTensor(r).to(device)

			# Compute the target Q value
			target_Q = self.critic_target(s_, self.actor_target(s_))   # a typical bootstrapping
			target_Q = r + ((1 - done) * args.gamma * target_Q).detach()          

			# Get current Q estimate
			current_Q = self.critic(s, a)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)
			self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(s, self.actor(s)).mean() ##? this is mean of Q
			self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			l1=float(actor_loss.detach().numpy())
			l2=float(critic_loss.detach().numpy())
			self.policy_cost.append(l1)
			self.value_cost.append(l2)
			self.total_cost.append(l1+l2)

			# Update the frozen target models
			# a little different from our implementation in pytorch-DQN and tensorflow DQN/DDPG. Instead of 100% copy per iters, progressive update is used here.
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

			self.num_actor_update_iteration += 1
			self.num_critic_update_iteration += 1

	def save(self):
		torch.save(self.actor.state_dict(), directory + 'actor.pth')
		torch.save(self.critic.state_dict(), directory + 'critic.pth')
		# print("====================================")
		# print("Model has been saved...")
		# print("====================================")

	def load(self):
		self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
		self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
		print("====================================")
		print("model has been loaded...")
		print("====================================")

	def plot(self):
		#plt.ion()
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
		axis[0,1].set_ylabel('logPi*td(V) in Policy Network')
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
		axis[1,1].set_ylabel('Total loss')
		axis[1,1].set_xlabel('Trainning steps')
		axis[1,1].grid() 
		axis[1,1].set_title("Total Cost")
		figure.tight_layout()
		plt.savefig(dir_name+'/DDPG_Continous_Pendulum-V0.png')
		plt.show()

def get_smoothed(original):
	smoothed=[sum(original[e-9:e+1])/10 if e>9 else sum(original[:e+1])/(e+1) for e in range(len(original))]
	return smoothed

def main():
	agent = DDPG(state_dim, action_dim, max_action)
	ep_r = 0
	if args.mode == 'test':
		agent.load()
		for i in range(args.test_iteration):
			s = env.reset()
			for t in count():
				a = agent.select_action(s)
				s_, r, done, info = env.step(np.float32(a))
				ep_r += r
				env.render()
				if done or t >= args.max_length_of_trajectory:
					print("Ep_i\t{:>2}, the ep_r is \t{:>9.2f}, the step is \t{:>4}".format(i, ep_r, t))
					ep_r = 0
					break
				s = s_

	elif args.mode == 'train':
		print("====================================")
		print("Collection Experience...")
		print("====================================")
		if args.load: agent.load()
		for i in range(args.max_episode):
			s = env.reset()
			for t in count():
				a = agent.select_action(s)

				# add noise to action
				#np.random.normal(mean, std, shape) size=1
				a = (a + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
					env.action_space.low, env.action_space.high)

				s_, r, done, info = env.step(a)

				##some reward engineeering?

				ep_r += r
				if args.render and i >= args.render_interval : env.render()
				agent.replay_buffer.push((s, s_, a, r, np.float(done)))
				# if (i+1) % 10 == 0:
				#     print('Episode {},  The memory size is {} '.format(i, len(agent.replay_buffer.storage)))

				s= s_
				if done or t >= args.max_length_of_trajectory:
					agent.reward.append(ep_r)

					#tensorboard --logdir=logs， note that you have to cd to the upper folder (DDPG) before using this command
					agent.writer.add_scalar('ep_r', ep_r, global_step=i)

					if i % args.print_log == 0:
						print("Ep_i\t{:>6},|ep_r:\t{:>9.2f},|step:\t{:>4}".format(i,ep_r,t))
					ep_r = 0
					break

			if i % args.log_interval == 0:
				agent.save()
			# once the memory buffer is full , start trainning
			if len(agent.replay_buffer.storage) >= args.capacity-1:
				agent.update() 
		agent.plot()

	else:
		raise NameError("mode wrong!")

if __name__ == '__main__':
	main()
