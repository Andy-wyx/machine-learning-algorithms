"""
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://localhost:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations") # 接收 observation
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num") # 接收我们在这个回合中选过的 actions
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")  # discounted_ep_rs_norm,用来估计Q value
        #seems more concise than _build_net function in DQN, due to the use of keras
        # fc1   type:tensor
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  ## tanh activation why tanh？
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2  type:tensor
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,    ##can I use softmax here directly？
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        #all_act:Tensor("fc2/BiasAdd:0", shape=(?, 2), dtype=float32)
        
        # 我们这个例子是离散的动作空间，神经网络架构上依然让l2的神经元数量==action_space，
        # 不过与DQN不同的是，DQN网络模拟Q函数，输出的是每个动作的价值，而policy network是模拟策略函数，输出的应当是某个动作或者说选择动作的策略，
        # 因此我们use softmax to convert to probability
        ## 连续动作空间问题的神经网络架构应该怎么设置呢，直接在最后一层用一个神经元输出，并通过函数映射到action_space的值域这样合理吗？

        #tf方法中参数logits In machine learning, the term “logits” refers to the raw outputs of a model before they are transformed into probabilities. 
        # Specifically, logits are the unnormalized outputs of the last layer of a neural network.
        # logit transformation(分对数变换)：p'=ln(p/(1-p))， it maps a p within (0,1) to (-inf,+inf). Looks like a decent mapping but they don't sum to 1.
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  

        #self.all_act_prob:Tensor("act_prob:0", shape=(?, 2), dtype=float32)
        
        ## 
        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss 
        ##
        with tf.name_scope('train'):     
            # minimize method internally includes compute_gradients() and apply_gradients()
            # the key is how 'compute_gradients() and apply_gradients()' are implemented!
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    #根据概率分布随机选一个动作
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    #DQN中是把s,a,r,s_做成一个数组放进有上限的memeory里，而这里PG是单步更新，
    # 存储s，a,r只是给reinforce拿来计算discounted rewards来估计Q的
    # Q(s,a)=E[R(trajectory)|s,a]
    # 那么除了初始的s和a，其余s和a不是没必要记录吗？
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    #好奇是哪里自动算梯度的？: self.train_op是_build_net中定义好的optimizer
    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards() # 这拿到的是一个list

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm
    
    #游戏进行到现在的累积weighted reward
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        ##难到不是只有最后一个有用吗
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        #len=N [0,1,2,3,4,5,N-1],reversed:[N-1,N-2,...,2,1], d[N-1]=rN-1, d[N-2]=rN-2+rN-1*Gamma, d[N-3]=rN-3+rN-2*Gamma+rN-3*Gamma^2,...
        # normalize episode rewards
        
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    
        



