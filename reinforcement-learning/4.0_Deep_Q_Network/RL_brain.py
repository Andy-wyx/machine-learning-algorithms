"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
#import tensorflow as tf
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() 

# 这样是reproducible吧
np.random.seed(1)
tf.set_random_seed(1)
#tf2是tf.random.set_seed(),不过直接改了import


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,   # Q target network fixed target的更新周期
            memory_size=500,           # 
            batch_size=32,             # 取样的minibatch的size
            e_greedy_increment=None,   # 怎么设置比较好呢
            output_graph=False,        # 
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy 
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max         #一开始全随机啊？！有点狠

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))   # 这里s和s_都各有n个features需要记忆，+一个action，一个reward

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params') #从collection 中调用参数
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net 创建 eval 神经网络 及时提升参数------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input  用来接收 observation
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 用来接收 q_target 的值 for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables  
            # 这里是初始化w和b的参数，n_l1指的是将l1的神经元个数设置为10个，C_names则是tf中的一种collection，我们把eval net的params存进去，在更新 target_net 参数时调用它
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            # l1,eval_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
                #差不多就是l1=s*W1+b1的感觉=？*10+1*10=?*10
                # l1,w,x,b分别是m*n的matrix
                #w1=2*10，s=?*2,b1=1*10 l1=?*n_actions

            # second layer. collections is used later when assign to target net
            # l2,eval_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2
                #w2=10*n_actions,b2=1*n_actions
                #q_eval=l1*w2+b2=(?+1)*n_actions+1*n_actions=?*n_actions

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input 这里是s_
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        #hasattr(object, name) 用来判断对象是否包含对应的属性
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        #https://blog.csdn.net/csdn15698845876/article/details/73380803
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation}) #用刚才定义好的placeholder self.s来接受observation输入，计算q_eval，输出为action values
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters with eval network params
        # 每学200次（1000步），将target Q network的params改成eval network的最新参数 （Fixed Q-target strategy）
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\nlearn steps='+str(self.learn_step_counter)+': target_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size: #已经存满
            #https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else: #还未存满
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :] #随机抽取的32条记忆


        #用刚才定义好的placeholder self.s_和self.s来接受输入，
        # 其中，q_next记target network需要的是s_的features，也就是每条memory中的后2列
        # q_eval即eval network需要的则是s的features,也就是每条memory中的前两列
        # q_next and q eval: 32*2
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })
        
        #下面这一部分为什么要把q_eval赋值给q_target呢？
        # change q_target w.r.t q_eval's action

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # (其实我觉得别的值可以不归0，这一步没走到的action或许也有需要训练的地方，因为反向传播毕竟也有速率，这一步没用到的Q也可以去继续帮助用来tune params
        # 但是的确只要标准统一就行了，每次学习只专注于这次经过的action也很好，而且其他归零让计算更简便，思路更清晰
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1, 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.
        # 说白了就是我们Q target计算得到的矩阵，因为需要去对照好q eval的元素位置，所以要想办法调整一下原先的站位，
        # 也即s'中we take a' to maximize Q(s'), suppose index of a' in the q_target array is different to the index of a inside q_eval array 
        # then we need to find a way to match their position for the sake of matrix subtraction

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.
        # 具体在下面还有一个举例说明.

        q_target = q_eval.copy()

        # try these:
        '''import numpy as np 
            a=np.arange(32,dtype=np.int32)
            print(a)
            print(type(a))
            print(a.ndim)
            print(a.shape)'''
        batch_index = np.arange(self.batch_size, dtype=np.int32) #[0,1,2,...,31]
        eval_act_index = batch_memory[:, self.n_features].astype(int) #第三列 action 32*1
        reward = batch_memory[:, self.n_features + 1] #第四列 reward 32*1

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network 
        # 计算_train_op和loss不是只需要q_target 和 q_eval吗，为什么还要用s呢？？
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})   
        self.cost_his.append(self.cost) # plot时用的

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        #print('epsilon= '+str(self.epsilon))
        print('learn steps til now= '+str(self.learn_step_counter))

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig("cost_output.png")
        plt.show() # this clears the plot, needs to save first
        



