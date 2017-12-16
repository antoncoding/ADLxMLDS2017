from agent_dir.agent import Agent
# from agent_dir.network import QNetwork
# from agent_dir.utility import ReplayMemory, SerialNumberManager
import numpy as np
import tensorflow as tf
import pdb
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)

        tf.reset_default_graph() 
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
        
        # tf. session
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        

        #if args.train_dqn:
        #    os.environ['CUDA_VISIBLE_DEVICES']='1'


        network_params = dict(
            action_size = env.get_action_space().n,
            learning_rate_minimum = 0.00025,
            learning_rate = 0.00025,
            learning_rate_decay_step = 50000,
            learning_rate_decay = 0.96
        )

        memory_params = dict(
            memory_size = 100000,
            action_size = env.get_action_space().n,
            observ_size = (84, 84, 4),
            batch_size = 4
        )

        self.q_network = QNetwork(network_params, self.sess)
        self.memory = ReplayMemory(memory_params)
        self.max_steps = 500000000

        # number of steps before learning starts
        self.learning_start = 50000

        # random action
        self.rand_epsilon = 0.05
        self.epsilon_start = 1
        self.epsilon_min = 0.1
        self.epsilon_end_time = 1000000
        self.step = 0

        self.gamma = 0.99
        self.target_update_frequency = 10000
        self.train_frequncy = 4
        self.testing_frequency = 5000

        self.total_loss = 0.0
        self.total_q = 0.0
        self.total_reward = 0.0
        self.update_cnt = 0

        self.ep_reward = 0.0
        self.ep_reward_list = []
        self.episodes = 0

        # double q and duel q option
        self.do_double_q = 1
        self.do_duel_q = 0


        # if args.train_dqn:
        #     self.model_saver = tf.train.Saver()    
        #     self.serial_man = SerialNumberManager('./models_dqn', './log', './params', './out')
        #     self.model_path, self.params_path, self.out_path, self.log_path = self.serial_man.get_paths('1-1')
        #     print('[INFO]:\n model_path={}\n params_path={}\n'.format(self.model_path, self.params_path))
        #     self.log_file = open(self.log_path, 'w')
        #     self.test_mode = False
        
        if args.test_dqn:
            model_saver = tf.train.Saver()
            model_saver.restore(self.sess, './final_models/dqn')
            self.test_mode = True
            



    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        pass
        
        
    def train(self):
        state = self.env.reset()
            
        for self.step in range(self.max_steps):
            if self.step == self.learning_start:
                self.total_reward = 0.0
                self.total_q = 0.0
                self.total_loss = 0.0
                self.update_cnt = 0
                self.ep_reward = 0.0
                self.ep_reward_list = []
                #self.episodes = 0
            
            # Decide an action
            action = self.make_action(state)
            
            # Act
            state, reward, done, info = self.env.step(action)

            # Update replay memory
            self.memory.store(state[:,:,-1], reward, action, done)

            # Update netowrks
            if self.step > self.learning_start:
                if self.step % self.train_frequncy == 0:
                    # Gradient descent
                    #print('updating network')
                    self.update_network()
                if self.step % self.target_update_frequency == self.target_update_frequency - 1:
                    #print('updating target network')
                    self.q_network.update_target_network()
            
            # model logging
            self.total_reward += reward

            
            # check done
            if done:
                state = self.env.reset()
                self.ep_reward_list.append(self.ep_reward)
                self.ep_reward = 0.0
                self.episodes += 1
            else:
                self.ep_reward += reward

            if self.step > self.learning_start and self.step % self.testing_frequency == 0:
                avg_loss = self.total_loss / self.update_cnt
                avg_q = self.total_q / self.update_cnt
                avg_reward = self.total_reward / self.testing_frequency
                msg_str = 'step = {}, loss={},  Q={}, reward={}'.format(self.step, avg_loss, avg_q, avg_reward)
                print(msg_str)
                self.log_file.write(msg_str + '\n')

                self.total_reward = 0.0
                self.total_q = 0.0
                self.total_loss = 0.0
                self.update_cnt = 0

                self.model_saver.save(self.sess, self.model_path, global_step=self.step)

            if self.step > self.learning_start and self.episodes % 100 == 99 and done:
                try:
                    avg_ep_reward = np.mean(self.ep_reward_list)
                except:
                    avg_ep_reward = 0.0
                msg_str = 'ep={}, avg. reward = {}'.format(self.episodes, avg_ep_reward)
                print(msg_str)
                self.log_file.write(msg_str + '\n')
                self.ep_reward = 0.0
                self.ep_reward_list = []

    def make_action(self, observation, test=True):
        
        if self.test_mode:
            #epsilon = self.rand_epsilon
            epsilon = 0.001
        else:
            epsilon = (self.epsilon_min + max(0., (self.epsilon_start - self.epsilon_min) \
                    * (self.epsilon_end_time - max(0., self.step - self.learning_start)) / self.epsilon_end_time))
        do_random_action = random.random() < epsilon
        if do_random_action:
            return self.env.get_random_action()
        else:
            return np.asscalar(self.q_network.q_action.eval({self.q_network.input: [observation]}, session=self.sess))
        

    
    def update_network(self):
        # sample training data from replay memory
        prev_state, actions, rewards, next_state, dones = self.memory.sample()
        
        # If using double DQN
        if self.do_double_q:
            # Decide action by action Q network
            q_actions = self.q_network.q_action.eval({self.q_network.input: next_state}, session=self.sess)
            # Determine expected future reward by value Q network
            feed_dict = {
                self.q_network.target_input: next_state,
                self.q_network.selection_idx: [[idx, val] for idx, val in enumerate(q_actions)]
            }
            future_qval = self.q_network.selected_q_val.eval(feed_dict, session=self.sess)
            target_q_val = (1. - dones) * self.gamma * future_qval + rewards

        else:
            future_qval = self.q_network.target_qval.eval({self.q_network.target_input: next_state}, session=self.sess)
            max_qval = np.max(future_qval, axis=1)
            target_q_val = (1. - dones) * self.gamma * max_qval + rewards
        
        feed_dict = {
            self.q_network.target_q_t: target_q_val,
            self.q_network.action: actions,
            self.q_network.input: prev_state,
            self.q_network.learning_rate_step: self.step
        }
        _, train_q, train_loss = self.sess.run([self.q_network.train_op, self.q_network.qval, self.q_network.loss], feed_dict=feed_dict)

        self.total_loss += train_loss
        self.total_q += np.mean(train_q)
        self.update_cnt += 1


class QNetwork:
    def __init__(self, params, sess):
        self.learning_rate = params['learning_rate']
        self.action_size = params['action_size'] 
        self.params = params
        self.sess = sess        
        self.build_train_network()
        self.build_target_network()
        self.build_optimizer()
        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

    def build_train_network(self):
        self.w = {}
        self.input = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.conv1, self.w['conv1_w'], self.w['conv1_b'] =self.conv2d(self.input, 32, [8, 8], [4, 4], scope='conv1')
        self.conv2, self.w['conv2_w'], self.w['conv2_b'] =self.conv2d(self.conv1, 64, [4, 4], [2, 2], scope='conv2')
        self.conv3, self.w['conv3_w'], self.w['conv3_b'] =self.conv2d(self.conv2, 64, [3, 3], [1, 1], scope='conv3')
        
        shape = self.conv3.get_shape().as_list()
        self.conv3_flat = tf.reshape(self.conv3, [-1, shape[1]*shape[2]*shape[3]])
        
        self.h1, self.w['h1_w'], self.w['h1_b'] = self.linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, scope='h1')
        self.qval, self.w['h2_w'], self.w['h2_b'] = self.linear(self.h1, self.action_size, scope='qval')
        self.q_action = tf.argmax(self.qval, 1)

        # Logging
        q_summary = []
        avg_q = tf.reduce_mean(self.qval, 0)
        for idx in range(self.action_size):
            q_summary.append(tf.summary.histogram('q/{}'.format(idx), avg_q[idx]))
        self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    def build_target_network(self):
        self.t_w = {}
        self.target_input = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.target_conv1, self.t_w['conv1_w'], self.t_w['conv1_b'] = \
            self.conv2d(self.target_input, 32, [8, 8], [4, 4], scope='target_conv1')
        self.target_conv2, self.t_w['conv2_w'], self.t_w['conv2_b'] = \
            self.conv2d(self.target_conv1, 64, [4, 4], [2, 2], scope='target_conv2')
        self.target_conv3, self.t_w['conv3_w'], self.t_w['conv3_b'] = \
            self.conv2d(self.target_conv2, 64, [3, 3], [1, 1], scope='target_conv3')
        
        shape = self.target_conv3.get_shape().as_list()
        self.target_conv3_flat = tf.reshape(self.target_conv3, [-1, shape[1]*shape[2]*shape[3]])
        
        self.target_h1, self.t_w['h1_w'], self.t_w['h1_b'] = \
            self.linear(self.target_conv3_flat, 512, activation_fn=tf.nn.relu, scope='target_h1')
        self.target_qval, self.t_w['h2_w'], self.t_w['h2_b'] = \
            self.linear(self.target_h1, self.action_size, scope='target_qval')
        
        self.selection_idx = tf.placeholder('int32', [None, 2], 'selection_idx')
        self.selected_q_val = tf.gather_nd(self.target_qval, self.selection_idx)

        with tf.variable_scope('target_update'):
            self.t_w_input = {}
            self.t_w_update = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_update[name] = self.t_w[name].assign(self.t_w_input[name])
    
    def build_optimizer(self):
        self.target_q_t = tf.placeholder('float32', [None], 'target_q_t')
        self.action = tf.placeholder('int32', [None], 'action')
        self.learning_rate_step = tf.placeholder('int32', None, name='learning_rate_step')
        action_one_hot = tf.one_hot(self.action, self.action_size, name='action_one_hot')
        action_q = tf.reduce_sum(tf.multiply(self.qval, action_one_hot), axis=1, name='action_q')

        self.td_error = self.target_q_t - action_q
        self.loss = tf.reduce_mean(clipped_error(self.td_error), name='loss')
        self.learning_rate_op = tf.maximum(self.params['learning_rate_minimum'],
        tf.train.exponential_decay(
            self.params['learning_rate'],
            self.learning_rate_step,
            self.params['learning_rate_decay_step'],
            self.params['learning_rate_decay'], 
            staircase=True))
        self.train_op = tf.train.RMSPropOptimizer(
            self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    def update_target_network(self):
        for w_key in self.w.keys():
            self.t_w_update[w_key].eval({self.t_w_input[w_key]: self.w[w_key].eval(session=self.sess)}, session=self.sess)

    def conv2d(self, x, n_filter, kerner_size, stride, activation_fn=tf.nn.relu, scope='conv2d'):
        with tf.variable_scope(scope):
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [kerner_size[0], kerner_size[1], x.get_shape()[-1], n_filter]
            
            w = tf.get_variable('w', kernel_size, tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(x, w, stride, padding='VALID')

            b = tf.get_variable('b', [n_filter], initializer=tf.constant_initializer(0.0))
            out = conv + b
            
            out = activation_fn(out)

        return out, w, b

    def linear(self, x, n_hid, activation_fn=None, scope='linear'):
        shape = x.get_shape().as_list()
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [shape[1], n_hid], tf.float32,
                tf.random_normal_initializer(stddev=0.02))
            b = tf.get_variable('bias', [n_hid],
                initializer=tf.constant_initializer(0))
            
            out = tf.matmul(x, w) + b
            
            if activation_fn != None:
                return activation_fn(out), w, b
            else:
                return out, w, b


class ReplayMemory:
    def __init__(self, params):
        self.memory_size = params['memory_size']
        self.observ_size = params['observ_size']
        self.batch_size = params['batch_size']
        self.actions = np.empty(self.memory_size, dtype=np.uint32)
        self.rewards = np.empty(self.memory_size, dtype=np.uint32)
        self.observ = np.empty((self.memory_size, self.observ_size[0], self.observ_size[1]), dtype=np.float32)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = 4
        # number of elements in the memory, <= self.memory_size
        self.count = 0
        # pointer to current element
        self.idx = 0

        # pre-allocation previous and next states
        self.prev_state = np.empty((self.batch_size, self.history_length, self.observ_size[0], self.observ_size[1]), dtype=np.float32)
        self.next_state = np.empty((self.batch_size, self.history_length, self.observ_size[0], self.observ_size[1]), dtype=np.float32)

    def store(self, observ, reward, action, terminal):
        # assert observ.shape == self.observ_size

        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.observ[self.idx, ...] = observ
        self.terminals[self.idx] = terminal
        self.count = max(self.count, self.idx + 1)
        self.idx = (self.idx + 1) % self.memory_size
    
    def get_state(self, index):
        #assert self.idx >= self.history_length - 1
        index = max(index, self.history_length - 1)
        return self.observ[index - (self.history_length - 1) : index + 1, ...]
    
    def sample(self):
        """
        Return a batch of training data
        """
        sample_idx = []
        for cnt in range(self.batch_size):
            while True:
                cand_idx = random.randint(self.history_length-1, self.count-1)
                if cand_idx >= self.idx and cand_idx - self.history_length < self.idx:
                    continue

                if self.terminals[(cand_idx - self.history_length):cand_idx].any():
                    continue
                
                break
            self.next_state[cnt, ...] = self.get_state(cand_idx)
            self.prev_state[cnt, ...] = self.get_state(cand_idx - 1)
            sample_idx.append(cand_idx)

            
        return (np.transpose(self.prev_state, (0, 2, 3, 1)), 
                self.actions[sample_idx], 
                self.rewards[sample_idx], 
                np.transpose(self.next_state, (0, 2, 3, 1)),
                self.terminals[sample_idx])

    def get_memory_size(self):
        return self.memory_size


class SerialNumberManager:
    """
    Manager for serial number generation
    """
    def __init__(self, model_base_path, log_base_path, params_base_path, out_base_path):
        self.model_base_path = model_base_path
        self.log_base_path = log_base_path
        self.params_base_path = params_base_path
        self.out_base_path = out_base_path
    
    def list_model_files(self):
        major_num = []
        minor_num = []
        for f in os.listdir(self.model_base_path):
            if f[0] != '.' and f != 'checkpoint':
                major_num.append(int(f.split('-')[0]))
                minor_num.append(int(f.split('-')[1].split('.')[0]))
        return major_num, minor_num

    def get_max_serial(self):
        model_major, model_minor = self.list_model_files()
        log_files = [float(f) for f in os.listdir(self.log_base_path) if f[0] != '.']
        params_files = [float(f.split('.')[0]) for f in os.listdir(self.params_base_path) if f[0] != '.']
        if len(model_major) == 0 or len(log_files) == 0 or len(params_files) == 0:
            serial = 0
        else:
            serial = max([max(model_major), max(log_files), max(params_files)])
        return int(serial)

    def gen_paths(self, serial=None):      
        if serial is None:
            serial = self.get_max_serial()

        m = os.path.join(self.model_base_path, str(serial + 1))
        l = os.path.join(self.log_base_path, str(serial + 1))
        p = os.path.join(self.params_base_path, str(serial + 1))
        return m, l, p

    def get_paths(self, serial=None):
        """
        Example of serial: '1-1'
        """
        if serial is None:
            major_num, minor_num = self.list_model_files()
            major = max(major_num)
            minor = max(minor_num)
        else:
            major = serial.split('-')[0]
            minor = serial.split('-')[1]

        m = os.path.join(self.model_base_path, str(major)+'-'+str(minor))
        p = os.path.join(self.params_base_path, str(major))
        o = os.path.join(self.out_base_path, str(major))
        l = os.path.join(self.log_base_path, str(major))
        return m, p, o, l


def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)