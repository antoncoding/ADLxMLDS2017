from agent_dir.agent import Agent
import os
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    I = I.astype(np.float).ravel()
    return I.reshape((80,80,1))

class Agent_PG(Agent):
    def __init__(self, env, args):
 
        super(Agent_PG,self).__init__(env)
        self.action_size = 3
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.model_name = "pong_v7"
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.prev_x = None
        self.record = []
        self.data_path = "data/{}/".format(self.model_name)
 
        if args.test_pg:
            print('Loading trained model')
            self.load()

        if args.train_pg:
            print('Building new model')
            self.model = self.build_model()
            # self.model.summary()
            with open("models_pg/{}.json".format(self.model_name), 'w') as f:
                json.dump(self.model.to_json(), f)
            if not (os.path.exists(self.data_path)):
                os.makedirs(self.data_path)
 
 
    def init_game_setting(self):
        self.prev_x = None
 
 
    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (6, 6), activation="relu",padding="same",input_shape=(80, 80, 1), strides=(3, 3), kernel_initializer="he_uniform"))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
 
        return model
 
 
    def memorize(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)
 
 
    def make_action(self, observation, test=True):
        state = preprocess(observation) # shape -> (80, 80, 1)
        x = state - self.prev_x if self.prev_x is not None else np.zeros(state.shape)
        self.prev_x = state
        
        x = np.expand_dims(x,axis=0)
        aprob = self.model.predict(x, batch_size=1).flatten()
        self.probs.append(aprob)
        action = np.random.choice(self.action_size, 1, p=aprob)[0]
        if test:
            return action + 1 
        else :
            return action, aprob, x
 
 
    def discount_rewards(self, rewards):
        # [0, 0, 0, 1] - > [0.97, 0.98, 0.99, 1]
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
 
 
    def fit(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards-np.mean(rewards)) / np.std(rewards) # standarize 
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]),axis = 1)
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
 
 
    def save(self):
        self.model.save_weights(os.path.join('models_pg',"{}_weight.hdf5".format(self.model_name)))
 
    def load(self):
        from keras.models import load_model, model_from_json
        self.model = self.build_model()
        model_weight_name = 'final_models/pong_weight.hdf5'
        self.model.load_weights(model_weight_name)
        print("loaded weight: {}".format(model_weight_name))
 
 
    def save_record(self):
        import csv
 
        x = []
        y = []
 
        for k in range(0,len(self.record)):
            x.append(self.record[k][0])
            size = min(self.record[k][0],30)
            ma_score = np.array([ self.record[t][1] for t in range(self.record[k][0] - size,self.record[k][0]) ])
            y.append(ma_score.mean() )
 
        with open("{}score.csv".format(self.data_path), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(x)
            writer.writerow(y)
 
    def train(self, env):
        score = 0
        episode = 0
        state = self.env.reset()
        prev_x = None
 
        while True:
            try:
                action, prob, x = self.make_action(state, test=False) # This is training code
                state, reward, done, info = self.env.step(action+1) 
                score += reward
                self.memorize(x, action, prob, reward)
 
                if done:
                    episode += 1
                    print('Training Episode: {} - Score: {}.'.format(episode, score))
                    self.record.append([episode, score])
                    score = 0
                    state = self.env.reset()
                    self.prev_x = None
                    # if episode > 1 and episode % 5 == 0:
                    self.fit()
                    if episode > 1 and episode % 10 == 0:
                        self.save()
                        self.save_record()
            
            except KeyboardInterrupt:
                self.fit()
                self.save()
                self.save_record()
                return
