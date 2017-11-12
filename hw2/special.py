# coding: utf-8
from __future__ import print_function
from keras.layers import Input, LSTM, Dense, Dropout, Permute, Flatten, Activation, RepeatVector, merge
import os
import numpy as np
import json
from keras.preprocessing import sequence
from keras.models import Sequential, load_model, Model
from keras import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
import sys
import random as rn
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

rn.seed(12345)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)



tf.set_random_seed(1234)

#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

#K.set_learning_phase(1)
# DATA_DIR = 'datalite/'
DATA_DIR = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

training_video_dir = DATA_DIR + 'training_data/video/'
testing_video_dir = DATA_DIR + 'testing_data/video/'
training_feature_dir = DATA_DIR + 'training_data/feat/'
testing_feature_dir = DATA_DIR + 'testing_data/feat/'


id_train = []
id_test = []
caption = {}
features = {}
for _type in ['training','testing']:
    labels = open(DATA_DIR+ _type + '_label.json').read()
    labels_data = json.loads(labels)

    for data in labels_data:
        ID = data['id'].split('.')[0]
        if _type == 'training':
            id_train.append(ID)
        else:
            id_test.append(ID)

        caption[ID] = []
        for cap in data['caption']:
            caption[ID].append(cap)

        if _type == 'training':
            features[ID] = np.load(training_feature_dir + data['id']+'.npy')
        else:
            features[ID] = np.load(testing_feature_dir + data['id']+'.npy')


for k,v in caption.items():
    for idx, cap in enumerate(v):
        v[idx] = 'bos ' + v[idx]
        v[idx] = v[idx] + ' eos'


# In[5]:
all_text = []
for idKey, captions in caption.items():
    for cap in captions:
        all_text.append(cap)
tokenizer = Tokenizer(num_words = 100000, filters='1234567890!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(all_text)

import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
reverse_word_map[0] = ''

#set caption threshold
k  = 2
num_vocab =1000
X_train = []
y_train = []
idx_something = tokenizer.word_index['something']
for trainID in id_train:
    padded_captions = pad_sequences(tokenizer.texts_to_sequences(caption[trainID]),maxlen=40,padding='post')
    for text in padded_captions[:k]:
        for idx, char in enumerate(text):
            if char >= num_vocab:
                text[idx] = 0
        y_train.append(to_categorical(text ,num_classes=num_vocab))
        X_train.append(features[trainID])

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)


batch_size = 256  # Batch size for training.
epochs = 20  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = X_train.shape[0]

num_encoder_tokens = 4096
num_decoder_tokens = num_vocab

# max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_encoder_seq_length = 80
max_decoder_seq_length = 40

target_token_index = tokenizer.word_index

decoder_target_data = y_train[:,1:,:]
decoder_input_data = y_train[:,0:-1,:]
encoder_input_data = X_train

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True, return_sequences=False)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)


decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

print(model.summary())


try:
     model.fit([encoder_input_data, decoder_input_data],
               decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
#Save model
except KeyboardInterrupt:
     print('\nSave Model')

model.save('special.h5')

model = load_model('special.h5')
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

encoder_model.save("encoder.h5")
decoder_model.save("decoder.h5")

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['bos']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    count_len = 0
    last_char = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        count_len+=1
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index!= 0 and reverse_word_map[sampled_token_index]!='eos':
            sampled_char = reverse_word_map[sampled_token_index]
            if sampled_char != last_char:
                decoded_sentence += sampled_char+' '
                last_char = sampled_char

        if sampled_char=='eos' or count_len > 40:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

print("Training Text:")

print("Loading Testing Text...")

# X_test = []
for testID in id_test:
    padded_captions = pad_sequences(tokenizer.texts_to_sequences(caption[testID]),maxlen=40,padding='post')
    for text in padded_captions[:k]:
        for idx, char in enumerate(text):
            if char >= num_vocab:
                text[idx] = 0
#         X_test.append(features[testID])

with open(OUTPUT_PATH,'w') as outfile:
    # for special mission
    for tid in ['klteYv1Uv9A_27_33','5YJaS2Eswg0_22_26','UbmZAe5u5FI_132_141','JntMAcTlOF0_50_70','tJHUH9tpqPg_113_118']:
        sample = features[tid]
        decoded_sentence = decode_sequence(np.asarray([sample]))
        outfile.write(tid+'.avi,'+decoded_sentence+'\n')
        print(tid)
        print(decoded_sentence)
