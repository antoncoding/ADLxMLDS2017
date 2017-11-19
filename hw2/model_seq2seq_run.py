# coding: utf-8
from __future__ import print_function
import os
import numpy as np
import json
import pickle
import sys

from keras.engine.topology import Layer
from keras.layers import Input, LSTM, Dense, Dropout, Permute, Flatten, Activation, RepeatVector, Reshape
from keras.layers import merge, concatenate, multiply
from keras.preprocessing import sequence
from keras.models import Sequential, load_model, Model
from keras import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from keras import initializers, constraints, regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K.set_learning_phase(1)


DATA_DIR = sys.argv[1]
OUTFILE = sys.argv[2]
PEEROUT = sys.argv[3]

training_video_dir = DATA_DIR + 'training_data/video/'
testing_video_dir = DATA_DIR + 'testing_data/video/'

training_feature_dir = DATA_DIR + 'training_data/feat/'
testing_feature_dir = DATA_DIR + 'testing_data/feat/'
peer_feature_dir = DATA_DIR + 'peer_review/feat/'

id_train = []
id_test = []
id_peer = []

caption = {}
features = {}
for _type in ['training','testing']:
    labels = open(DATA_DIR+ _type + '_label.json').read()
    labels_data = json.loads(labels)

    for data in labels_data:
        ID = data['id'].split('.')[0]
        if _type == 'training':
            id_train.append(ID)
        else :
            id_test.append(ID)

        caption[ID] = []
        for cap in data['caption']:
            caption[ID].append(cap)

        if _type == 'training':
            features[ID] = np.load(training_feature_dir + data['id']+'.npy')
        else:
            features[ID] = np.load(testing_feature_dir + data['id']+'.npy')

with open(DATA_DIR+'peer_review_id.txt') as f:
    content = f.readlines()
    peerID = content.split('.')[0]  
    id_peer.append(ID)

for peerID in id_peer:
    features[peerID] = np.load(peer_feature_dir + peerID + '.npy')


for k,v in caption.items():
    for idx, cap in enumerate(v):
        v[idx] = 'bos ' + v[idx]
        v[idx] = v[idx] + ' eos'

all_text = []
for idKey, captions in caption.items():
    for cap in captions:
        all_text.append(cap)
tokenizer = Tokenizer(num_words = 100000, filters='1234567890!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(all_text)


reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
reverse_word_map[0] = ''

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#set caption threshold
k  = 4
num_vocab =2300
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

latent_dim = 256
num_samples = X_train.shape[0]

num_encoder_tokens = 4096
num_decoder_tokens = num_vocab

# model start

# max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_encoder_seq_length = 80
max_decoder_seq_length = 40

target_token_index = tokenizer.word_index

decoder_target_data = y_train[:,1:,:]
decoder_input_data = y_train[:,0:-1,:]
encoder_input_data = X_train

encoder_inputs = Input(shape=(80, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Imporvising
# output 線性轉換結果為permute 2
permute1 = Permute((2,1))(encoder_outputs)
dense1 = Dense(39)(permute1)
permute2 = Permute((2,1))(dense1)
dense2 = Dense(latent_dim,activation='softmax')(permute2)

# Use Concatenate to merge dense2 and permute2, 
# and then transfrom to (output_seq_len, latent_dim)
attention = concatenate(inputs=[dense2, permute2])
attention = Dense(latent_dim)(attention)


# Set up Decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(39, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_state=False, return_sequences=True)
decoder_lstm_outputs= decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Merge attention and lstmoutput 
merge_att_lstm = concatenate(inputs=[attention, decoder_lstm_outputs])
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(merge_att_lstm)

# Final Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.load_weights('weights.h5')

# # Run training


# print(model.summary())

# batch_size = 256
# epochs = 45

# try:
#     history = model.fit([encoder_input_data, decoder_input_data],
#               decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
    
#     with open('history.pickle', 'wb') as handle:
#         pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

# except KeyboardInterrupt:
#     print('\nSave Model')

# model.save('v728_model.h5')

def decode_str(model, Xtest):
    Xtest = Xtest.reshape(1,80,4096)
    Ydata = np.zeros((1,39,num_vocab))
    Ydata[0,0,target_token_index['bos']] = 1
    ans = ['bos']
    last_word = 'yo'
    while len(ans) < 39 and ans[-1] != 'eos':
        
        y_pred = model.predict([Xtest,Ydata])
        Ydata[0,len(ans),np.argmax(y_pred[0,len(ans)-1,:])] = 1
        
        word = reverse_word_map[np.argmax(y_pred[0,len(ans)-1,:])]
        ans.append(word)

    ans_string = ''
    last_word = 'Anton'
    for word in ans[1:-1]:
        if word != last_word:
            ans_string += word + ' '
            last_word = word
            
    return ans_string

# X_test = []
# for testID in id_test:
#     padded_captions = pad_sequences(tokenizer.texts_to_sequences(caption[testID]),maxlen=40,padding='post')
#     for text in padded_captions[:k]:
#         for idx, char in enumerate(text):
#             if char >= num_vocab:
#                 text[idx] = 0

# model = load_model('v728_model.h5'.format(num_vocab, latent_dim))


with open(OUTFILE,'w') as outfile:
    for tid in id_test:
        sample = features[tid]
        decoded_sentence = decode_str(model, np.asarray([sample]))
        outfile.write(tid+'.avi,'+decoded_sentence+'\n')

# with open(PEEROUT,'w') as outfile:
#     for tid in id_peer:
#         sample = features[tid]
#         decoded_sentence = decode_str(model, np.asarray([sample]))
#         outfile.write(tid+'.avi,'+decoded_sentence+'\n')