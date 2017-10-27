# coding: utf-8
import sys
import numpy as np

DATA_DIR = sys.argv[1]
OUTFILE_NAME = sys.argv[2]

_MAP_FILE_PATH = DATA_DIR + 'phones/48_39.map'
_MAP_CHAR_PATH = DATA_DIR + '48phone_char.map'

_TRAIN_LABEL_PATH = DATA_DIR + 'label/train.lab'

_FBANK_TRAIN_PATH = DATA_DIR + 'fbank/train.ark'
_FBANK_TEST_PATH = DATA_DIR + 'fbank/test.ark'
_MFCC_TRAIN_PATH = DATA_DIR + 'mfcc/train.ark'
_MFCC_TEST_PATH = DATA_DIR + 'mfcc/test.ark'

def get_map_dics():
    dic = {}
    f_map = open(_MAP_FILE_PATH, 'r')
    for line in f_map:
        dic[line.split('\t')[0]] = line.split('\t')[1].split('\n')[0]
    f_map.close()
    
    dic_2 = {}
    ph_map = open(_MAP_CHAR_PATH,'r')
    for line in ph_map:
        dic_2[line.split('\t')[0]] = line.split('\t')[2].split('\n')[0]
    ph_map.close()
    
    dic_frame_label = {}
    f_label = open(_TRAIN_LABEL_PATH,'r')
    for line in f_label:
        dic_frame_label[line.split(',')[0]] = dic_2[dic[line.split(',')[1].split('\n')[0]]]                                                                    
    f_label.close()
    
    return dic, dic_2, dic_frame_label


def strim_n_to_num(s):
    return float(s.split('\n')[0])

def get_data(t='fbank'):
    train_data = []
    test_data = []
    if t=='fbank':
        train_f = open(_FBANK_TRAIN_PATH,'r')
    elif t == 'mfcc':
        train_f = open(_MFCC_TRAIN_PATH,'r')
    for line in train_f:
        train_data.append([line.split(' ')[0],[strim_n_to_num(x) for x in line.split(' ')[1:]] ])
    train_f.close()
    
    if t=='fbank':
        test_f = open(_FBANK_TEST_PATH,'r')
    elif t == 'mfcc':
        test_f = open(_MFCC_TEST_PATH,'r')
    for line in test_f:
        test_data.append([line.split(' ')[0], [strim_n_to_num(x) for x in line.split(' ')[1:]] ])
    test_f.close()
    
    return train_data, test_data


dic_48_39, dic_39_char, dic_frame_label = get_map_dics()

MFCC_TRAIN, MFCC_TEST = get_data(t='mfcc')

TRAIN_LABEL = []
for spk_id, arr in MFCC_TRAIN:
    TRAIN_LABEL.append(dic_frame_label[spk_id])

from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing import sequence
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from keras import metrics
from keras.models import load_model
import pickle


def get_train_test_data(train_raw, test_raw):
    X_train = []
    X_train_ids = []
    X_test = []
    X_test_ids = []
    y_train = []
    #     for training data
    tmp_sp_id = train_raw[0][0].split('_')[0]
    tmp_stn_id =  train_raw[0][0].split('_')[1]
    count_label = 0
    tmp_data_array = []
    tmp_data_array_y = []
    for rawid, arr in train_raw:
        if rawid.split('_')[0] == tmp_sp_id and rawid.split('_')[1] == tmp_stn_id:
            tmp_data_array.append(arr)
            tmp_data_array_y.append(TRAIN_LABEL[count_label])
        else:
            X_train_ids.append('{}_{}'.format(tmp_sp_id, tmp_stn_id))
            X_train.append(tmp_data_array)
            y_train.append(tmp_data_array_y)
            tmp_data_array = [arr]
            tmp_data_array_y = [TRAIN_LABEL[count_label]]
        
        count_label += 1
        tmp_sp_id = rawid.split('_')[0]
        tmp_stn_id = rawid.split('_')[1]

    X_train_ids.append('{}_{}'.format(tmp_sp_id, tmp_stn_id))
    X_train.append(tmp_data_array)
    y_train.append(tmp_data_array_y)
    

    #     for testing part
    tmp_sp_id = test_raw[0][0].split('_')[0]
    tmp_stn_id =  test_raw[0][0].split('_')[1]
    tmp_data_array=[]
    for rawid, arr in test_raw:
        if rawid.split('_')[0] == tmp_sp_id and rawid.split('_')[1] == tmp_stn_id:
            tmp_data_array.append(arr)
        else:
            X_test_ids.append('{}_{}'.format(tmp_sp_id, tmp_stn_id))
            X_test.append(tmp_data_array)
            tmp_data_array = [arr]
        tmp_sp_id = rawid.split('_')[0]
        tmp_stn_id = rawid.split('_')[1]
    
    X_test_ids.append('{}_{}'.format(tmp_sp_id, tmp_stn_id))
    X_test.append(tmp_data_array)
    tmp_data_array = [arr]

    return X_train, y_train, X_test, X_train_ids, X_test_ids


# In[44]:


X_train, y_train, X_test, X_train_ids, X_test_ids = get_train_test_data(MFCC_TRAIN, MFCC_TEST)

from sklearn.preprocessing import StandardScaler
def scale_sample(x):
    new_x = []
    for sample in x:
        scaler = StandardScaler()
        scaler.fit(sample)
        new_x.append(scaler.transform(sample))
    return new_x

X_train = scale_sample(X_train)
X_test = scale_sample(X_test)

NUM_CLASS = 40

train_max_stn_len = max([len(s) for s in X_train])
test_max_stn_len = max([len(s) for s in X_test])
MAX_STN_LEN = max(test_max_stn_len, train_max_stn_len)

pad_X_train = []
for idx, sentenc in enumerate(X_train):
    for i in range(MAX_STN_LEN-len(sentenc)):
        sentenc = np.append(sentenc, [np.zeros(39)], axis=0)
        y_train[idx].append('.')
    pad_X_train.append(sentenc)
X_train = pad_X_train                        

pad_X_test = []
for idx, sentenc in enumerate(X_test):
    for i in range(MAX_STN_LEN-len(sentenc)):
        sentenc = np.append(sentenc, [np.zeros(39)], axis=0)
    pad_X_test.append(sentenc)
X_test= pad_X_test

# In[47]:


X_train_np = np.asarray(X_train)
X_test= np.asarray(X_test)



y_train_np = []
label_index = {'.':0}
count_label = 1
for sentence in y_train:
    tmp_arr = []
    for char in sentence:
        if char not in label_index.keys():
            label_index[char] = count_label
            count_label += 1
        tmp_arr.append(label_index[char])
    y_train_np.append(tmp_arr)




y_train_np = np.asarray(y_train_np)


# In[91]:


new_y = []
for arr in y_train_np:
    tmp_metrix = np.zeros((MAX_STN_LEN, NUM_CLASS))
    for idx, num in enumerate(arr):
        tmp_metrix[idx,num] = 1
    new_y.append(tmp_metrix)
y_train_np = np.asarray(new_y)


# test for cnn
def reshape_to_cnn_input(X):
    x_new = []
    for sample in X:
        x_new.append(sample.reshape(1,sample.shape[0],sample.shape[1]))
    return np.asarray(x_new)

#X_train_np = reshape_to_cnn_input(X_train_np)
#X_test = reshape_to_cnn_input(X_test)

from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(X_train_np, y_train_np, test_size=0.18, random_state=777)


opt = RMSprop(lr=0.0004, decay=1e-6, clipvalue=0.5)
model = Sequential()
model.add(Bidirectional(GRU(512, recurrent_dropout = 0.35, dropout=0.4, return_sequences=True, activation='relu', implementation=2),input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Bidirectional(GRU(256, recurrent_dropout = 0.5, dropout=0.5, return_sequences=True, activation='relu', implementation=2)))
model.add(Bidirectional(GRU(128, recurrent_dropout = 0.5, dropout=0.5, return_sequences=True, activation='relu', implementation=2)))
model.add(GRU(256, recurrent_dropout = 0.5, dropout=0.5, return_sequences=True, activation='relu', implementation=2))
model.add(TimeDistributed(Dense(NUM_CLASS, activation='softmax')))
model.compile(loss='categorical_crossentropy',optimizer=opt)

#print("model got")

# model = load_model("models/697/model007_cont_4.h5")

print(model.summary())
# earlystopping = EarlyStopping(monitor='val_loss', patience = 40, verbose=1)
# checkpoint = ModelCheckpoint(filepath='trybest_cont_4.hdf5',verbose=1,save_best_only=True, save_weights_only=True, monitor='val_loss')

# model.fit(X_train, y_train, epochs=450, batch_size=90, validation_data=(X_val, y_val), callbacks=[earlystopping,checkpoint])

# model_best = model

model.load_weights('rnn_weights.hdf5')

out_last = model.predict(X_test)


index_label_d = {}
for k, v in label_index.items():
    index_label_d[v] = k

output_list = []
for sentence in out_last:
    tmp_list = []
    tmp_char = ''
    continuous = 0
    last_write = ''
    for s in sentence:
        this_char = index_label_d[np.argmax(s)]
        if this_char != tmp_char and this_char != '.':
            if continuous > 1 and last_write != tmp_char :
                tmp_list.append(tmp_char)
                last_write = tmp_char
            continuous = 0
        elif this_char == tmp_char and this_char != '.': 
            continuous += 1
        tmp_char = this_char        
    output_list.append(''.join(tmp_list[1:]))

outputFile = open(OUTFILE_NAME,'w')
outputFile.write('id,phone_sequence\n')
for idx, seq in enumerate(output_list):
    outputFile.write('{},{}\n'.format(X_test_ids[idx],seq))
outputFile.close()