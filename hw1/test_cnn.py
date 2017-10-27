# coding: utf-8
import numpy as np
import sys


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


# In[4]:


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


# In[8]:


dic_48_39, dic_39_char, dic_frame_label = get_map_dics()


# In[9]:


MFCC_TRAIN, MFCC_TEST = get_data(t='mfcc')


# In[10]:


TRAIN_LABEL = []
for spk_id, arr in MFCC_TRAIN:
    TRAIN_LABEL.append(dic_frame_label[spk_id])


from keras.models import Sequential,load_model
from keras.layers import Reshape
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
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
    X_new = []
    for sample in X:
        new_sample = []
        for idx, col in enumerate(sample):
            small_list = []
            if idx > 0 and idx < len(sample)-1:
                small_list.append(sample[idx-1].reshape(39,1))
                small_list.append(col.reshape(39,1))
                small_list.append(sample[idx+1].reshape(39,1))
            elif idx == 0:
                small_list.append(col.reshape(39,1))                                      
                small_list.append(col.reshape(39,1))                                        
                small_list.append(sample[idx+1].reshape(39,1)) 
            elif idx == len(sample)-1:
                small_list.append(sample[idx-1].reshape(39,1))                                      
                small_list.append(col.reshape(39,1))
                small_list.append(col.reshape(39,1)) 
            new_sample.append(np.asarray(small_list))
        X_new.append(np.asarray(new_sample))
    return np.asarray(X_new)

X_train_np = reshape_to_cnn_input(X_train_np)
X_test = reshape_to_cnn_input(X_test)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train_np, y_train_np, test_size=0.18, random_state=42)
print("Yay: X_train shape: {}".format(X_train.shape))


opt = RMSprop(lr=0.001, decay=1e-6, clipvalue=0.5)

# prefix ="cnn_2d_2"

model = Sequential()

model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu',padding='same'),input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3],X_train.shape[4])))
model.add(TimeDistributed(MaxPooling2D(pool_size=(3,2))))
model.add(Dropout(0.4))

model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(GRU(512, recurrent_dropout = 0.4, dropout=0.4, return_sequences=True, activation='relu', implementation=2)))
model.add(Bidirectional(GRU(128, recurrent_dropout = 0.45, dropout=0.45, return_sequences=True, activation='relu', implementation=2)))
model.add(Bidirectional(GRU(64, recurrent_dropout = 0.43, dropout=0.43, return_sequences=True, activation='relu', implementation=2)))
model.add(TimeDistributed(Dense(NUM_CLASS, activation='softmax')))
model.compile(loss='categorical_crossentropy',optimizer=opt)

print(model.summary())

model.load_weights('cnn_weights.hdf5')

# earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1)

# checkpoint = ModelCheckpoint(filepath=prefix+'weight.hdf5',verbose=1,save_best_only=True, save_weights_only=True, monitor='val_loss')

# history = model.fit(X_train, y_train, epochs=80, batch_size=64, validation_data=(X_val, y_val), callbacks=[earlystopping,checkpoint])

# model.load_weights(prefix+'weight.hdf5')

# with open(prefix+'_history.pickle', 'wb') as handle:
#     pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

out = model.predict(X_test)

# model.save(prefix+'model008.h5')


index_label_d = {}
for k, v in label_index.items():
    index_label_d[v] = k

output_list = []
for sentence in out:
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

print("result written to {}".format(OUTFILE_NAME))


