embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)
preds = Dense(2, activation='softmax')(l_lstm_sent)
model = Model(review_input, preds)

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_2 (InputLayer)             (None, 15, 100)       0
____________________________________________________________________________________________________
timedistributed_1 (TimeDistribute(None, 15, 200)       8217800     input_2[0][0]
____________________________________________________________________________________________________
bidirectional_2 (Bidirectional)  (None, 200)           240800      timedistributed_1[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 2)             402         bidirectional_2[0][0]
====================================================================================================
Total params: 8459002