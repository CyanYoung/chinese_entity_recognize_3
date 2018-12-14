from keras.layers import LSTM, Conv1D, Dense, Bidirectional, Dropout


win_len = 7


def cnn(embed_input, class_num):
    ca = Conv1D(filters=128, kernel_size=win_len, padding='valid', activation='relu')
    da1 = Dense(200, activation='relu')
    da2 = Dense(class_num, activation='softmax')
    x = ca(embed_input)
    x = da1(x)
    x = Dropout(0.2)(x)
    return da2(x)


def rnn(embed_input, class_num):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(class_num, activation='softmax')
    x = ba(embed_input)
    x = Dropout(0.5)(x)
    return da(x)
