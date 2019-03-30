from keras.layers import Conv1D, LSTM, Dense, Bidirectional, Dropout, Multiply


win_len = 7


def cnn_crf(embed_input, crf):
    conv = Conv1D(filters=128, kernel_size=win_len, padding='valid')
    gate = Conv1D(filters=128, kernel_size=win_len, padding='valid', activation='sigmoid')
    da = Dense(200, activation='relu')
    g = gate(embed_input)
    x = conv(embed_input)
    x = Multiply()([x, g])
    x = da(x)
    x = Dropout(0.2)(x)
    return crf(x)


def rnn_crf(embed_input, crf):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    x = ba(embed_input)
    x = Dropout(0.2)(x)
    return crf(x)
