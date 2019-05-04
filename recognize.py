import pickle as pk

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding

from keras_contrib.layers import CRF

from keras.preprocessing.sequence import pad_sequences

from represent import add_buf

from nn_arch import cnn_crf, rnn_crf

from util import map_item


def define_model(name, embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    if name == 'cnn_crf':
        seq_len = seq_len + win_len - 1
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    crf = CRF(class_num)
    output = func(embed_input, crf)
    return Model(input, output)


def load_model(name, embed_mat, seq_len, class_num):
    model = define_model(name, embed_mat, seq_len, class_num)
    model.load_weights(map_item(name, paths), paths)
    return model


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


win_len = 7
seq_len = 50

path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

class_num = len(label_inds)

funcs = {'cnn_crf': cnn_crf,
         'rnn_crf': rnn_crf}

paths = {'cnn_crf': 'model/cnn_crf.h5',
         'rnn_crf': 'model/rnn_crf.h5'}

models = {'cnn_crf': load_model('cnn_crf', embed_mat, seq_len, class_num),
          'rnn_crf': load_model('rnn_crf', embed_mat, seq_len, class_num)}


def predict(text, name):
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    if name == 'cnn_crf':
        pad_seq = add_buf(pad_seq)
    model = map_item(name, models)
    probs = model.predict(pad_seq)[0]
    bound = min(len(text), seq_len)
    preds = np.argmax(probs, axis=1)[-bound:]
    if __name__ == '__main__':
        pairs = list()
        for word, pred in zip(text, preds):
            pairs.append((word, ind_labels[pred]))
        return pairs
    else:
        return preds


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('cnn_crf: %s' % predict(text, 'cnn_crf'))
        print('rnn_crf: %s' % predict(text, 'rnn_crf'))
