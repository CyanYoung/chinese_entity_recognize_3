import pickle as pk

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from keras_contrib.layers import CRF

from nn_arch import cnn_crf, rnn_crf

from util import map_item


batch_size = 32

path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
path_cnn_sent = 'feat/cnn_sent_train.pkl'
path_rnn_sent = 'feat/rnn_sent_train.pkl'
path_label = 'feat/label_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)
with open(path_cnn_sent, 'rb') as f:
    cnn_sents = pk.load(f)
with open(path_rnn_sent, 'rb') as f:
    rnn_sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

class_num = len(label_inds)

funcs = {'cnn_crf': cnn_crf,
         'rnn_crf': rnn_crf}

paths = {'cnn_crf': 'model/cnn_crf.h5',
         'rnn_crf': 'model/rnn_crf.h5',
         'cnn_crf_plot': 'model/plot/cnn_crf.png',
         'rnn_crf_plot': 'model/plot/rnn_crf.png'}


def compile(name, embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    crf = CRF(class_num)
    output = func(embed_input, crf)
    model = Model(input, output)
    model.summary()
    plot_model(model, map_item(name + '_plot', paths), show_shapes=True)
    model.compile(loss=crf.loss_function, optimizer=Adam(lr=0.001), metrics=[crf.accuracy])
    return model


def fit(name, epoch, embed_mat, class_num, sents, labels):
    seq_len = len(sents[0])
    model = compile(name, embed_mat, seq_len, class_num)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit(sents, labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    fit('cnn_crf', 10, embed_mat, class_num, cnn_sents, labels)
    fit('rnn_crf', 10, embed_mat, class_num, rnn_sents, labels)
