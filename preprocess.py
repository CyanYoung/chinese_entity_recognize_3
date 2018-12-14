import json
import pandas as pd

import re

from random import shuffle


def save(path, sents):
    with open(path, 'w') as f:
        json.dump(sents, f, ensure_ascii=False, indent=4)


def label_sent(path):
    sents = dict()
    for text, entity_str, label_str in pd.read_csv(path).values:
        entitys = entity_str.split()
        labels = label_str.split()
        if len(entitys) != len(labels):
            print('skip: %s', text)
            continue
        slots = ['O'] * len(text)
        for entity, label in zip(entitys, labels):
            heads = [iter.start() for iter in re.finditer(entity, text)]
            span = len(entity)
            for head in heads:
                slots[head] = 'B-' + label
                for i in range(1, span):
                    if slots[head + i] != 'O':
                        print('skip: %d of %s' % entity)
                        continue
                    slots[head + i] = 'I-' + label
        pairs = list()
        for word, label in zip(text, slots):
            pair = dict()
            pair['word'] = word
            pair['label'] = label
            pairs.append(pair)
        sents[text] = pairs
    return sents


def dict2list(sents):
    word_mat, label_mat = list(), list()
    for pairs in sents.values():
        words, labels = list(), list()
        for pair in pairs:
            words.append(pair['word'])
            labels.append(pair['label'])
        word_mat.append(words)
        label_mat.append(labels)
    return word_mat, label_mat


def list2dict(word_mat, label_mat):
    sents = dict()
    for words, labels in zip(word_mat, label_mat):
        text = ''.join(words)
        pairs = list()
        for word, label in zip(words, labels):
            pair = dict()
            pair['word'] = word
            pair['label'] = label
            pairs.append(pair)
        sents[text] = pairs
    return sents


def prepare(path_univ, path_train, path_test):
    sents = label_sent(path_univ)
    word_mat, label_mat = dict2list(sents)
    pairs = list(zip(word_mat, label_mat))
    shuffle(pairs)
    word_mat, label_mat = zip(*pairs)
    bound = int(len(sents) * 0.9)
    save(path_train, list2dict(word_mat[:bound], label_mat[:bound]))
    save(path_test, list2dict(word_mat[bound:], label_mat[bound:]))


if __name__ == '__main__':
    path_univ = 'data/univ.csv'
    path_train = 'data/train.json'
    path_test = 'data/test.json'
    prepare(path_univ, path_train, path_test)
