import os

import json
import pandas as pd

import re

from random import shuffle, choice

from util import load_pair, load_poly


path_zh_en = 'dict/zh_en.csv'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
zh_en = load_pair(path_zh_en)
homo_dict = load_poly(path_homo)
syno_dict = load_poly(path_syno)


def save(path, sents):
    with open(path, 'w') as f:
        json.dump(sents, f, ensure_ascii=False, indent=4)


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


def select(part):
    if part[0] == '[' and part[-1] == ']':
        word = part[1:-1]
        cands = set()
        cands.add(word)
        if word in syno_dict:
            cands.update(syno_dict[word])
        if word in homo_dict:
            cands.update(homo_dict[word])
        return choice(list(cands))
    elif part[0] == '(' and part[-1] == ')':
        word = part[1:-1]
        return choice([word, ''])
    else:
        return part


def generate(temps, slots, num):
    word_mat, label_mat = list(), list()
    for i in range(num):
        parts = choice(temps)
        words, labels = list(), list()
        for part in parts:
            if part in slots:
                entity = choice(slots[part])
                words.extend(entity)
                labels.append('B-' + part)
                if len(entity) > 1:
                    labels.extend(['I-' + part] * (len(entity) - 1))
            else:
                word = select(part)
                words.extend(word)
                labels.extend(['O'] * len(word))
        word_mat.append(words)
        label_mat.append(labels)
    return word_mat, label_mat


def sync_shuffle(list1, list2):
    pairs = list(zip(list1, list2))
    shuffle(pairs)
    return zip(*pairs)


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
            entity_len = len(entity)
            for head in heads:
                tail = head + entity_len
                if slots[head:tail] != ['O'] * entity_len:
                    print('skip: %s in %s' % (entity, text))
                    continue
                slots[head] = 'B-' + label
                for i in range(1, entity_len):
                    slots[head + i] = 'I-' + label
        pairs = list()
        for word, label in zip(text, slots):
            pair = dict()
            pair['word'] = word
            pair['label'] = label
            pairs.append(pair)
        sents[text] = pairs
    return sents


def expand(word_mat, label_mat, extra_sents):
    word_mat, label_mat = sync_shuffle(word_mat, label_mat)
    bound = int(len(word_mat) * 0.9)
    train_sents = list2dict(word_mat[:bound], label_mat[:bound])
    train_sents.update(extra_sents)
    test_sents = list2dict(word_mat[bound:], label_mat[bound:])
    return train_sents, test_sents


def prepare(paths):
    temps = list()
    with open(paths['template'], 'r') as f:
        for line in f:
            parts = line.strip().split()
            temps.append(parts)
    slots = dict()
    files = os.listdir(paths['slot_dir'])
    for file in files:
        label = zh_en[os.path.splitext(file)[0]]
        slots[label] = list()
        with open(os.path.join(paths['slot_dir'], file), 'r') as f:
            for line in f:
                slots[label].append(line.strip())
    word_mat, label_mat = generate(temps, slots, num=1000)
    extra_sents = label_sent(paths['extra'])
    train_sents, test_sents = expand(word_mat, label_mat, extra_sents)
    save(paths['train'], train_sents)
    save(paths['test'], test_sents)


if __name__ == '__main__':
    paths = dict()
    paths['train'] = 'data/train.json'
    paths['test'] = 'data/test.json'
    paths['template'] = 'data/template.txt'
    paths['slot_dir'] = 'data/slot'
    paths['extra'] = 'data/extra.csv'
    prepare(paths)
