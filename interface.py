import json

from recognize import predict

from util import load_pair, get_logger


path_zh_en = 'dict/zh_en.csv'
zh_en = load_pair(path_zh_en)

path_log_dir = 'log'
logger = get_logger('recognize', path_log_dir)


def insert(entity, label, entitys, slots):
    entitys.append(entity)
    slots.append(zh_en[label])


def make_dict(entitys, slots):
    slot_dict = dict()
    for slot, entity in zip(slots, entitys):
        if slot not in slot_dict:
            slot_dict[slot] = list()
        slot_dict[slot].append(entity)
    return slot_dict


def merge(pairs):
    entitys, slots = list(), list()
    entity, label = [''] * 2
    for word, pred in pairs:
        if pred[:2] == 'B-':
            if entity:
                insert(entity, label, entitys, slots)
            entity = word
            label = pred[2:]
        elif pred[:2] == 'I-' and entity:
            entity = entity + word
        elif entity:
            insert(entity, label, entitys, slots)
            entity = ''
    if entity:
        insert(entity, label, entitys, slots)
    return make_dict(entitys, slots)


def response(text, name):
    data = dict()
    pairs = predict(text, name)
    slot_dict = merge(pairs)
    data['content'] = text
    data['slot'] = slot_dict
    data_str = json.dumps(data, ensure_ascii=False)
    logger.info(data_str)
    return data_str


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print(response(text, 'cnn'))
        print(response(text, 'rnn'))
