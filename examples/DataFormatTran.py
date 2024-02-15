import json
import random
from PIL import Image

with open('path.json', 'r', encoding='utf-8') as f:
    _path = json.load(f)


def random_sample_dict(dictionary, num_samples):
    keys = list(dictionary.keys())
    random.shuffle(keys)
    sampled_dict = {}
    for key in keys[:num_samples]:
        sampled_dict[key] = dictionary[key]
    return sampled_dict


def small_data_format(_type):
    res = {}
    with open(_path['small']['root'] + _path['small']['matching'][_type], 'r', encoding='utf-8') as f:
        matching_txt = f.readlines()

    with open(_path['small']['root'] + _path['small']['formulas'][_type], 'r', encoding='utf-8') as f:
        formulas_txt = f.readlines()

    for line in matching_txt:
        item = line.strip().split()
        _key, line_num = item[0], int(item[1])
        img_path = _path['small']['root'] + _path['small']['images'][_type] + _key
        img = Image.open(img_path)

        res[_key] = {}
        res[_key]['img_path'] = './data/small/' + _path['small']['images'][_type] + _key
        res[_key]['size'] = list(img.size)
        res[_key]['caption'] = formulas_txt[line_num]
        res[_key]['caption_len'] = len(res[_key]['caption'].split()) + 2

    with open('../data/small_data_format/' + _type + '.json', 'w', encoding='utf-8') as f:
        json.dump(res, f)

    print("已存入data/small_data_format/" + _type + '.json')


def full_data_format(_type):
    res = {}

    with open(_path['full']['root'] + _path['full']['matching'][_type], 'r', encoding='utf-8') as f:
        matching_txt = f.readlines()

    with open(_path['full']['root'] + _path['full']['formulas'][_type], 'r', encoding='utf-8') as f:
        formulas_txt = f.readlines()

    for line in matching_txt:
        item = line.strip().split()
        _key, line_num = item[0], int(item[1])
        _new_key = 'full_' + _key
        img_path = _path['full']['root'] + _path['full']['images'][_type] + _key
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            print(f"Failed to open image: {img_path}")
            continue
        print(img_path)

        res[_new_key] = {}
        res[_new_key]['img_path'] = './data/full/' + _path['full']['images'][_type] + _key
        res[_new_key]['size'] = list(img.size)
        res[_new_key]['caption'] = formulas_txt[line_num]
        res[_new_key]['caption_len'] = len(res[_new_key]['caption'].split()) + 2

    return res


def fullhand_data_format(_type):
    res = {}

    with open(_path['fullhand']['root'] + _path['fullhand']['matching'][_type], 'r', encoding='utf-8') as f:
        matching_txt = f.readlines()

    with open(_path['fullhand']['root'] + _path['fullhand']['formulas']['formulasTxt'], 'r', encoding='utf-8') as f:
        formulas_txt = f.readlines()

    for line in matching_txt:
        item = line.strip().split()
        _key, line_num = item[0], int(item[1])
        _new_key = 'fullhand_' + _key
        img_path = _path['fullhand']['root'] + _path['fullhand']['images'] + _key
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            print(f"Failed to open image: {img_path}")
            continue
        print(img_path)

        res[_new_key] = {}
        res[_new_key]['img_path'] = './data/fullhand/' + _path['fullhand']['images'] + _key
        res[_new_key]['size'] = list(img.size)
        res[_new_key]['caption'] = formulas_txt[line_num]
        res[_new_key]['caption_len'] = len(res[_new_key]['caption'].split()) + 2

    return res


def full_fullhand_combine():
    full_test_data_format = full_data_format("test")
    full_train_data_format = full_data_format("train")
    full_val_data_format = full_data_format("val")

    fullhand_test_data_format = fullhand_data_format("test")
    fullhand_train_data_format = fullhand_data_format("train")
    fullhand_val_data_format = fullhand_data_format("val")

    test_dt = {**full_test_data_format, **fullhand_test_data_format}
    train_dt = {**full_train_data_format, **fullhand_train_data_format}
    val_dt = {**full_val_data_format, **fullhand_val_data_format}

    with open("../data/E2L/test.json", 'w', encoding='utf-8') as f:
        json.dump(test_dt, f)
    print("test.json => E2L Finish!")
    with open("../data/E2L/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_dt, f)
    print("train.json => E2L Finish!")
    with open("../data/E2L/val.json", 'w', encoding='utf-8') as f:
        json.dump(val_dt, f)
    print("val.json => E2L Finish!")

    return train_dt, test_dt, val_dt


def full_fullhand_vocab():
    paths = ['../data/E2L/test.json', '../data/E2L/train.json', '../data/E2L/val.json']
    vocab_lt = []
    count = 1
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            dt = json.load(f)
        for key, item in dt.items():
            caption = item['caption']
            for word in caption.split():
                vocab_lt.append(word)
    vocab_set = set(vocab_lt)
    vocab_dt = {}
    for vocab in vocab_set:
        vocab_dt[vocab] = count
        count += 1
    vocab_dt["<unk>"] = count
    count += 1
    vocab_dt["<start>"] = count
    count += 1
    vocab_dt["<end>"] = count
    vocab_dt["<pad>"] = 0
    with open('../data/E2L/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dt, f)
    print("vocab.json => E2L Finish!")


if __name__ == '__main__':
    '''
    small文件夹的数据处理(结果存放在../data/small_data_format下)
    '''
    # small_data_format('test')
    # small_data_format('train')
    # small_data_format('val')

    """
    full 和 fullhand 文件夹的数据处理(结果存放在../data/E2L下)
    """
    train_dt, test_dt, val_dt = full_fullhand_combine()

    """
    统计full 和 fullhand 的 vocab
    """
    full_fullhand_vocab()

    """
    取出部分数据集
    """
    test_dt = random_sample_dict(test_dt, 250)
    train_dt = random_sample_dict(train_dt, 2000)
    val_dt = random_sample_dict(val_dt, 250)

    with open("../data/E2L/test.json", 'w', encoding='utf-8') as f:
        json.dump(test_dt, f)
    print("test.json => E2L Finish!")
    with open("../data/E2L/train.json", 'w', encoding='utf-8') as f:
        json.dump(train_dt, f)
    print("train.json => E2L Finish!")
    with open("../data/E2L/val.json", 'w', encoding='utf-8') as f:
        json.dump(val_dt, f)
    print("val.json => E2L Finish!")
