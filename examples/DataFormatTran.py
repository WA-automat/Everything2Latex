import json
from PIL import Image

with open('path.json', 'r', encoding='utf-8') as f:
    _path = json.load(f)


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

    with open('../data/temp/' + _type + '.json', 'w', encoding='utf-8') as f:
        json.dump(res, f)

    print("已存入data/temp/" + _type + '.json')


if __name__ == '__main__':
    '''
    small文件夹的数据处理(结果存放在../data/temp下)
    '''
    small_data_format('test')
    small_data_format('train')
    small_data_format('val')
