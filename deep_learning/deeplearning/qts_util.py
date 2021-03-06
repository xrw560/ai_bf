# -*- conding:utf-8 -*-


VOCAB_SIZE = 8200


def chinese_to_index(text):
    """汉字->编码下标（汉字分类）"""
    result = []
    bs = text.encode('gb2312')  # 每个汉字占两个字节(区码(94),位码)
    num = len(bs)
    i = 0
    while i < num:
        b = bs[i]
        if b <= 160:
            result.append(b)
        else:
            b = b - 160
            if b >= 16:
                b -= 6  # 区号为10~15的区为空
            b -= 1
            i += 1
            b2 = bs[i] - 160 - 1
            result.append(161 + b * 94 + b2)  # 汉字区码从161开始
        i += 1
    return result


def index_to_chinese(index_list):
    """根据下标转换为汉字"""
    result = ''
    for index in index_list:
        if index < 161:
            result += chr(index)
        else:
            index -= 161
            block = int(index / 94) + 1
            if block >= 10:
                block += 6
            block += 160
            location = int(index % 94) + 1 + 160
            result += str(bytes([block, location]), encoding='gb2312')
    return result


def read_poems(path='qts_7X4.txt'):
    result = []
    error = 0
    with open(path, encoding='gb18030') as f:
        while True:
            poem = f.readline()
            if poem is None or poem == '':
                break
            # print('%d: ' % lines)
            # print(poem)
            try:
                segs = poem[:-1].split('.')
                segs = [seg[:-2] + '.' for seg in segs]
                poem = ''.join(segs)
                index = chinese_to_index(poem)
                result.append(index)
            except:  # 由于编码问题会出现一些错误，忽略这些诗
                # print(poem)
                error += 1
                continue
    print('Get %d poems, found %d errors.' % (len(result), error), flush=True)
    return result


if __name__ == '__main__':
    # a = '啊a中b.a quick brown dog jumps over a lazzy dog. 中华人民共和国！'
    # index = chinese_to_index(a)
    # print(index)
    # b = index_to_chinese(index)
    # print(b)
    # print(b==a)
    #
    # print(index_to_chinese([94*9+161]))

    samples = read_poems()
    print('=' * 200)
    for sample in samples:
        print(index_to_chinese(sample))
