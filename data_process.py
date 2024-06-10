import random
import re
import jieba
import numpy as np

DATA_PATH = './jyxstxtqj_downcc.com/'


def get_single_corpus(file_path):
    """
    获取file_path文件对应的内容
    :return: file_path文件处理结果
    """
    corpus = ''
    # unuseful items filter
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@★、…【】《》‘’[\\]^_`{|}~「」『』（）]+'
    # with open('../stopwords.txt', 'r', encoding='utf8') as f:
    #     stop_words = [word.strip('\n') for word in f.readlines()]
    #     f.close()
    # print(stop_words)
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(r1, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
        f.close()
    # corpus.replace('。', ' 。')
    # corpus.replace('，', ' ，')
    words = list(jieba.cut(corpus))
    print("Corpus length: {}".format(len(words)))
    return words
    # return [word for word in words if word not in stop_words]


def get_dataset(data):
    """
    :param data: 分词结果
    :return: 落库，段落对应的下一个词，词库和词库索引
    """
    max_len = 60
    step = 3
    sentences = []
    next_tokens = []

    # for i in range(0, len(data)-max_len, step):
    #     sentences.append(data[i: i+max_len])
    #     next_tokens.append(data[i+max_len])
    # print('Number of sequences: {}'.format(len(sentences)))

    tokens = list(set(data))
    tokens_indices = {token: tokens.index(token) for token in tokens}
    print('Unique tokens:', len(tokens))

    for i in range(0, len(data) - max_len, step):
        sentences.append(
            list(map(lambda t: tokens_indices[t], data[i: i + max_len])))
        next_tokens.append(tokens_indices[data[i + max_len]])
    print('Number of sequences:', len(sentences))

    print('Vectorization...')
    next_tokens_one_hot = []
    for i in next_tokens:
        y = np.zeros((len(tokens),))
        y[i] = 1
        next_tokens_one_hot.append(y)
    # print(sentences[0], next_tokens_one_hot[0])
    # print(len(sentences), len(next_tokens_one_hot))
    return sentences, next_tokens_one_hot, tokens, tokens_indices


if __name__ == '__main__':
    file = DATA_PATH + '笑傲江湖.txt'
    d = get_single_corpus(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)