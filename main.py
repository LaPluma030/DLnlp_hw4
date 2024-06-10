import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_process import *
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras

callbacks_list = [
    keras.callbacks.ModelCheckpoint(  # 在每轮完成后保存权重
        filepath='text_gen.h5',
        monitor='loss',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(  # 不再改善时降低学习率
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    keras.callbacks.EarlyStopping(  # 不再改善时中断训练
        monitor='loss',
        patience=3,
    ),
]


class SeqToSeq(nn.Module):
    def __init__(self, len_token, embedding_size):
        super(SeqToSeq, self).__init__()
        self.encode = nn.Embedding(len_token, embedding_size)
        self.lstm = nn.LSTM(embedding_size, embedding_size, 2, batch_first=True)
        self.decode = nn.Sequential(
            nn.Linear(embedding_size, len_token),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(x.shape)
        em = self.encode(x).unsqueeze(dim=1)
        print(em.shape)
        mid, _ = self.lstm(em)
        print(mid[:,0,:].shape)
        res = self.decode(mid[:, 0, :])
        print(res.shape)
        return res


def sample(preds, temperature=1.0):
    """
    对模型得到的原始概率分布重新加权，并从中抽取一个 token 索引
    :param preds:预测的结果
    :param temperature:温度
    :return:重新加权后的最大值下标
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train(x, y, tokens, tokens_indices, epochs=200):
    x = np.asarray(x)
    y = np.asarray(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = models.Sequential([
        layers.Embedding(len(tokens), 256),
        layers.LSTM(256),
        layers.Dense(len(tokens), activation='softmax')
    ])

    optimizer = optimizers.RMSprop(lr=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    for e in range(epochs):

        model.fit(dataset, epochs=1, callbacks=callbacks_list)

    # text = '令狐冲这时已退到殿口，与教主的座位相距已遥，灯光又暗，远远望见去，任我行的容貌已颇为朦胧，心下忽想：“坐在这位子上的，是任我行还是东方不败，却有什么分别？”'
        text = '青衣剑士连劈三剑，锦衫剑士一一格开。青衣剑士一声吒喝，长剑从左上角直划而下，势劲力急。锦衫剑士身手矫捷，向后跃开，避过了这剑。他左足刚着地，身子跟着弹起，刷刷两剑，向对手攻去。青衣剑士凝里不动，嘴角边微微冷笑，长剑轻摆，挡开来剑。'
        print(text, end='')
        if e % 20 == 0:
            for temperature in [0.2, 0.5, 1.0, 1.2]:
                text_cut = list(jieba.cut(text))[:60]
                print('\n temperature: ', temperature)
                print(''.join(text_cut), end='')
                for i in range(100):

                    sampled = np.zeros((1, 60))
                    for idx, token in enumerate(text_cut):
                        if token in tokens_indices:
                            sampled[0, idx] = tokens_indices[token]
                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature=1)
                    next_token = tokens[next_index]
                    print(next_token, end='')

                    text_cut = text_cut[1: 60] + [next_token]


if __name__ == '__main__':
    file = DATA_PATH + '越女剑.txt'
    d = get_single_corpus(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)
    train(_x, _y, _tokens, _tokens_indices)