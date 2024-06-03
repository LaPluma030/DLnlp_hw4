import sys
sys.path.append("C:\Users\Administrator\Desktop\课程\深度学习与自然语言处理\第四次作业")
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import utils
import jieba

import sys

sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""加载数据集"""
f = open("corpus_sentence.txt", encoding="utf-8")
corpus_chars = f.read()
# print(corpus_chars)
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
# print(corpus_chars)

corpus_chars = corpus_chars[0:10000]
corpus_chars = jieba.lcut(corpus_chars)
idx_to_char = list(set(corpus_chars))
#      print(idx_to_char)
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
# print(char_to_idx)
vocab_size = len(char_to_idx)
print("vocab_size=", vocab_size)
corpus_indices = [char_to_idx[char] for char in corpus_chars]
# print(corpus_indices)
# # return corpus_indices, char_to_idx, idx_to_char, vocab_size

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

num_epochs, num_steps, batch_size, lr, clipping_theta, num_layers = 50, 35, 32, 1e2, 1e-2, 3
pred_period, pred_len, prefixes = 50, 70, ['他跑了过去', '她哭了起来']

lr = 1e-2  # 注意调整学习率
lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens, num_layers=num_layers)
model = utils.RNNModel(lstm_layer, vocab_size)
utils.train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                    corpus_indices, idx_to_char, char_to_idx,
                                    num_epochs, num_steps, lr, clipping_theta,
                                    batch_size, pred_period, pred_len, prefixes)
