import math
import random
import time
import zipfile
import torch
from torch import nn


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)  # 32,312
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()  # long()  函数将数字或字符串转换为一个长整型。
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    # print(x.view(- 1, 1).shape)
    res.scatter_(1, x.view(- 1, 1), 1)
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None


    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        X = to_onehot(inputs, self.vocab_size)  # X 是个 list
        Y, self.state = self.rnn(torch.stack(X), state)
        output = self.dense(Y.view(- 1, Y.shape[- 1]))
        return output, self.state


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                        char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output 会记录 prefix 加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[- 1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)
        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                  corpus_indices, idx_to_char, char_to_idx,
                                  num_epochs, num_steps, lr, clipping_theta,
                                  batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            (output, state) = model(X, state)  # output:  形状为(num_steps * batch_size, vocab_size)
            y = torch.transpose(Y, 0, 1).contiguous().view(- 1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            #  梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

    try:
        perplexity = math.exp(l_sum / n)
    except OverflowError:
        perplexity = float('inf')
    if (epoch + 1) % pred_period == 0:
        print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
        for prefix in prefixes:
            print(' -', predict_rnn_pytorch(
            prefix, pred_len, model, vocab_size, device, idx_to_char,
            char_to_idx))
