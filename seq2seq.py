import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# 小说段落（输入数据）
# f = open("corpus_sentence.txt", encoding="utf-8")
f = open("D:\postgraduate\DeepNLP\DLnlp_hw4\jyxstxtqj_downcc.com\三十三剑客图.txt", encoding='gb18030')
source_texts = f.read()
# print(corpus_chars)
source_texts = source_texts.replace('\n', ' ').replace('\r', ' ')
source_texts = source_texts[0:10000]
# 创建词汇表
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        for char in text:
            counter[char] += 1
    vocab = {char: idx for idx, (char, _) in enumerate(counter.items(), start=4)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    vocab["<SOS>"] = 2
    vocab["<EOS>"] = 3
    return vocab

source_vocab = build_vocab(source_texts)

# 文本转换为索引序列
def text_to_indices(text, vocab):
    return [vocab.get(char, vocab["<UNK>"]) for char in text]

source_indices = [text_to_indices(text, source_vocab) for text in source_texts]

# 数据集定义
class TextDataset(Dataset):
    def __init__(self, source_data):
        self.source_data = source_data

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index):
        src = self.source_data[index]
        src_tensor = torch.tensor([source_vocab["<SOS>"]] + src + [source_vocab["<EOS>"]])
        return src_tensor

# 数据加载
dataset = TextDataset(source_indices)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# 模型参数
INPUT_DIM = len(source_vocab)
OUTPUT_DIM = len(source_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# 模型初始化
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
else:
    print("CUDA is not available. Using CPU instead.")
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(enc, dec, device).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=source_vocab["<PAD>"])


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, src in enumerate(iterator):
        src = src.to(device)
        trg = src

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, src in enumerate(iterator):
            src = src.to(device)
            trg = src

            output = model(src, trg, 0)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, dataloader, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')


def generate_text(model, start_text, source_vocab, max_len=200):
    model.eval()
    tokens = [source_vocab["<SOS>"]] + [source_vocab.get(char, source_vocab["<UNK>"]) for char in start_text] + [source_vocab["<EOS>"]]
    src_tensor = torch.tensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indices = [source_vocab["<SOS>"]]
    for _ in range(max_len):
        trg_tensor = torch.tensor([trg_indices[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            top1 = output.argmax(1).item()
            trg_indices.append(top1)
            if top1 == source_vocab["<EOS>"]:
                break

    trg_tokens = [list(source_vocab.keys())[list(source_vocab.values()).index(i)] for i in trg_indices]
    return ''.join(trg_tokens[1:-1])

# 示例生成
start_text = "令狐冲习得了九阳神功，并且学会了乾坤大挪移"
generated_text = generate_text(model, start_text, source_vocab)
print("Generated text:", generated_text)
