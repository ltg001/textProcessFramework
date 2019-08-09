from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from sklearn import metrics
from utils import get_time_dif, load_data, word_w2v_model, char_w2v_model, TextDataset, replace
from tensorboardX import SummaryWriter
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = './save_dict/' + 'TextRNN' + '.ckpt'
class_list = [x.strip() for x in open('./data/class.txt').readlines()]


class TextRNN(nn.Module):
    def __init__(self, char_vocab_size=None):
        super(TextRNN, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2
        self.bidirectional = True
        self.char_vocab_size = char_vocab_size
        self.dropout = 0.5
        self.num_classes = 10

        self.BiLSTM = nn.LSTM(300, self.hidden_size, self.num_layers, bidirectional=self.bidirectional,
                              batch_first=True, dropout = self.dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)
        if self.char_vocab_size:
            self.embedding = nn.Embedding(char_vocab_size, 300, padding_idx=char_vocab_size + 1)

    def forward(self, x):
        x = x.type(torch.float32)
        if self.char_vocab_size:
            x = self.embedding(x)
        out, _ = self.BiLSTM(x)
        out = self.fc(out[:, -1, :])
        return out


class TextRCNN(nn.Module):
    def __init__(self):
        super(TextRCNN, self).__init__()
        self.dropout = 0.5
        self.num_classes = 10
        self.hidden_size = 256
        self.num_layers = 1
        self.bidirectional = True
        self.embedding_dim = 300

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers,
            bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)
        self.maxpool = nn.MaxPool1d(30)
        self.fc = nn.Linear(2 * self.hidden_size + self.embedding_dim, self.num_classes)

    def forward(self, x):
        x = x.type(torch.float32)
        out, _ = self.lstm(x)
        out = torch.cat((x, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        self.num_classes = 10

        self.conv = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, 300)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)

    def forward(self, x):
        x = x.type(torch.float32)
        x = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv], 1)
        # print(out.size)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def conv_and_pool(self, x, conv):
        x = x.type(torch.float32)
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def validation(model, dev_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            texts, labels = Variable(texts).to(device), Variable(labels).to(device)
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    # import ipdb; ipdb.set_trace()
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion
    return acc, loss_total / len(dev_iter)


def test(model, test_iter: DataLoader):
    model.load_state_dict(torch.load(save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = validation(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def train(train_iter, dev_iter, test_iter, model, epochs, learning_rate):
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    print(model.parameters)
    init_network(model)
    model.train()
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_batch = 0
    last_improve = 0
    writer = SummaryWriter(log_dir=time.strftime('%m-%d_%H.%M', time.localtime()))
    dev_best_loss = float('inf')
    flag = False
    for epoch in range(epochs):
        for train_batch, label_batch in train_iter:
            train_batch, label_batch = Variable(train_batch).to(device), Variable(label_batch).to(device)
            # import ipdb; ipdb.set_trace()
            outputs = model(train_batch)
            model.zero_grad()
            loss = F.cross_entropy(outputs, label_batch)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = label_batch.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = validation(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(model, test_iter)


if __name__ == "__main__":
    train_text, train_labels = load_data('./data/train.txt')
    dev_text, dev_labels = load_data('./data/dev.txt')
    test_text, test_labels = load_data('./data/test.txt')
    word_model, train_text = word_w2v_model(train_text, 30)
    # import ipdb; ipdb.set_trace()
    train_text = replace(train_text, word_model)
    dev_text = replace(dev_text, word_model)
    test_text = replace(test_text, word_model)
    # import ipdb; ipdb.set_trace()

    # print(f"len\ntrain: text {len(train_text)} label {len(train_labels)}\n {len(dev_text)} {len(dev_labels)}\n {len(test_text)} {len(test_labels)}")
    train_iter = DataLoader(TextDataset(train_text, train_labels, word_model),
        batch_size=64, shuffle=True)
    dev_iter = DataLoader(TextDataset(dev_text, dev_labels, word_model),
        batch_size=64, shuffle=True)
    test_iter = DataLoader(TextDataset(test_text, test_labels, word_model),
        batch_size=64, shuffle=True)

    model = TextCNN().to(device)
    train(train_iter, dev_iter, test_iter, model, 20, 1e-3)
