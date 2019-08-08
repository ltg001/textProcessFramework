# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import jieba
from gensim.models import Word2Vec
import re
from string import digits, punctuation


MAX_VOCAB_SIZE = 20000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
# torch.backends.cudnn.enabled = False
stopwords = set([line.strip() for line in open('./THUCNews/data/stop_words.txt', 'r', encoding='utf-8').readlines()])
rule = re.compile(r'[a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：=' + digits + punctuation + ']+')


def w2v_model(file_path, tokenizer, min_freq):
    with open(file_path, 'r', encoding='UTF-8') as f:
        contents, labels = [], []
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            labels.append(lin.split('\t')[1])
            content = re.sub(rule, '', content)
            content = [word for word in tokenizer(content) if word not in stopwords]
            contents.append(content)
    model = Word2Vec(contents, window=10, size=300, workers=4, min_count=min_freq)
    return model


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        contents = []
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
                contents.append(content)
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        re_vocab_dic = {}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
        model = Word2Vec(contents, window=5, size=300, workers=4)
    return vocab_dic, re_vocab_dic, model


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: list(jieba.cut(x))  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
        # vocab, re_vocab, re_vocab_model= build_vocab(config.train_path,
        #                                              tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=30)
    w2v_vec = w2v_model(config.train_path, tokenizer=tokenizer, min_freq=30)
        # pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(w2v_vec.wv.vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                content = re.sub(rule, '', content)
                words_line = []
                token = tokenizer(content)
                token = [item for item in token if item not in stopwords]
                seq_len = len(token)
                for word in token:
                    # import ipdb; ipdb.set_trace()
                    if word in w2v_vec.wv.vocab:
                        words_line.append(word)
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return w2v_vec, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, w2v_model):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.w2v_model = w2v_model

    def _to_tensor(self, datas):
        # import ipdb; ipdb.set_trace()
        datas = np.array(datas)
        sens = datas[:, 0]
        for index in range(sens.shape[0]):
            vec = []
            for item in sens[index]:
                vec.append(self.w2v_model.wv[item] if item in self.w2v_model.wv.vocab else np.array([0.0] * 300))
            if len(vec) > 30:
                vec = vec[:30]
            else:
                for _ in range(32 - len(vec)):
                    vec.append(np.array([0.0] * 300))
            # import ipdb; ipdb.set_trace()
            sens[index] = np.array(np.array(vec).tolist())
        datas[:, 0] = sens
        # import ipdb; ipdb.set_trace()
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # import ipdb; ipdb.set_trace()
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, w2v_model):
    iter = DatasetIterater(dataset, config.batch_size, config.device, w2v_model)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
