from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
import jieba
import re
import time
from datetime import timedelta
from string import digits, punctuation


stopwords = set([line.strip() for line in open('./data/stop_words.txt', 'r',
                                               encoding='utf-8').readlines()])
rule = re.compile(r'[a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：=' +
                  digits + punctuation + ']+')


def word_tokenizer(sentence):
    return list(jieba.cut(sentence))


def char_tokenizer(sentence):
    return [char for char in sentence]


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_data(file_path):
    contents, labels = [], []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            content = re.sub(rule, '', content)
            labels.append(int(lin.split('\t')[1]))
            contents.append(content)
            # import ipdb; ipdb.set_trace()
    return contents, labels


def word_w2v_model(text, word_min_freq):
    contents = []
    for sen in text:
        if type(sen) == type([]):
            sen = ''.join(sen)
        sen = word_tokenizer(sen)
        sen = [word for word in sen if word not in stopwords]
        contents.append(sen)
    model = Word2Vec(contents, size=300, workers=4, window=10, min_count=word_min_freq)
    print('word vocab size: ', len(model.wv.vocab))
    return model, contents


def char_w2v_model(text, char_min_freq):
    contents = []
    for sen in text:
        if type(sen) == type([]):
            sen = ''.join(sen)
        sen = char_tokenizer(sen)
        sen = [char for char in sen if char not in stopwords]
        contents.append(sen)
    model = Word2Vec(contents, size=300, workers=4, window=10, min_count=char_min_freq)
    print('char vocab size: ', len(model.wv.vocab))
    return model, contents


def replace(text, model: Word2Vec):
    contents = []
    for sen in text:
        # import ipdb; ipdb.set_trace()
        contents.append([item for item in sen if item in model.wv.vocab])
    return contents


class TextDataset(Dataset):
    def __init__(self, text, label, w2v_model: Word2Vec):
        self.text = text
        self.label = label
        self.model = w2v_model
        assert len(self.text) == len(self.label)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ans = []
        for item in self.text[index]:
            if item in self.model.wv.vocab:
                # import ipdb; ipdb.set_trace()
                ans.append(self.model.wv[item])
            else:
                ans.append([0.0] * 300)
        
        if len(ans) > 30:
            ans = ans[:30]
        elif len(ans) < 30:
            length = 30 - len(ans)
            zeros = np.zeros(300)
            for i in range(length):
                ans.append(zeros)
        ans = np.array(ans)
        # import ipdb; ipdb.set_trace()
        return np.array(ans.tolist()), int(self.label[index])
