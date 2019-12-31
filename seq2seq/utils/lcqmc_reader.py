import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re, math
import random
from random import randint
from collections import Counter
from random import shuffle
import pprint
pp = pprint.PrettyPrinter(indent=1)
import torch
import torch.utils.data as data
# from torch.autograd import Variable
from utils import config
import pickle
import os


def DistJaccard(str1, str2):
    str1 = set(str1.split())
    str2 = set(str2.split())
    return float(len(str1 & str2)) / len(str1 | str2)

def dist_matrix(array_str):
    matrix = []
    for i,s_r in enumerate(array_str):
        row = []
        for j,s_c in enumerate(array_str):
            # row.append(get_cosine(text_to_vector(s_r),text_to_vector(s_c)))
            row.append(DistJaccard(s_r,s_c))
        matrix.append(row)
    mat_ = np.array(matrix)
    print("Mean",np.mean(mat_))
    print("Var", np.var(mat_))
    return matrix

def plot_mat(mat):
    ax = sns.heatmap(mat, cmap="YlGnBu")
    # g = sns.clustermap(mat,cmap="YlGnBu", figsize=(8,8))
    plt.show()

def create_str_array(data):
    arr = []
    for k,v in data.items():
        arr.append(" ".join(v[0][0]))
    return arr

def show_example(mat_jac,arr, a,b):
    print("Example with {}<= VAL < {}".format(a,b))
    for i in range(len(mat_jac)):
        for j in range(len(mat_jac)):  
            if(i>j and float(mat_jac[i][j])>=a and float(mat_jac[i][j])<b):
                print("Dial 1\n",arr[i])  
                print("Dial 2\n",arr[j])    
                return 

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS"} 
        self.n_words = 4 # Count default tokens
      
    def index_words(self, sentence):
        # NOTE: chinese character
        for word in sentence:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.src = []
        self.trg = [] 
        self.max_len_words = 0
        
        for d in data["input_txt"]:
            if(len(list(d)) > self.max_len_words): self.max_len_words = len(list(d))           
            self.src.append(d)    

        for d in data["target_txt"]:
            if(len(list(d)) > self.max_len_words): self.max_len_words = len(list(d))      
            self.trg.append(d)

        self.vocab = vocab
        self.num_total_seqs = len(data["input_txt"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["input_txt"] = self.src[index]
        item["target_txt"] = self.trg[index]
        item["input_batch"] = self.preprocess(self.src[index]) 
        item["target_batch"] = self.preprocess(self.trg[index], anw=True)
        
        if config.pointer_gen:
            item["input_ext_vocab_batch"], item["article_oovs"] = self.process_input(item["input_txt"])
            item["target_ext_vocab_batch"] = self.process_target(item["target_txt"], item["article_oovs"])
        return item

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, arr, anw=False):
        """Converts words to ids."""
        if(anw):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]  # NOTE: <EOS>
        else:
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr]
        return torch.LongTensor(sequence)


    ## for know I ignore unk
    def process_input(self, input_txt):
        seq = []
        oovs = []
        for word in list(input_txt.strip()):
            if word in self.vocab.word2index:
                seq.append(self.vocab.word2index[word])
            else:
                seq.append(config.UNK_idx)
            # else:
            #     if word not in oovs:
            #         oovs.append(word)
            #     seq.append(self.vocab.n_words + oovs.index(word))
        
        seq = torch.LongTensor(seq)
        return seq, oovs

    def process_target(self, target_txt, oovs):
        # seq = [self.word2index[word] if word in self.word2index and self.word2index[word] < self.output_vocab_size else UNK_idx for word in input_txt.strip().split()] + [EOS_idx]
        seq = []
        for word in list(target_txt.strip()):
            if word in self.vocab.word2index:
                seq.append(self.vocab.word2index[word])
            elif word in oovs:
                seq.append(self.vocab.n_words + oovs.index(word))
            else:
                seq.append(config.UNK_idx)
        seq.append(config.EOS_idx)
        seq = torch.LongTensor(seq)
        return seq

def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    data.sort(key=lambda x: len(x["input_batch"]), reverse=True) ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    input_batch, input_lengths = merge(item_info['input_batch'])
    target_batch, target_lengths = merge(item_info['target_batch'])

    input_batch = input_batch.transpose(0, 1)
    target_batch = target_batch.transpose(0, 1)
    input_lengths = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor(target_lengths)

    if config.USE_CUDA:
        input_batch = input_batch.cuda()
        target_batch = target_batch.cuda()
        input_lengths = input_lengths.cuda()
        target_lengths = target_lengths.cuda()

    d = {}
    d["input_batch"] = input_batch
    d["target_batch"] = target_batch
    d["input_lengths"] = input_lengths
    d["target_lengths"] = target_lengths
    d["input_txt"] = item_info["input_txt"]
    d["target_txt"] = item_info["target_txt"]

    if 'input_ext_vocab_batch' in item_info:
        input_ext_vocab_batch, _ = merge(item_info['input_ext_vocab_batch'])
        target_ext_vocab_batch, _ = merge(item_info['target_ext_vocab_batch'])
        input_ext_vocab_batch = input_ext_vocab_batch.transpose(0, 1)
        target_ext_vocab_batch = target_ext_vocab_batch.transpose(0, 1)
        if config.USE_CUDA:
            input_ext_vocab_batch = input_ext_vocab_batch.cuda()
            target_ext_vocab_batch = target_ext_vocab_batch.cuda()
        d["input_ext_vocab_batch"] = input_ext_vocab_batch
        d["target_ext_vocab_batch"] = target_ext_vocab_batch
        if "article_oovs" in item_info:
            d["article_oovs"] = item_info["article_oovs"]
            d["max_art_oovs"] = max(len(art_oovs) for art_oovs in item_info["article_oovs"])
    return d 


def read_langs(file_name):
    print(("Reading lines from {}".format(file_name)))
    # Read the file and split into lines
    data = {"input_txt": [], "target_txt": []}
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            if config.read_renmin:
                sent = line.rstrip('\n').rstrip('\t')
                sent = re.compile(r"(?=/).*?(?= )").sub('', sent + ' ').rstrip(' ')
                sent = "".join(sent.split())
                data["input_txt"].append(sent)
                data["target_txt"].append("测试样例")
            else:
                line = line.strip()
                sent1, sent2, label= line.split('\t')          
                if label == '0':
                    data["input_txt"].append(sent1)
                    data["target_txt"].append(sent2)
    return data


def prepare_data_seq():
    file_train = '../corpus/LCQMC/train.txt'
    file_dev = '../corpus/LCQMC/dev.txt'
    if config.read_renmin:
        file_test = '../paraphrase_beam_with_bert/data/2014_renmin_news_sentences_filter.txt'
    else:
        file_test = '../corpus/LCQMC/test.txt'
    train = read_langs(file_train)
    valid = read_langs(file_dev)
    test = read_langs(file_test)
    vocab = Lang()
    
    train = preprocess(train, vocab)
    valid = preprocess(valid, vocab)
    test = preprocess(test, vocab)
    print("Vocab_size %s " % vocab.n_words)

    # for read renmin
    if not os.path.exists(config.save_path_dataset):
        os.makedirs(config.save_path_dataset)
    with open(config.save_path_dataset+'vocab.p', "wb") as f:
        pickle.dump(vocab, f)
        print("Saved vocab")

    if config.read_renmin:
        if(os.path.exists(config.save_path_dataset+'vocab.p')):
            with open(config.save_path_dataset+'vocab.p', "rb") as f:
                vocab = pickle.load(f)

    with open(config.save_path_dataset+'dataset.p', "wb") as f:
        pickle.dump([train,valid,test,vocab], f)
        print("Saved PICKLE")
    
    return train, valid, test, vocab


def preprocess(data, vocab):
    for sent in data["input_txt"]:
        vocab.index_words(sent)
    for sent in data["target_txt"]:
        vocab.index_words(sent)
    return data


class Personas:
    def __init__(self):
        random.seed(999)
        if(os.path.exists(config.save_path_dataset+'dataset.p')):
            with open(config.save_path_dataset+'dataset.p', "rb") as f:
                [self.meta_train, self.meta_valid, self.meta_test, self.vocab] = pickle.load(f)
            self.type = {'train': self.meta_train,
                        'valid': self.meta_valid,
                        'test': self.meta_test}
            print("DATASET LOADED FROM PICKLE")
        else:
            self.meta_train, self.meta_valid, self.meta_test, self.vocab = prepare_data_seq()
            self.type = {'train': self.meta_train,
                        'valid': self.meta_valid,
                        'test': self.meta_test} 
    
    def get_all_data(self, batch_size):
        dataset_train = Dataset(self.meta_train, self.vocab)
        data_loader_tr = torch.utils.data.DataLoader(dataset=dataset_train,
                                                batch_size=batch_size,
                                                shuffle=True, collate_fn=collate_fn)

        dataset_val = Dataset(self.meta_valid, self.vocab)
        data_loader_val = torch.utils.data.DataLoader(dataset=dataset_val,
                                                batch_size=batch_size,
                                                shuffle=False,collate_fn=collate_fn)  

        dataset_test = Dataset(self.meta_test, self.vocab)
        data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size,
                                                shuffle=False,collate_fn=collate_fn)           
        return data_loader_tr, data_loader_val, data_loader_test


if __name__=='__main__':
    train, valid, test, vocab = prepare_data_seq()
