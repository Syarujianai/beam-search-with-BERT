# encoding=utf-8
# refer: https://blog.csdn.net/xieyan0811/article/details/62056558
import re
import sys
import json
import codecs
import traceback

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

import math
import numpy as np
# import threading
# from threading import Thread
import multiprocessing
from time import sleep
from tqdm import trange, tqdm


# input_file = "./2014_renmin_news_sentences_filter.txt"
vocab_file = "./vocab_senti.json"
senti_filter_set = {'/vd','/vf', '/vg', '/vl', '/vn', '/vshi', '/vx', '/vyou', '/dl', '/dg', '/an'}
senti_set = {'v', 'd', 'a'}
# lock = threading.Lock()

def update_senti_dict(words, part_of_speechs, word_net, 
                      senti_word_dict, senti_info, 
                      update_senti_info=True):
    
    global senti_filter_set, senti_set

    count = 0
    p, n = 0., 0.
    for i in range(len(part_of_speechs)):
        # get sentiment score
        word_net_ids = search_WordNet(word_net, words[i])
        if (len(word_net_ids) == 0):
            continue
        pos_score, neg_socre = [], []
        for wid in word_net_ids:

            # lock.acquire()
            # try:
            syn_set = id2ss(wid, words[i])
            if syn_set is not None:
                swninfo = get_senti(syn_set)
            else:
                continue
            # finally:
            #     lock.release()

            pos_score.append(swninfo.pos_score())
            neg_socre.append(swninfo.neg_score())
        if len(pos_score) == 0:
            continue
        pos_score = np.mean(pos_score)
        neg_score = np.mean(neg_socre)

        # update sentiment word dictionary
        if part_of_speechs[i][1] in senti_set and part_of_speechs[i] not in senti_filter_set:
            if pos_score > 0.05 and pos_score > neg_score:
                tag_w = "pos"
            elif neg_score > 0.05 and neg_score > pos_score:
                tag_w = "neg"
            elif neg_score < 0.05 and neg_score < 0.05:
                tag_w = "neu"
            else:
                continue

            # lock.acquire()
            # try:
            if words[i] not in senti_word_dict[tag_w][part_of_speechs[i][1]].keys():
                print(words[i], '-> p ', pos_score, ", n ", neg_score, words, '\n')
                senti_word_dict[tag_w][part_of_speechs[i][1]][words[i]] = 1
            else:
                senti_word_dict[tag_w][part_of_speechs[i][1]][words[i]] += 1         
            # finally:
            #     lock.release()

            count += 1
            p += pos_score
            n += neg_score
    
    # vote for update sentiment info
    if count != 0:
        p = p / count
        n = n / count
    if update_senti_info:
        if p > n:
            tag_s = "pos"
        elif n > p:
            tag_s = "neg"
        else:
            tag_s = "neu"

        # lock.acquire()
        # try:
        senti_info[tag_s] += 1
        # finally:
        #     lock.release()

    return p, n

def write_senti_dict(senti_word_dict, stat_io):
    json.dump(senti_word_dict, stat_io, ensure_ascii=False, indent=4)
    stat_io.close()

def get_words_and_POS_from_sentence(sentence):
    # matching functional POS pattern that between '/' and ' ' 
    reg_POS = re.compile(r"(?=/).*?(?= )")
    POS = reg_POS.findall(sentence.lstrip('/w')+' ')  # NOTE: for matching pattern of last POS tag
    words = reg_POS.sub('', sentence + ' ').rstrip(' ').split()
    return words, POS

def do_segment(sentence, stop_words):   
    sentence = sentence.rstrip('\n').rstrip('\t')
    words_list, POS_list = get_words_and_POS_from_sentence(sentence)
    if len(words_list) != len(POS_list):
        return None, None
    else:
        words_list_filter, POS_list_filter = [], []
        for i, word in enumerate(words_list):
            if word.encode("utf-8") not in stop_words and word != '':
                words_list_filter.append(word)
                POS_list_filter.append(POS_list[i])
        return words_list_filter, POS_list_filter

def load_stop_words():
    "load stop words dictionary"
    stop_words = []  
    for word in open("./WordNet/stop_words.txt", "r"):  
        stop_words.append(word.strip())
    return stop_words

def load_WordNet():
    f = codecs.open("./WordNet/cow-not-full.txt", "rb", "utf-8")
    known = set()
    for l in f:
        if l.startswith('#') or not l.strip():
            continue
        row = l.strip().split("\t")
        if len(row) == 3:
            (synset, lemma, status) = row 
        elif len(row) == 2:
            (synset, lemma) = row 
            status = 'Y'
        else:
            print("illformed line: ", l.strip())
        if status in ['Y', 'O' ]:
            if not (synset.strip(), lemma.strip()) in known:
                known.add((synset.strip(), lemma.strip()))
    return known
 
def search_WordNet(known, key):
    ll = []
    for kk in known:
        if (kk[1] == key):
             ll.append(kk[0])
    return ll
 
def id2ss(ID, word):
    try:
        syn_set = wn.synset_from_pos_and_offset(str(ID[-1:]), int(ID[:8]))
    except Exception as e:
        traceback.print_exc()
        print(word)
        return None
    return syn_set
 
def get_senti(word):
    return swn.senti_synset(word.name())
 
def run_process(n, iter, sentences, stop_words, word_net, senti_word_dict, senti_info):
    for i in tqdm(range(iter[0], iter[1]), ncols=80):
    # for i in range(iter[0], iter[1]):
        words, POSs = do_segment(sentences[i], stop_words)
        if words is None:
            continue
        pos_score_sent, neg_score_sent = \
            update_senti_dict(words, POSs, word_net, senti_word_dict, senti_info)
        # progresser(n)

def progresser(n):
    "refer: https://github.com/tqdm/tqdm#nested-progress-bars"
    interval = 0.001 / (n + 2)
    total = 5000
    text = "#{}, est. {:<04.2}s".format(n, interval * total)
    for _ in trange(total, desc=text, position=n):
        sleep(interval)

if __name__ == '__main__' :
    word_net = load_WordNet()
    # read file
    with open(sys.argv[1], 'r', encoding='utf-8') as reader:
        sentences = reader.readlines()
    stop_words = load_stop_words()

    # preprocess and statistics
    num_process = 16
    step = len(sentences) / (num_process*10)
    iter_list = [(math.ceil(i*step), math.floor((i+1)*step)) for i in range(num_process)]
    multiprocessing.freeze_support()  # for Windows support
    p = multiprocessing.Pool()
    with multiprocessing.Manager() as mg:
        senti_word_dict = mg.dict({
            "pos": {'v': {}, 'd': {}, 'a': {}}, 
            "neg": {'v': {}, 'd': {}, 'a': {}},
            "neu": {'v': {}, 'd': {}, 'a': {}},
        })
        senti_info = mg.dict({
            "pos": 0, 
            "neg": 0,
            "neu": 0,
        })

        # statistics (parallelize)
        for n, iter in enumerate(iter_list):
            # refer: https://blog.csdn.net/jackliu16/article/details/82598298
            p.apply_async(run_process, args=(n, iter, sentences, 
                           stop_words, word_net, senti_word_dict, senti_info))    
        p.close()
        p.join()
        
        # print statistics
        for k, v in dict(senti_info).items():
            print(k, '-> ', v, '\n')
    
        # write out sentiment word vocab
        stat_io = open(vocab_file, 'w', encoding='utf-8')
        write_senti_dict(dict(senti_word_dict), stat_io)
    

