import matplotlib
matplotlib.use('Agg')
from utils.lcqmc_reader import Personas
from model.transformer import Transformer
import pickle
from utils import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
from utils.beam_omt import Translator
import os
import time
import numpy as np 
from random import shuffle
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import math
from model.common_layer import evaluate

def generate(model, data):
    t = Translator(model, model.vocab)
    for j, batch in enumerate(data):
        _, _, _ = model.train_one_batch(batch, train=False)
        sent_b, _ = t.translate_batch(batch)
        for i in range(len(batch["target_txt"])):
            new_words = []
            for w in sent_b[i][0]:
                if w==config.EOS_idx:
                    break
                new_words.append(w)
                if len(new_words)>2 and (new_words[-2]==w):
                    new_words.pop()
            sent_beam_search = ' '.join([model.vocab.index2word[idx] for idx in new_words])
            print("----------------------------------------------------------------------")
            print("----------------------------------------------------------------------")
            print("dialogue context:")
            print(pp.pformat(batch['input_txt'][i]))
            print("Beam: {}".format(sent_beam_search))
            print("Ref:{}".format(batch["target_txt"][i]))
            print("----------------------------------------------------------------------")
            print("----------------------------------------------------------------------")


p = Personas()

# Build model, and dataloader
print("Test model",config.model)
model = Transformer(p.vocab, model_file_path=config.save_path, is_eval=False)
train_iter, val_iter, test_iter = p.get_all_data(batch_size=config.batch_size)

#evaluate
loss, ppl_val, bleu_score_b = evaluate(
    model, test_iter, model_name=config.model,ty='test',verbose=True)

#generate
generate(model, val_iter)
