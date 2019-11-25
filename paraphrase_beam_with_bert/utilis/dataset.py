# coding=utf-8
import logging
import os
import re
import time
import pickle

import torch
from torch.utils.data import Dataset
from .config import opt as opt

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer

        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_' + '_' + filename)

        if os.path.exists(cached_features_file):
            logger.info("Loading examples from cached file %s", cached_features_file)
            
            start = time.time()
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            end = time.time()
            print("Loading time %d s" % (end - start))
        else:
            logger.info("Creating cached file from examples at %s", directory)
            
            start = time.time()
            self.examples, self.lengths = [], []
            with open(file_path, "r", encoding="utf-8-sig") as reader: 
                sentences = reader.readlines()
                for sentence in sentences:
                    example = self._preprocess_text(sentence)
                    self.examples.append(example)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            end = time.time()
            print("Saving time %d s" % (end - start))

    def _preprocess_text(self, text):
        text = text.rstrip('\n').rstrip('\t')
        # POS_tags = re_POS.findall(text + ' ')
        # tokenized_POS_tags = tokenizer.convert_tokens_to_ids(POS_tags)
        text = re.compile(r"(?=/).*?(?= )").sub('', text + ' ').rstrip(' ')
        tokenized_text = self.tokenizer.build_inputs_with_special_tokens(
            self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(text)))
        return tokenized_text
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

    def collate_fn(self, data):
        """HACK: https://github.com/HLTCHKUST/PAML/blob/master/utils/data_reader.py."""
        
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.full((len(sequences), max(lengths)), opt.PAD_idx).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths 

        input_batch, input_lengths = merge(data)
        input_batch = input_batch.transpose(0, 1)
        input_lengths = torch.LongTensor(input_lengths)

        if opt.USE_CUDA:
            input_batch = input_batch.cuda(opt.gpu_device)
            input_lengths = input_lengths.cuda(opt.gpu_device)

        d = {}
        d["input_batch"] = input_batch
        d["input_lengths"] = input_lengths

        return d 

