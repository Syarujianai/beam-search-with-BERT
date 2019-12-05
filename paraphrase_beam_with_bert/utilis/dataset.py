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
        self.device = torch.device("cuda:"+str(opt.device_ids[0]) if torch.cuda.is_available() else "cpu")

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
            self.examples = []
            with open(file_path, "r", encoding="utf-8-sig") as reader: 
                sentences = reader.readlines()
                for sentence in sentences:
                    example = {}
                    text, temp_ids, ids = self._preprocess_text(sentence)
                    example["text"]= text
                    example["temp_ids"]= temp_ids
                    example["ids"]= ids
                    self.examples.append(example)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
            end = time.time()
            print("Saving time %d s" % (end - start))

    def _preprocess_text(self, text_orig):
        text_orig = text_orig.rstrip('\n').rstrip('\t')
        # POS_tags = re_POS.findall(text_orig + ' ')
        # tokenized_POS_tags = tokenizer.convert_tokens_to_ids(POS_tags)
        text_orig = re.compile(r"(?=/).*?(?= )").sub('', text_orig + ' ').rstrip(' ')
        text = "".join(text_orig.split())
        temp_ids = self.tokenizer.build_inputs_with_special_tokens(
                        self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer._tokenize(text_orig, do_templatize_func_word=True)))
        
        ids = self.tokenizer.build_inputs_with_special_tokens(
                        self.tokenizer.convert_tokens_to_ids(
                            self.tokenizer._tokenize(text_orig)))
        
        return text, temp_ids, ids
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return {
            "text":self.examples[item]["text"],
            "temp_ids":torch.tensor(self.examples[item]["temp_ids"]),
            "ids":torch.tensor(self.examples[item]["ids"])}

    def collate_fn(self, data):
        """HACK: https://github.com/HLTCHKUST/PAML/blob/master/utils/data_reader.py."""
        
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.full((len(sequences), max(lengths)), opt.PAD_idx).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths 
 
        seq_batch = [d["ids"] for d in data]
        seq_temp_batch = [d["temp_ids"] for d in data]
        text_batch = [d["text"] for d in data]
        
        input_batch, _ = merge(seq_batch)
        input_temp_batch, input_lengths = merge(seq_temp_batch)
        input_batch = input_batch.transpose(0, 1)
        input_temp_batch = input_temp_batch.transpose(0, 1)
        input_lengths = torch.LongTensor(input_lengths)

        if opt.USE_CUDA:
            input_batch = input_batch.cuda(self.device)
            input_temp_batch = input_temp_batch.cuda(self.device)
            input_lengths = input_lengths.cuda(self.device)

        d = {}
        d["input_text"] = text_batch
        d["input_batch"] = input_batch
        d["input_temp_batch"] = input_temp_batch
        d["input_lengths"] = input_lengths

        return d 

