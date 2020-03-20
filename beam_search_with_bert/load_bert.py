import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import TensorDataset, Dataset, DataLoader, SequentialSampler

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from processor.glue_processors import glue_processors as processors  # refer: https://stackoverflow.com/questions/43147823/typeerror-module-object-is-not-subscriptable
from utilis.config import opt as opt

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, text, block_size=512):
        logger.info("Creating features from dataset file")
        # TODO: no need to truncate raw corpus for sentence input
        self.examples = []
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
        # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
        # If your dataset is small, first you should loook for a bigger one :-) and second you
        # can change this behavior by adding (model specific) padding.

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


class bert_model(object):

    def __init__(self):
        # Evaluating settings
        self.n_gpu = opt.n_gpu
        if opt.n_gpu > 1:
            # torch.distributed.init_process_group(backend="nccl")
            self.device_ids = opt.device_ids
        self.device = torch.device("cuda:"+str(opt.device_ids[0]) if torch.cuda.is_available() else "cpu")
        self.eval_batch_size = opt.eval_batch_size

        self.num_labels = 2
        self.max_seq_length = opt.max_seq_length
        self.processor = processors["mrpc"]()
        self.label_list = self.processor.get_labels()  
        self.block_size = opt.block_size    

        # Load a trained model that you have fine-tuned in LCQMC
        config_cls = BertConfig.from_pretrained(opt.cls_output_dir, num_labels=self.num_labels, finetuning_task="mrpc")
        self.tokenizer_cls = BertTokenizer.from_pretrained(opt.cls_output_dir, do_lower_case=False)
        self.bert_cls = BertForSequenceClassification.from_pretrained(opt.cls_output_dir, from_tf=False, config=config_cls)

		# Load a pre-training BERT
        config_mlm = BertConfig.from_pretrained(opt.mlm_output_dir)
        self.tokenizer_mlm = BertTokenizer.from_pretrained(opt.mlm_output_dir, do_lower_case=False)
        self.bert_mlm = BertForMaskedLM.from_pretrained(opt.mlm_output_dir, from_tf=False, config=config_mlm)
        
        if self.n_gpu > 1:
            self.bert_cls = DataParallel(self.bert_cls, device_ids=self.device_ids).cuda(self.device)
            self.bert_mlm = DataParallel(self.bert_mlm, device_ids=self.device_ids).cuda(self.device)
        else:
            self.bert_cls = self.bert_cls.to(self.device)
            self.bert_mlm = self.bert_mlm.to(self.device)

    def get_mlm_score(self, sents_beam):
        """
        :sents_beam: beam search candidates (string) which length is beam_size, shape like (n_active_inst, beam_size, len_dec_seq)
        """
        self.bert_mlm.eval()

        # Run prediction for input sentences
        eval_data = TextDataset(self.tokenizer_mlm, sents_beam, block_size=self.block_size)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)  # TODO: batch_size
        for batch in eval_dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                mlm_score = self.bert_mlm(batch)[0]

        return mlm_score

    def get_cls_score(self, sents, sents_beam):
        """
        :params sents: sentence (string) times beam_size
        :sents_beam: beam search candidates (string) which length is beam_size
        """
        eval_examples = self.processor.create_pair_batch(sents, sents_beam)
		
        logger.info("Creating features...")
        eval_features = convert_examples_to_features(
            eval_examples,
            self.tokenizer_cls,
            label_list=self.label_list,
            max_length=self.max_seq_length,
            output_mode=output_modes["mrpc"],
            pad_on_left=False,  # pad on the left for xlnet
            pad_token=self.tokenizer_cls.convert_tokens_to_ids([self.tokenizer_cls.pad_token])[0],
            pad_token_segment_id=0,
		)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", self.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)  # NOTE: fake labels
        eval_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        self.bert_cls.eval()
        for input_ids, attention_mask, token_type_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            # label_ids = label_ids.to(self.device)
            with torch.no_grad():
                logits = self.bert_cls(input_ids, attention_mask, token_type_ids)[0]  # NOTE: cls scores (not provided labels)
            logits = logits.detach().cpu().numpy()

        return logits  # TODO: batch logit??
