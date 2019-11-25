import argparse
import json
import random
import prettytable as pt
from datetime import datetime

import torch
import numpy as np

def parse_opt():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="data/",
                        type=str,
                        required=False,
                        help="The input dataset directory.")
    parser.add_argument("--vocab_file_path",
                        default="data/vocab.txt",
                        type=str,
                        required=False,
                        help="The vocab file path.")
    parser.add_argument("--func_dict_file_path",
                        default="data/2014_renmin_func_dict.json",
                        type=str,
                        required=False,
                        help="Path of the vocab file of functional word.")
    parser.add_argument("--data_file_path",
                        default="data/2014_renmin_news_sentences_filter.txt",
                        type=str,
                        required=False,
                        help="The input dataset file path.")
    parser.add_argument("--out_data_file_path",
                        default="data/2014_renmin_news_sentences_gen.txt",
                        type=str,
                        required=False,
                        help="The output dataset file path.")

    parser.add_argument("--cls_output_dir",
                        default="data/paraphrase/",
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--mlm_output_dir",
                        default="data/pretraining/",
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--block_size", default=256, type=int,
                        help="Optional input sequence length after tokenization."
                                "The training dataset will be truncated in block of this size for training."
                                "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                                "Sequences longer than this will be truncated, and sequences shorter \n"
                                "than this will be padded.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--gpu_device', type=int, default=3,
                        help="Specify GPU device id used further")
    parser.add_argument('--beam_size', type=int, default=5)

    # parse args and check if args are valid
    args = parser.parse_args()

    return args

opt = parse_opt()

# print the main arguments(options) in table form
opt_tb = pt.PrettyTable()
opt_tb.field_names = ['arguments', 'values']
for key, value in opt.__dict__.items():
    opt_tb.add_row([key, value])
print(opt_tb)

# beam
opt.output_beam_size = opt.beam_size

# vocab
opt.PAD_idx = 0
opt.CLS_idx = 1
opt.SEP_idx = 2
opt.MASK_idx = 3
opt.UNK_idx = 17963
opt.unused1_idx = 17964
opt.max_len_sub_ids = 4
with open(opt.mlm_output_dir+'config.json', 'r') as handle:
    opt.mlm_vocab_size = json.load(handle)["vocab_size"]

# random seed setting
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# cudnn setting
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
  opt.USE_CUDA = True