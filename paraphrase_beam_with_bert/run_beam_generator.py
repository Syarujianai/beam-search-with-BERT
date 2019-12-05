
import ipdb
import json
from tqdm import tqdm

import torch
from torch.utils.data import SequentialSampler, DataLoader

from utilis.config import opt as opt
from utilis.dataset import TextDataset as MyTextDataset
from utilis.tokenization import BertTokenizer as MyTokenizer

from load_bert import bert_model
from beam_omt import Translator as MyTranslator
from beam_omt import convert_ids_to_string

def load_and_cache_examples(tokenizer):
    dataset = MyTextDataset(tokenizer, file_path=opt.data_file_path)
    return dataset


bert = bert_model()
my_tokenizer = MyTokenizer(
    opt.vocab_file_path, 
    opt.func_dict_file_path,
    do_lower_case=False,
    do_basic_tokenize=False,
    tokenize_chinese_chars=False,
    do_tokenize_func_chars=False
)
my_translator = MyTranslator(bert, my_tokenizer)
eval_data = load_and_cache_examples(my_tokenizer)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, collate_fn=eval_data.collate_fn, batch_size=opt.eval_batch_size)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda(opt.device_ids[0])
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand > seq_length_expand


def test_bert(batch, bert):
    bert.bert_mlm.eval()

    ## print out input batch
    input_b = batch['input_batch']
    convert_ids_to_string(input_b.transpose(1, 0), my_tokenizer)

    ## test greedy decoding
    input_len = batch['input_lengths']
    src_mask = sequence_mask(input_len)
    input_b = input_b.masked_fill(input_b>=opt.mlm_vocab_size, 103).transpose(1, 0)
    with torch.no_grad():
        (logits,) = bert.bert_mlm(input_b)
    logits_norm = torch.log_softmax(logits, -1)
    logits_label = logits_norm.gather(-1, input_b.unsqueeze(-1))
    print(torch.exp(logits_label.squeeze(-1).masked_fill(src_mask, 0).sum(-1)))
    
    ## print out output batch
    convert_ids_to_string(logits.argmax(-1), my_tokenizer)
    

out_io = open(opt.out_data_file_path, 'w', encoding='utf-8')
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    sent_b, _ = my_translator.translate_batch(batch, is_reward_non_para=True)
    for i in range(batch["input_temp_batch"].size(1)):
        for j in range(opt.output_beam_size):
            new_words = []
            for w in sent_b[i][j]:
                if w==opt.SEP_idx:
                    break
                new_words.append(w)
                if len(new_words)>2 and (new_words[-2]==w):
                    new_words.pop()            
            sent_beam_search = ' '.join([my_tokenizer._convert_id_to_token(idx) for idx in new_words])
            out_io.write(str(j) + "\t" + sent_beam_search + "\n")
        # ref.append(batch['input_txt'][i])
out_io.close()
