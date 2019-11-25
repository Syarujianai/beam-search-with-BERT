
import ipdb
import json

import torch
from torch.utils.data import SequentialSampler, DataLoader

from utilis.config import opt as opt
from utilis.dataset import TextDataset as MyTextDataset
from utilis.tokenization import BertTokenizer as MyTokenizer

from load_bert import bert_model
from beam_omt import Translator as MyTranslator

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
my_translator = MyTranslator(bert, my_tokenizer.vocab, my_tokenizer.func_vocab)
eval_data = load_and_cache_examples(my_tokenizer)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, collate_fn=eval_data.collate_fn, batch_size=opt.eval_batch_size)

out_io = open(opt.out_data_file_path, 'w', encoding='utf-8')
for batch in eval_dataloader:
    # print(bert.mlm_model(batch))
    # TODO: 1. cls_penalty=True; 2. add `input_txt`
    sent_b, _ = my_translator.translate_batch(batch)
    for i in range(batch["input_batch"].size(1)):
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
