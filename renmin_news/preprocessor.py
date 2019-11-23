import os
import re
import json

input_file = "./2014_corpus_deduplication.txt"
output_file = "./2014_renmin_news_sentences_filter.txt"
vocab_file = "./2014_renmin_vocab_filter.json"
# vocab_file=../bert-pretrained/chinese_L-12_H-768_A-12/vocab.txt 
count_cat = {'/p': [], '/u': [], '/e': [], '/c': [], '/d': []}
functional_POS_set = {'p', 'u', 'e', 'c', 'd'}
functional_POS_sub_set = {'p', 'u', 'e', 'c', 'd', 'y'}

# initialize mapping
sub_func_POS_to_func_POS_map = {}
for func_POS in functional_POS_sub_set:
    if func_POS in functional_POS_set:
        sub_func_POS_to_func_POS_map[func_POS] = func_POS
sub_func_POS_to_func_POS_map['y'] = 'e'  # NOTE: y归入叹词e类

def count_data(stat_io, func_word_count, sent_count):
    # write out vocab
    for k, v in func_word_count.items():
        for cat in count_cat.keys():
            if sub_func_POS_to_func_POS_map[re.search(r"(?=/).*", k).group()[1]] == cat[1]:
                count_cat[cat].append((re.sub(r"(?=/).*", '', k), v))  # NOTE: only store word (as key)

    # for cat in count_cat.keys():
    #     sort_items = sorted(count_cat[cat], key=lambda x: x[1], reverse=True)
    #     for k, v in sort_items:
    #         stat_io.write("result_{:s}: {:f}\t".format(k, v))
    #         stat_io.write("\n")
    #     stat_io.write("\n")
    json.dump(count_cat, stat_io, ensure_ascii=False, indent=4)
    stat_io.close()

def prepare_data(in_io, out_io, stat_io):
    """
        Revised from:
            https://github.com/yanwii/machine-translation/tree/5b736cdc99a64bd177e59d0abcecd3d855f4cfe5
    """
    sentence_count = 0
    line_count = 0
    functional_words_count = {}
    for line in in_io:
        line_count += 1
        # filter URL
        reg_url = re.compile("http/x|www/x|cn/x|com/x")
        # filter content in parentheses
        reg_parentheses = re.compile(r"(?=[\（|\【]/w).*?([\）|\】]/w)|(?=\().*?(\))")
        # filter terminated punctuation except \？
        reg_question_mark = re.compile(r"\？")
        # filter combination word POS (between ']' and ' ')
        reg_comb_POS = re.compile(r"(?=\]).*?(?= )")
        # filter chinese period within chinese quote (or punctuation in the end of paragraph)
        reg_period = re.compile(r"。/w(?= ”)")
        reg_comma = re.compile(r"，(?=[^/w])")
        reg_en_comma = re.compile(r",(?=[^/w])")
        reg_pause = re.compile(r"、(?=[^/w])")
        reg_colon = re.compile(r"：(?=[^/w])")
        reg_last_period = re.compile(r"。(?=[^/w])")
        reg_last_exclaim = re.compile(r"！(?=[^/w])")
        # filter irrelated symbol (℃)
        reg_symbols = re.compile(r"℃")
        # transform English punctuation to chinese
        reg_en_to_zh = re.compile(r"!|……")
        # preprocess of chinese quotation
        reg_left_chinese_quote = re.compile(r'“(?=[ |[\u4e00-\u9fa5])')
        reg_right_chinese_quote = re.compile(r"”(?=[^/w])")  # NOTE: those '”' followed except for '/w'
        reg_left_mid_quote = re.compile(r"《(?=[^/w])")
        reg_right_mid_quote = re.compile(r"》(?=[^/w])")
        # preprocess of paragraph
        line = line.rstrip('\n')
        line = reg_parentheses.sub('', line)
        line = reg_symbols.sub('', line)
        line = reg_question_mark.sub("$/w ？", line)
        line = reg_en_to_zh.sub("！", line)
        # complete /w symbol and drop last punctuation of paragraph
        line = reg_last_exclaim.sub("！/w", line+' ')  
        line = reg_last_period.sub("。/w", line)
        line = reg_comma.sub("，/w", line)
        line = reg_en_comma.sub(",/w ", line)
        line = reg_pause.sub("、/w ", line)
        line = reg_colon.sub("：/w ", line)
        line = line.rstrip(' ').rstrip(' ')
        if reg_url.search(line): continue
        # split paragraph to sentences by terminated punctuation
        sents = re.split(" [\！\？\。/w\；]", line)
        for sent in sents:
            sent = sent.lstrip('/w').lstrip(' ').lstrip("\●/w")
            # filter combination word identification and POS (save previous)
            sent = sent.replace('[', '')
            sent = reg_comb_POS.sub('', sent+' ').rstrip(' ')
            sent = reg_period.sub('', sent)
            sent = re.compile(r"\$(?=/w)").sub('？', sent)
            sent = reg_left_chinese_quote.sub('“/w ', sent)
            sent = reg_right_chinese_quote.sub('”/w ', sent+' ')
            sent = reg_left_mid_quote.sub('《/w ', sent)
            sent = reg_right_mid_quote.sub('》/w ', sent)
            # matching functional POS pattern that between '/' and ' ' 
            reg_POS = re.compile(r"(?=/).*?(?= )")
            sent_POS = reg_POS.findall(sent+' ')  # NOTE: for matching pattern of last POS tag
            # split sentence to words
            sent = list(filter(None, sent.split(' ')))
            # filter sentences by length and write down those sentences
            len_sent = len(sent)
            if len_sent<= 50 and len_sent >= 10:
                # perform functional statistics
                sentence_count += 1
                # filter those sentences without functional word
                is_exist_func_word = False
                assert len(sent_POS) == len(sent)
                for pos, word in zip(sent_POS, sent):
                    if pos[1] in functional_POS_sub_set:
                        is_exist_func_word = True
                        if word in functional_words_count:
                            functional_words_count[word] += 1
                        else:
                            functional_words_count[word] = 1
                if is_exist_func_word:
                    out_io.write((' ').join(sent) + "\t" + "\n")
    out_io.close()
    count_data(stat_io, functional_words_count, sentence_count)
    
out_io = open(output_file, 'w', encoding='utf-8')
stat_io = open(vocab_file, 'w', encoding='utf-8')
with open(input_file, 'r', encoding='utf-8') as in_io:
    prepare_data(in_io, out_io, stat_io)

