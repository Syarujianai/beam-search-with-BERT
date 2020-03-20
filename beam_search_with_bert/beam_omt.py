""" Manage beam search info structure.
    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import random
import numpy as np
from itertools import chain
from operator import itemgetter
from utilis.config import opt as opt

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

FUNC_POS_TAG = {'/p', '/u', '/e', '/c', '/d'}


class Beam():
    ''' Beam search '''
    def __init__(self, size, device=False):

        self.cnt = 0
        self.size = size
        self._done = False

        # The score for each translation on the beam (init: 0).
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step (init: [torch.tensor([CLS_idx<top>, PAD_idx, ..., PAD_idx])]).
        self.next_ys = [torch.full((size,), opt.PAD_idx, dtype=torch.long, device=device)]
        self.next_ys[0][0] = opt.CLS_idx
        

    def get_current_state(self, is_sub_func_idx=True, vocab=None, len_seq_left=None, padding_on_left=True):
        "Get the outputs for the current timestep."
        if is_sub_func_idx:
            assert vocab is not None
        return self.get_tentative_hypothesis(is_sub_func_idx, vocab, len_seq_left, padding_on_left)

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)  # vocab_size

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)  # (beam_size, vocab_size), summarize log probability to scores(along with beams) (traversal whole vocab)
        else:
            beam_lk = word_prob[0]  # init: scores == [score<CLS>, ..., 0], no need to sum up

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort, top k (beam_size)
        # best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort ???

        # NOTE: update all_scores and scores (by best_scores)
        self.all_scores.append(self.scores)  # append previous best_scores
        self.scores = best_scores

        # NOTE: update prev_ks and next_ys (by best_scores_id)
        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words  # locate previous beam this word inherit from, i.e., previous beam k
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)  # i.e., token_idx, best_scores_id % num_words

        # End condition is when top-of-beam is [SEP].
        if self.next_ys[-1][0].item() == opt.SEP_idx:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done  # this instance is done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self, is_sub_func_idx=True, vocab=None, len_seq_left=None, padding_on_left=True):
        "Get the decoded sequence for the current timestep."
        if len(self.next_ys) == 1:  # init: next_ys: [[CLS_idx(top)], [PAD_idx], ..., [PAD_idx]], [len_dec_seq, batch_size]
            dec_seq = self.next_ys[0].unsqueeze(1)
            lens_dec_seq = [1]*self.size
        else:
            if is_sub_func_idx:
                assert vocab is not None
            assert len_seq_left is not None

            _, keys = self.sort_scores()  # NOTE: sorted scores for getting score-ordered beam
            hyps = [self.get_hypothesis(k, is_sub_func_idx=is_sub_func_idx, vocab=vocab) for k in keys]
            lens_dec_seq = [len(h)+1 for h in hyps]

            if padding_on_left:
                # NOTE: 1. padding on left is convenient for getting masked `opt.max_len_sub_ids` logits of `ERNIE`
                #       2. first token of `ERNIE` input must be [CLS]
                hyps = [[opt.PAD_idx]*(opt.block_size-opt.max_len_sub_ids-len(h)-len_seq_left-1) + [opt.CLS_idx] + h for h in hyps]
                # hyps = [[opt.CLS_idx] + [opt.MASK_idx]*(opt.block_size-opt.max_len_sub_ids-len(h)-len_seq_left-1) + h for h in hyps]
            else:
                hyps = [torch.LongTensor([opt.CLS_idx]+h+[opt.unused1_idx]*(opt.block_size-opt.max_len_sub_ids-len(h)-len_seq_left-1)) for h in hyps]  # TODO: more elegant type -- attention mask
            dec_seq = torch.LongTensor(hyps)
 
        return dec_seq, lens_dec_seq

    def get_hypothesis(self, k, is_sub_func_idx=False, vocab=None):
        "Walk back to construct the full hypothesis."
        hyp = []
        for j in range(len(self.prev_ks)-1, -1, -1):  # postorder walk (len_dec_seq ~ 1)
            try:
                if vocab is not None:
                    assert k is not None
                    assert k < vocab["FUNC_IDX_START"]
                dec_token_back = self.next_ys[j+1][k]
            except ValueError as e:
                print("ValueError{:s}".format(e))
            # finally:
            #     print("cnt:{:d}".format(self.cnt), '\t', "step:{:d}".format(j), '\t', "index:{:d}".format(k), '\n', self.sort_scores())

            # HACK: substitute functional word index to corresponding sub words indices (for char-wise ERNIE)
            if is_sub_func_idx:
                assert vocab is not None
                if dec_token_back.item() >= vocab["FUNC_IDX_START"]:
                    dec_sub_ids_back = vocab["func_id_to_sub_ids_map"][dec_token_back.item()]
                    hyp.extend([torch.tensor(sub_idx).to(dec_token_back.device) for sub_idx in dec_sub_ids_back[::-1]])
            else:
                hyp.append(dec_token_back)  # next_ys: [len_dec_seq, beam_size], store all stored token_idx
            
            k = self.prev_ks[j][k]  # prev_ks: [len_dec_seq, beam_size], a pointer which point to correspond previous beam id
        
        return list(map(lambda x: x.item(), hyp[::-1]))  # reverse sequence


class Translator(object):
    ''' Load with trained model and handle the beam search '''
    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.func_vocab = tokenizer.func_vocab
        self.vocab_size = self.func_vocab["FUNC_IDX_START"]
        self.vocab_size_with_func = len(self.vocab)
        print("Vocab size with functional words: {:d}".format(self.vocab_size_with_func))
        self.beam_size = opt.beam_size
        self.device = torch.device("cuda:"+str(opt.device_ids[0]) if torch.cuda.is_available() else "cpu")
        self.cnt = 0


    def translate_batch(self, batch, is_reward_non_para=False):
        """ Translation work in one batch.
        :params batch: a dictionary of input batch
        """ 
        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            "Indicate the position of an instance in a tensor."
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            "Collect tensor parts associated to active instances."
            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst*n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)
            
            return beamed_tensor

        def collate_active_info(src_seq_temp, inst_idx_to_position_map, active_inst_idx_list, src_seq=None):
            "Index select (reshape) src_seq, src_enc, and update inst_idx_to_position_map, with active_inst_idx_list"
            # NOTE: inst_idx_to_position_map is useful for collected active sequences and embeddings
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)  # previous active instance list
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]  # current active instance list
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            # select sequences with regard to active instances
            active_src_seq_temp = collect_active_part(src_seq_temp, active_inst_idx, n_prev_active_inst, n_bm)

            # select embeddings with regard to active instances
            # active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            if self.is_reward_non_para and src_seq is not None:
                active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
                return active_src_seq_temp, active_src_seq, active_inst_idx_to_position_map

            return active_src_seq_temp, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq_temp, enc_lens, inst_idx_to_position_map, n_bm, src_seq=None):
            """ Decode and update beam status, and then return active beam idx.
            :params src_seq_temp: batch templatized sequences start with [CLS] and end with [SEP] ([CLS] X [SEP])
            :params src_seq: batch sequences start with [CLS] and end with [SEP] ([CLS] X [SEP])
            :params len_dec_seq: batch decoding step
            :params inst_idx_to_position_map: previous step's `inst_idx_to_position_map`, used for `collect_active_inst_idx_list`
            :returns active_inst_idx_list: indices of active instances of batch
            """
            def prepare_beam_dec_seq(inst_dec_beams, n_active_inst, n_bm, len_dec_seq, enc_lens, is_sub_func_idx=False, vocab=None):
                "Get decoded partial sequences from beams"
                # get current state of active instances
                len_seq_left = enc_lens.max().item() - len_dec_seq  # NOTE: left length of sequence except [CLS] and <seq>
                dec_partial_seq, lens_dec_partial_seq = [], []
                for b in inst_dec_beams:    
                    if not b.done:
                        dps, len_dps = b.get_current_state(is_sub_func_idx, vocab, len_seq_left)
                        dec_partial_seq.append(dps)
                        lens_dec_partial_seq.append(len_dps)  # list(list(int))
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device).view(n_active_inst*n_bm, -1)
                return dec_partial_seq, len_seq_left, lens_dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq+1, dtype=torch.long, device=self.device)  # [1, .., len_dec_seq]
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst*n_bm, 1)  # [n_active_inst*n_bm, len_dec_seq]
                return dec_partial_pos
                  
            def predict_word(dec_seq, src_seq_temp, n_active_inst, n_bm, len_dec_seq, len_seq_left, src_seq=None, lens_dec_seq=None, enc_lens=None):
                "Get predict logits over vocab"

                def get_candidate_sequence(dec_seq, sub_ids_in_POS, future_seq, lens_dec_seq, lens_sub_ids_in_POS, len_dec_seq, enc_lens):    
                    lists_dec_seq = _convert_tensor_to_lists(dec_seq, lens_dec_seq, padding_on_left=False)
                    lists_future_seq = _convert_tensor_to_lists(future_seq)
                    assert len(lists_dec_seq) == len(lists_future_seq) 

                    import ipdb; ipdb.set_trace()

                    vocab_size_in_POS = len(sub_ids_in_POS)
                    lists_sub_ids_in_POS = _convert_tensor_to_lists(sub_ids_in_POS.squeeze(-1), lens_sub_ids_in_POS)  # [vocab_size_in_POS, opt.max_len_sub_ids]                    
                    
                    # repeat vocab_size_in_POS
                    candidate_seq = []
                    for list_sub_ids_inst_beam in lists_sub_ids_in_POS:
                        for dec_seq_inst_beam, future_seq_inst_beam in zip(lists_dec_seq, lists_future_seq):
                            # <CLS>+<sentence1>+<SEP>+<unused1>+<sentence2>+<SEP>+<PAD>
                            # repeat `list_sub_ids_inst_beam` n_POS_inst*n_bm
                            candidate_seq.append(torch.LongTensor(dec_seq_inst_beam + list_sub_ids_inst_beam + future_seq_inst_beam))
 
                    vocab_size_in_POS = len(sub_ids_in_POS)
                    n_POS_inst_mul_n_bm = len(lists_dec_seq)
                    candidate_seq = pad_sequence(candidate_seq, opt.unused1_idx) \
                                    .view(vocab_size_in_POS, n_POS_inst_mul_n_bm,-1).to(self.device)  # NOTE: pad on right with [unused1]
                    return candidate_seq
                
                def _flatten_lists(list_of_lists):
                    return chain.from_iterable(list_of_lists)
                
                def _convert_tensor_to_lists(tensor_dec_seq, lens_dec_seq=None, padding_on_left=True):
                    lists_dec_seq = []
                    if lens_dec_seq is not None:
                        if len(lens_dec_seq) != tensor_dec_seq.size(0):
                            lens_dec_seq = _flatten_lists(lens_dec_seq)
                            assert len(lens_dec_seq) == tensor_dec_seq.size(0)
                        for seq, len_seq in zip(tensor_dec_seq, lens_dec_seq):
                            if padding_on_left:
                                lists_dec_seq.append(seq[:len_seq].cpu().numpy().tolist())
                            else:
                                lists_dec_seq.append(seq[-len_seq:].cpu().numpy().tolist())
                        return lists_dec_seq
                    else:
                        return [hyp.cpu().numpy().tolist() for hyp in tensor_dec_seq]  # (n_active_inst*beam_size,)

                # get next decode token
                next_dec_token = src_seq_temp[:, len_dec_seq]  # (n_active_inst*beam_size,)

                # mask next `max_len_sub_ids` decode tokens
                dummy_next_dec_tokens = torch.full((n_active_inst*n_bm, opt.max_len_sub_ids),
                                                    opt.MASK_idx, dtype=torch.long, device=self.device)

                # get future sequence
                future_seq = src_seq_temp[:, len_dec_seq:]  # [n_active_inst*beam_size, max_len_seq-len_dec_seq]
                future_seq = future_seq.masked_fill(future_seq>=opt.mlm_vocab_size, opt.MASK_idx)  # mask POS template
                masked_candidate_seq = torch.cat((dec_seq, dummy_next_dec_tokens, future_seq), -1)  # [n_active_inst*n_bm, block_size] / [n_active_inst*n_bm, max_seq_len+opt.max_len_sub_ids]

                # get MLM scores of masked next word
                (logits_mlm,) = self.model.bert_mlm(masked_candidate_seq)  # [n_active_inst*n_bm, block_size, vocab_size]

                # convert score of masked tokens to log-likelihood
                masked_next_words_prob = torch.log_softmax(logits_mlm[:,(-len_seq_left-opt.max_len_sub_ids):(-len_seq_left),:], dim=-1)  # [n_active_inst*n_bm, opt.max_len_sub_ids, vocab_size]

                # initialize log-likelihood with GROUND TRUTH (and POS instance will be covered later), HACK: userdefined word_prob
                masked_next_word_prob_with_func = torch.log(torch.zeros(n_active_inst*n_bm, self.vocab_size_with_func)  # [n_active_inst*n_bm, self.vocab_size_with_func]
                                                    .to(self.device).scatter_(-1, next_dec_token.unsqueeze(1), 1.))

                for POS_tag in FUNC_POS_TAG:
                    next_dec_token_eq = next_dec_token.eq(self.vocab[POS_tag])
                    if not next_dec_token_eq.any().item():
                        continue
                    
                    # get met POS instances (that next decode token is POS template)
                    indices_POS_inst = next_dec_token_eq.nonzero()  # [n_POS_inst*n_bm, 1]
                    n_POS_inst_mul_n_bm = indices_POS_inst.size(0)
                    masked_next_words_prob_POS = masked_next_words_prob.index_select(0, indices_POS_inst.squeeze(-1))  # [n_POS_inst*n_bm, max_len_sub_ids, vocab_size], TODO: optimize

                    # get extend indices of functional words in POS
                    idx_range = self.func_vocab["FUNC_IDS_SET_CAT"][POS_tag]
                    if POS_tag == '/d': idx_range = random.sample(idx_range, 100)  # subsample functional word vocab, TODO: sample by tf

                    ## sclice dict: https://stackoverflow.com/questions/18453566/python-dictionary-get-list-of-values-for-list-of-keys
                    sub_ids_in_POS = list(itemgetter(*idx_range)(self.func_vocab["func_id_to_sub_ids_map"]))
                    lens_sub_ids_in_POS = [len(sub_ids) for sub_ids in sub_ids_in_POS]
                    sub_ids_in_POS = [sub_ids + [opt.PAD_idx]*(opt.max_len_sub_ids-len(sub_ids)) for sub_ids in sub_ids_in_POS]
                    sub_ids_in_POS = torch.LongTensor(sub_ids_in_POS).unsqueeze(0).unsqueeze(-1) \
                                        .expand(n_POS_inst_mul_n_bm, -1, -1, -1).to(self.device)  # [n_POS_inst*n_bm, vocab_size_in_POS, max_len_sub_ids, 1] 

                    ## masking
                    mask_padding = sub_ids_in_POS.eq(opt.PAD_idx)  # [n_POS_inst*n_bm, vocab_size_in_POS, max_len_sub_ids, 1] 

                    # Goal: [n_POS_inst*n_bm, max_len_sub_ids, vocab_size] -> [n_POS_inst*n_bm, vocab_size_in_POS]
                    vocab_size_in_POS = sub_ids_in_POS.size(1)
                    masked_next_sub_words_prob = torch.zeros(n_POS_inst_mul_n_bm, vocab_size_in_POS, opt.max_len_sub_ids).to(self.device)  # [n_POS_inst*n_bm, vocab_size_in_POS, max_len_sub_ids]
                    for vi in range(vocab_size_in_POS):
                        # gather log-likelihood of sub indices corresponding to functional word in current POS vocabulary
                        masked_next_sub_words_prob[:,vi,:] = masked_next_words_prob_POS.gather(-1, sub_ids_in_POS[:,vi,:,:]) \
                                                            .masked_fill(mask_padding[:,vi,:,:], 0.).squeeze(-1)  # NOTE: fill log-likelihood 0. in padding position
                
                    # sum up log-likelihood of sub indices corresponding functional word in current POS vocabulary
                    masked_next_word_prob_with_func_in_POS = torch.exp((masked_next_sub_words_prob).sum(-1))  # [n_POS_inst*n_bm, vocab_size_in_POS]
                    indices_func_set_in_POS = torch.LongTensor(idx_range).unsqueeze(0).expand_as(masked_next_word_prob_with_func_in_POS).to(self.device)  # [n_POS_inst*n_bm, vocab_size_in_POS]
                    
                    # scatter into overall vocabulary
                    # import ipdb; ipdb.set_trace()
                    masked_next_word_prob_with_func_in_POS = torch.zeros(n_POS_inst_mul_n_bm, self.vocab_size_with_func).to(self.device) \
                                                            .scatter_(-1, indices_func_set_in_POS, masked_next_word_prob_with_func_in_POS)  # [n_POS_inst*n_bm, vocab_size_with_func]
                    
                    # normalize by functional vocabulary and convert to log-likelihood
                    masked_next_word_prob_with_func_in_POS = torch.log_softmax(masked_next_word_prob_with_func_in_POS, dim=-1)

                    # scatter log-likelihoods with regard to met POS instances
                    masked_next_word_prob_with_func = masked_next_word_prob_with_func.scatter_(
                                                            0, indices_POS_inst.expand_as(masked_next_word_prob_with_func_in_POS),
                                                            masked_next_word_prob_with_func_in_POS
                                                        )  # [n_active_inst*n_bm, vocab_size_with_func]
                    
                    ## get non-paraphrase scores of candidate sequence
                    if self.is_reward_non_para and src_seq is not None:
                        # get <sentence_1> of paraphrase classifier
                        dec_seq_POS = dec_seq.index_select(0, indices_POS_inst.squeeze(-1))
                        future_seq_POS = future_seq.index_select(0, indices_POS_inst.squeeze(-1))
                        lens_dec_seq_POS = torch.LongTensor(lens_dec_seq).flatten().to(self.device) \
                                            .index_select(0, indices_POS_inst.squeeze(-1)).tolist()
                        candidate_seq_POS = get_candidate_sequence(dec_seq_POS, sub_ids_in_POS[0], future_seq_POS, 
                                                                   lens_dec_seq_POS, lens_sub_ids_in_POS,
                                                                   len_dec_seq, enc_lens)  # [vocab_size_in_POS, n_POS_inst*n_bm, len_seq+2]

                        # get <sentence_2> of paraphrase classifier and repeat vocab_size_in_POS
                        src_seq_POS = src_seq[:, 1:].index_select(0, indices_POS_inst.squeeze(-1)) \
                                        .unsqueeze(0).repeat(vocab_size_in_POS, 1, 1)
                        
                        # combine <sentence_1> and <sentence_2>
                        input_ids_cls_POS = torch.cat((candidate_seq_POS, src_seq_POS), -1).transpose(0, 1) \
                                            .reshape(n_POS_inst_mul_n_bm*vocab_size_in_POS, -1)  # [n_POS_inst*n_bm*vocab_size_in_POS, 2*len_seq+3]

                        # get non-paraphrase scores of input pairs
                        self.cnt += 1
                        (logits_cls,) = self.model.bert_cls(input_ids_cls_POS)  # (n_POS_inst*n_bm*vocab_size_in_POS,)
                        non_paraphrase_likehd = torch.log(torch.sigmoid(logits_cls[:,0])) \
                                                .view(n_POS_inst_mul_n_bm, vocab_size_in_POS)  # [n_POS_inst*n_bm, vocab_size_in_POS]
                        if self.cnt % 10 == 0:
                            # convert_ids_to_string(candidate_seq_POS[0].transpose(0,1), self.tokenizer)
                            # convert_ids_to_string(src_seq_POS[0].transpose(0,1), self.tokenizer)
                            # print('\n')
                            convert_ids_to_string(input_ids_cls_POS, self.tokenizer)
                            print('\n', "logits_cls: ", logits_cls)
                        
                        # scatter into overall vocabulary
                        non_paraphrase_likehd = torch.zeros(n_POS_inst_mul_n_bm, self.vocab_size_with_func).to(self.device) \
                                                .scatter_(-1, indices_func_set_in_POS, non_paraphrase_likehd)  # [n_POS_inst*n_bm, vocab_size_with_func]

                        # scatter add non-paraphrase scores with regard to met POS instances
                        masked_next_word_prob_with_func = masked_next_word_prob_with_func.scatter_add_(
                                                                0, indices_POS_inst.expand_as(non_paraphrase_likehd),
                                                                opt.alpha*non_paraphrase_likehd,
                                                            )

                masked_next_word_prob_with_func = masked_next_word_prob_with_func.view(n_active_inst, n_bm, -1)  # [n_active_inst, n_bm, vocab_size_with_func]

                return masked_next_word_prob_with_func

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                "Update the beam with predicted word prob information and collect active instances."
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])  # NOTE: for per instance: [n_bm, vocab_size]
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq, len_seq_left, lens_dec_seq = \
                prepare_beam_dec_seq(
                    inst_dec_beams, n_active_inst, n_bm, len_dec_seq, enc_lens,
                    is_sub_func_idx=True, vocab=self.func_vocab
                )  # [n_active_inst*n_bm, len_dec_seq] 

            
            if self.is_reward_non_para and src_seq is not None:
                word_prob = predict_word(dec_seq, src_seq_temp, n_active_inst, n_bm, len_dec_seq, len_seq_left, 
                                         src_seq=src_seq, lens_dec_seq=lens_dec_seq, enc_lens=enc_lens)
            else:
                word_prob = predict_word(dec_seq, src_seq_temp, n_active_inst, n_bm, len_dec_seq, len_seq_left)  # [n_active_inst, n_bm, vocab_szie]

            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):  # traversal all instances
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()  # Beam.sort_scores(): return values, keys
                all_scores += [scores[:n_best]]  # while: n_best == 1, return the top hypothesis of beam

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Eval
            self.model.bert_mlm.eval()
            self.model.bert_cls.eval()

            #-- Encode
            enc_temp_batch, enc_batch, enc_padding_mask, enc_lens = get_input_from_batch(batch)  # enc_batch: [batch_size, seq_len]

            #-- Repeat data beam_size times for each instance for beam search
            n_bm = self.beam_size
            n_inst, len_s = enc_temp_batch.size()
            src_seq_temp = enc_temp_batch.repeat(1, n_bm).view(n_inst*n_bm, len_s)  # [batch_size*beam_size, seq_len]
            self.is_reward_non_para = is_reward_non_para
            if self.is_reward_non_para:
                src_seq = enc_batch.repeat(1, n_bm).view(n_inst*n_bm, -1)
            
            #-- Prepare beams for each instance
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)] 

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)  # map active instances of batch of current positions

            #-- Decode
            for len_dec_seq in range(1, opt.max_seq_length + 1):  # NOTE: end condition
                if self.is_reward_non_para:
                    active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_seq_temp, enc_lens, inst_idx_to_position_map, n_bm, src_seq)
                else:
                    active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_seq_temp, enc_lens, inst_idx_to_position_map, n_bm)
                
                if not active_inst_idx_list:
                    break
                
                if self.is_reward_non_para:
                    src_seq_temp, src_seq, inst_idx_to_position_map = collate_active_info(src_seq_temp, inst_idx_to_position_map, active_inst_idx_list, src_seq)
                else:
                    src_seq_temp, inst_idx_to_position_map = collate_active_info(src_seq_temp, inst_idx_to_position_map, active_inst_idx_list)

        # NOTE: after beam_decode_step
        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, opt.output_beam_size)

        return batch_hyp, batch_scores


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
    return seq_range_expand < seq_length_expand
    
def get_input_from_batch(batch):
    """Get input batch and lengths from batch."""
    enc_batch = batch["input_batch"]
    enc_temp_batch = batch["input_temp_batch"]
    if enc_temp_batch.size(1) == opt.eval_batch_size:
        enc_batch = enc_batch.transpose(0,1)
        enc_temp_batch = enc_temp_batch.transpose(0,1)  # [batch_size, seq_len]
    enc_lens = batch["input_lengths"]  # (batch_size,)
    
    batch_size, max_enc_len = enc_temp_batch.size()
    assert enc_lens.size(0) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    return enc_temp_batch, enc_batch, enc_padding_mask, enc_lens

def convert_ids_to_string(seqs, tokenizer):
    for seq in seqs:
        new_words = []
        for idx in seq:
            new_words.append(tokenizer._convert_id_to_token(idx.item()))
        print(''.join(new_words) + "\n")