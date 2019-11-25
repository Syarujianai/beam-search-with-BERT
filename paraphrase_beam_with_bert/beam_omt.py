""" Manage beam search info structure.
    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch
import numpy as np
from operator import itemgetter
from utilis.config import opt as opt
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

FUNC_POS_TAG = {'/p', '/u', '/e', '/c', '/d'}


class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False):

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
        

    def get_current_state(self, is_sub_func_idx=True, vocab=None, len_seq_left=None):
        "Get the outputs for the current timestep."
        if is_sub_func_idx:
            assert vocab is not None
        return self.get_tentative_hypothesis(is_sub_func_idx, vocab, len_seq_left)

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
            beam_lk = word_prob[0]  # init: scores == [0, ..., 0], no need to sum up

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

    def get_tentative_hypothesis(self, is_sub_func_idx, vocab, len_seq_left):
        "Get the decoded sequence for the current timestep."
        if len(self.next_ys) == 1:  # init: next_ys: [[CLS_idx(top)], [PAD_idx], ..., [PAD_idx]], [len_dec_seq, batch_size]
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()  # NOTE: sorted scores for getting order keys ? 
            hyps = [self.get_hypothesis(k, is_sub_func_idx, vocab) for k in keys]
            # NOTE:
            # 1. padding on left is convenient for getting masked `opt.max_len_sub_ids` logits of `ERNIE`
            # 2. first token of `ERNIE` input must be [CLS]
            hyps = [[opt.CLS_idx] + [opt.MASK_idx]*(opt.block_size-opt.max_len_sub_ids-len(h)-len_seq_left-1) + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k, is_sub_func_idx=False, vocab=None):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks)-1, -1, -1):  # postorder walk (len_dec_seq ~ 1)
            dec_token_back = self.next_ys[j+1][k]
            # HACK: substitute functional word index to corresponding sub words indices (for char-wise ERNIE)
            if is_sub_func_idx:
                if dec_token_back.item() >= vocab["FUNC_IDX_START"]:
                    dec_sub_ids_back = vocab["func_id_to_sub_ids_map"][dec_token_back.item()]
                    hyp.extend([torch.tensor(sub_idx).to(dec_token_back.device) for sub_idx in dec_sub_ids_back[::-1]])
            else:
                hyp.append(dec_token_back)  # next_ys: [len_dec_seq, beam_size], store all stored token_idx
            
            k = self.prev_ks[j][k]  # prev_ks: [len_dec_seq, beam_size], a pointer which point to correspond previous beam id
        
        return list(map(lambda x: x.item(), hyp[::-1]))  # reverse sequence


class Translator(object):
    ''' Load with trained model and handle the beam search '''
    def __init__(self, model, vocab, func_vocab):
  
        self.model = model
        self.vocab = vocab
        self.func_vocab = func_vocab
        self.vocab_size = func_vocab["FUNC_IDX_START"]
        self.vocab_size_with_func = len(vocab)
        print("Vocab size with functional words: {:d}".format(self.vocab_size_with_func))
        self.beam_size = opt.beam_size
        self.device = torch.device("cuda:"+str(opt.gpu_device) if torch.cuda.is_available() else "cpu")


    def translate_batch(self, src_seq):  # TODO: bert_cls
        ''' Translation work in one batch 
        :params src_seq: a tensor shape like (batch_size, len_seq)
        '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst*n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)
            
            return beamed_tensor

        def collate_active_info(src_seq, inst_idx_to_position_map, active_inst_idx_list):
            ''' index select (reshape) src_seq, src_enc, and update inst_idx_to_position_map, with active_inst_idx_list '''
            # NOTE: inst_idx_to_position_map is useful for collected active sequences and embeddings
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)  # previous active instance list
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]  # current active instance list
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            # TODO: 未templatized src_seq在non-paraphrase penalties阶段也有用
            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)  # select sequences(src_seq) with active instance
            # active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)  # select embeddings(src_enc) with active instance
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_lens, inst_idx_to_position_map, n_bm):
            ''' Decode and update beam status, and then return active beam idx 
            :params len_dec_seq: batch decoding step
            :params src_seq: batch templatized sequences start with [CLS] and end with [SEP] ([CLS] X [SEP])
            :params inst_idx_to_position_map: previous step's `inst_idx_to_position_map`, used for `collect_active_inst_idx_list`
            :returns active_inst_idx_list: indices of active instances of batch
            '''

            def prepare_beam_dec_seq(inst_dec_beams, n_active_inst, n_bm, len_dec_seq, enc_lens, is_sub_func_idx=False, vocab=None):
                ''' get decoded partial sequences from beams '''
                # get current state of active instances
                len_seq_left = enc_lens.max().item() - len_dec_seq  # NOTE: except [CLS] and <seq>
                dec_partial_seq = [b.get_current_state(is_sub_func_idx, vocab, len_seq_left) for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(n_active_inst*n_bm, -1)  # [n_active_inst*n_bm, len_dec_seq] or [n_active_inst*n_bm, block_size]
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq+1, dtype=torch.long, device=self.device)  # [1, .., len_dec_seq]
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst*n_bm, 1)  # [n_active_inst*n_bm, len_dec_seq]
                return dec_partial_pos
                  
            def predict_word(dec_seq, src_seq, n_active_inst, n_bm, len_dec_seq):
                # # TODO: add mask tokens into dec_seq for per beam of instance
                # masked_next_words_prob_inst = []
                # for i, inst_dec_seq in enumerate(dec_seq):
                #     for inst_bm_dec_seq in inst_dec_seq:

                # get next decode token
                next_dec_token = src_seq[:, len_dec_seq]  # (n_active_inst*beam_size,)

                # mask next `max_len_sub_ids` decode tokens
                dummy_next_dec_tokens = torch.full((n_active_inst*n_bm, opt.max_len_sub_ids),
                                                    opt.MASK_idx, dtype=torch.long, device=self.device)
                # get future sequence
                if len_dec_seq > 1:
                    future_dec_seq = src_seq[:, len_dec_seq:]  # [n_active_inst*beam_size, max_len_seq - len_dec_seq]
                    future_dec_seq = future_dec_seq.masked_fill(future_dec_seq>=opt.mlm_vocab_size, opt.MASK_idx)  # mask POS template
                    masked_next_dec_seq = torch.cat((dec_seq, dummy_next_dec_tokens, future_dec_seq), -1)  # [n_active_inst*n_bm, block_size]
                else:
                    masked_next_dec_seq = torch.cat((dec_seq, dummy_next_dec_tokens), -1)  # [n_active_inst*n_bm, block_size]

                # get scores of masked next word
                self.model.bert_mlm.eval()
                with torch.no_grad():
                    (masked_word_prob,) = self.model.bert_mlm(masked_next_dec_seq)  # [n_active_inst*n_bm, block_size, vocab_size]
                masked_next_words_prob = torch.log_softmax(masked_word_prob[:,-opt.max_len_sub_ids:,:], dim=-1)  # [n_active_inst*n_bm, opt.max_len_sub_ids, vocab_size], NOTE: convert to log-likelihood

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
                    masked_next_words_prob_POS = masked_next_words_prob.index_select(0, indices_POS_inst.squeeze(-1))  # [n_POS_inst*n_bm, max_len_sub_ids, vocab_size]

                    # get extend indices of functional words in POS
                    idx_range = self.func_vocab["FUNC_IDS_SET_CAT"][POS_tag]
                    
                    ## sclice dict: https://stackoverflow.com/questions/18453566/python-dictionary-get-list-of-values-for-list-of-keys
                    sub_ids_in_POS = list(itemgetter(*idx_range)(self.func_vocab["func_id_to_sub_ids_map"]))
                    sub_ids_in_POS = [torch.LongTensor(sub_ids) for sub_ids in sub_ids_in_POS]
                    dummy_padding = torch.LongTensor([opt.PAD_idx]*opt.max_len_sub_ids)
                    sub_ids_in_POS.append(dummy_padding)  # NOTE: align with max_len_sub_ids
                    sub_ids_in_POS = pad_sequence(sub_ids_in_POS, batch_first=True, padding_value=opt.PAD_idx)[:-1] \
                                    .unsqueeze(0).unsqueeze(-1).expand(n_POS_inst_mul_n_bm, -1, -1, -1).to(self.device)  # [n_POS_inst*n_bm, vocab_size_in_POS, max_len_sub_ids, 1] 
                    ## masking
                    mask_padding = sub_ids_in_POS.eq(opt.PAD_idx)  # [n_POS_inst*n_bm, vocab_size_in_POS, max_len_sub_ids, 1] 

                    # Goal: [n_POS_inst*n_bm, max_len_sub_ids, vocab_size] -> [n_POS_inst*n_bm, vocab_size_in_POS]
                    vocab_size_in_POS = sub_ids_in_POS.size(1)
                    masked_next_sub_words_prob = torch.zeros(n_POS_inst_mul_n_bm, vocab_size_in_POS, opt.max_len_sub_ids).to(self.device)  # [n_POS_inst*n_bm, vocab_size_in_POS, max_len_sub_ids]
                    for vi in range(vocab_size_in_POS):
                        # gather log-likelihood of sub indices corresponding to functional word in current POS vocab
                        masked_next_sub_words_prob[:,vi,:] = masked_next_words_prob_POS.gather(-1, sub_ids_in_POS[:,vi,:,:]) \
                                                            .masked_fill(mask_padding[:,vi,:,:], 0.).squeeze(-1)  # NOTE: fill log-likelihood 0. in padding position
                    
                    # sum up log-likelihood of sub indices corresponding functional word in current POS vocab
                    masked_next_word_prob_with_func_in_POS = torch.exp((masked_next_sub_words_prob).sum(-1))  # [n_POS_inst*n_bm, vocab_size_in_POS]
                    indices_func_set = torch.LongTensor(idx_range).unsqueeze(0).expand_as(masked_next_word_prob_with_func_in_POS).to(self.device)  # [n_POS_inst*n_bm, vocab_size_in_POS]
                    masked_next_word_prob_with_func_in_POS = torch.zeros(n_POS_inst_mul_n_bm, self.vocab_size_with_func).to(self.device) \
                                                            .scatter_(-1, indices_func_set, masked_next_word_prob_with_func_in_POS)  # [n_POS_inst*n_bm, vocab_size_with_func]
                    # normalize likelihood by functional vocab
                    masked_next_word_prob_with_func_in_POS = torch.log_softmax(masked_next_word_prob_with_func_in_POS, dim=-1)
            
                    # fill log-likelihood of active instance with met POS instances'
                    masked_next_word_prob_with_func = masked_next_word_prob_with_func.scatter_(
                                                        0, indices_POS_inst.expand_as(masked_next_word_prob_with_func_in_POS),
                                                        masked_next_word_prob_with_func_in_POS)  # [n_active_inst*n_bm, vocab_size_with_func]

                masked_next_word_prob_with_func = masked_next_word_prob_with_func.view(n_active_inst, n_bm, -1)  # [n_active_inst, n_bm, vocab_size_with_func]

                return masked_next_word_prob_with_func

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                ''' update beams with predicted word prob '''
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])  # NOTE: for per instance: [n_bm, vocab_size]
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, n_active_inst, n_bm, len_dec_seq, enc_lens,
                                           is_sub_func_idx=True, vocab=self.func_vocab)  # [n_active_inst*n_bm, len_dec_seq]
            word_prob = predict_word(dec_seq, src_seq, n_active_inst, n_bm, len_dec_seq)  # [n_active_inst, n_bm, vocab_szie]

            # NOTE: Update the beam with predicted word prob information and collect incomplete instances
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
            #-- Encode
            enc_batch, enc_padding_mask, enc_lens = get_input_from_batch(src_seq)  # enc_batch: [batch_size, seq_len]
            mask_src = enc_batch.data.eq(opt.PAD_idx).unsqueeze(1)  # [batch_size, 1, seq_len]

            #-- Repeat data beam_size times for each instance for beam search
            n_bm = self.beam_size
            n_inst, len_s = enc_batch.size()
            src_seq = enc_batch.repeat(1, n_bm).view(n_inst*n_bm, len_s)  # [batch_size*beam_size, seq_len]
            
            #-- Prepare beams for each instance
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)] 

            #-- Bookkeeping for POS template instance
            all_POS_inst_idx_list = []

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)  # map active instances of batch of current positions

            #-- Decode
            for len_dec_seq in range(1, opt.max_seq_length + 1):  # NOTE: end condition
                active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_lens, inst_idx_to_position_map, n_bm)
                if not active_inst_idx_list:
                    break
                src_seq, inst_idx_to_position_map = collate_active_info(src_seq, inst_idx_to_position_map, active_inst_idx_list)

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
        seq_range_expand = seq_range_expand.cuda(opt.gpu_device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand
    
def get_input_from_batch(batch):
    """Get input batch and lengths from batch."""
    enc_batch = batch["input_batch"]
    if enc_batch.size(1) == opt.eval_batch_size:
        enc_batch = enc_batch.transpose(0,1)  # [batch_size, seq_len]
    enc_lens = batch["input_lengths"]  # (batch_size,)
    batch_size, max_enc_len = enc_batch.size()
    assert enc_lens.size(0) == batch_size

    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    return enc_batch, enc_padding_mask, enc_lens
