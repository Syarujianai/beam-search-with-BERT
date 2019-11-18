#!/usr/bin/env python
# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
from torch.nn import Embedding
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math


class FunctionalEnvironment(object):
    """
    the basic environment to interact with agent
    it has the following functions
    initial_state = reset(sample)
    next_state, reward, done = step(action)
    """
    def __init__(self, dict, pos_set, bert_file, max_position=256, position_dim=256, bert_feature_device=0, bert_reward_device=1):

        self.dict = dict
        self.pos_set = pos_set

        # load bert model
        print("load feature bert")
        self.bert_feature_device = bert_feature_device
        self.tokenizer = BertTokenizer.from_pretrained(bert_file)
        self.bert = BertModel.from_pretrained(bert_file)
        self.bert.eval()
        self.bert.to("cuda:%d" % bert_feature_device)

        # load bert for reward calculation
        print("load reward bert")
        self.bert_reward_device = bert_reward_device
        self.bert_lm = BertForMaskedLM.from_pretrained(bert_file)
        self.bert_lm.eval()
        self.bert.to("cuda:%d" % bert_reward_device)

        # list of segmented words
        self.sample = None
        self.original_sample = None
        # current sentence embedding
        self.sentence_embedding = None
        self.original_sentence = None
        # index in
        self.function_index = 0
        self.function_positions = None

        # position embedding layer
        self.position_embedding = Embedding(max_position, position_dim)

    def reset(self, sample):
        """
        start a new session with a given example
        :param sample: list of tokens
        :return: initial state
        """
        self.sample = sample.split()
        self.original_sample = sample.split()
        sentence, self.function_positions = self.ensemble_sentence(self.sample)

        if len(self.function_positions) == 0:
            return None
        print("start sentence", sentence)
        self.original_sentence = sentence
        self.sentence_embedding = self.get_sentence_embedding(sentence)
        self.function_index = 0
        state = self.get_state()
        return state

    def step(self, action):
        """
        perform an action and update the environment
        :param action: scalar value, word id in dict
        :return: next state, reward, done
        """
        target_functional_word = self.dict[action]
        reward = 0
        done = False
        next_state = None
        # update state
        previous_function_position = self.function_positions[self.function_index]
        sample_index = previous_function_position["sample_index"]
        self.sample[sample_index] = target_functional_word

        updated_sentence, _ = self.ensemble_sentence(self.sample)
        print("updated sentence", updated_sentence)
        self.sentence_embedding = self.get_sentence_embedding(updated_sentence)

        if self.function_index < len(self.function_positions) - 1:
            self.function_index += 1
            next_state = self.get_state()
        else:
            done = True
            reward = self.get_reward(updated_sentence, self.sample)
        return next_state, reward, done

    def get_state(self):
        """
        get state based on current position and sentence embedding
        :return: concat of position embedding and sentence embedding, pos tag as well
        """
        positions = self.function_positions[self.function_index]
        sentence_index = positions["sentence_index"]
        pos_tag = positions["pos"]
        position_embedding = self.get_position_embedding(sentence_index)
        state_embedding = torch.cat((self.sentence_embedding, position_embedding), 0)
        return tuple((state_embedding, pos_tag))

    def get_reward(self, final_sentence, final_sample):
        """
        get the final reward
        :param final_sentence:
        :param final_sample:
        :return:
        """
        original_probability = self.get_sentence_probability(self.original_sentence)
        final_probability = self.get_sentence_probability(final_sentence)
        probability_reward = final_probability - original_probability

        word_change = 0
        for i in range(len(final_sample)):
            final_word = final_sample[i]
            original_word = self.original_sample[i]
            if "/" in final_word:
                final_word = final_word.split("/")[0]

            if "/" in original_word:
                original_word = original_word.split("/")[0]
            if final_word != original_word:
                word_change += 1
        reward = probability_reward + word_change
        print("final reward ", reward)
        return reward

    def get_sentence_probability(self, sentence):
        """
        log likelihood of a sentence
        :param sentence: sentence string
        :return: likelihood
        """
        input_sequence = "[CLS]"
        result = 0
        for i in range(len(sentence)):
            token = sentence[i]
            token_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            input_sequence_with_mask = input_sequence + "[MASK]"
            tokenized_text = self.tokenizer.tokenize(input_sequence_with_mask)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [0 for _ in range(len(tokenized_text))]

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segment_ids])

            tokens_tensor = tokens_tensor.to("cuda:%d" % self.bert_reward_device)
            segments_tensors = segments_tensors.to("cuda:%d" % self.bert_reward_device)

            with torch.no_grad():
                outputs = self.bert_lm(tokens_tensor, token_type_ids=segments_tensors)
                predictions = F.softmax(outputs[0], dim=-1)
                # print(predictions.shape)
                target_probability = math.log(predictions[0, len(tokenized_text) - 1, token_id].item())
                # print(target_probability)
                result += target_probability
            input_sequence += token
        return result

    def get_position_embedding(self, sentence_index):
        """
        get position embedding based on start and end index
        :param sentence_index:
        :return:
        """
        position_tensor = torch.LongTensor(sentence_index)
        position_embedding = self.position_embedding(position_tensor)
        position_embedding = torch.cat((position_embedding[0], position_embedding[1]), 0)
        return position_embedding

    def get_sentence_embedding(self, sentence):
        """
        get sentence embedding from bert
        :param sentence: complete sentence
        :return:
        """
        sentence = "[CLS]" + sentence
        tokenized_text = self.tokenizer.tokenize(sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [0 for _ in range(len(tokenized_text))]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segment_ids])

        tokens_tensor = tokens_tensor.to("cuda:%d" % self.bert_feature_device)
        segments_tensors = segments_tensors.to("cuda:%d" % self.bert_feature_device)

        with torch.no_grad():
            outputs = self.bert(tokens_tensor, token_type_ids=segments_tensors)
            encoded_layers = outputs[0]
            sentence_embedding = encoded_layers[0][0]
        return sentence_embedding.cpu()

    def ensemble_sentence(self, sample):
        """
        :param sample: 这个/rz 策略/n 要/v 比/p
        :return: complete sentence string
                 function positions: {"sentence index", "pos tag", "sample index"}
        """
        pos_tag = {"p", "c", "u", "e", "d"}
        words = sample
        sentence = ""
        function_positions = []
        index = 0
        for i in range(len(words)):
            word = words[i]
            if "/" in word:
                elements = word.split("/")
                char = elements[0]
                tag = elements[1][0]
                sentence += char
                function_position = {"sentence_index": [index, index + len(char) - 1], "pos": tag, "sample_index": i}
                if tag in pos_tag:
                    function_positions.append(function_position)
                index += len(char)
            else:
                sentence += word
        return sentence, function_positions
