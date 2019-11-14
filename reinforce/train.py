import os
import argparse
from reinforce.environments.functional_environment import FunctionalEnvironment
from reinforce.models.a2c import A2CAgent


def build_dict(dict_folder):
    dict = {}
    index = 0
    pos_index = {}
    for filename in os.listdir(dict_folder):
        pos_tag = filename.split(".")[0]
        start_index = index
        with open(dict_folder + filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                dict[index] = line
                index += 1
        end_index = index - 1
        pos_index[pos_tag] = [start_index, end_index]
    return dict, pos_index


def train(dict, pos_index, bert_file, sentence_file):
    action_number = len(dict)
    state_dim = 256 * 3
    a2c_agent = A2CAgent(action_number, state_dim)
    env = FunctionalEnvironment(dict, bert_file)
    with open(sentence_file, "r") as sentences:
        lines = sentences.readlines()
        for line in lines:
            state = env.reset(line)
            done = False
            while not done:
                state_emb, pos = state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_folder", type=str, help="folder containing dict files")
    parser.add_argument("--bert_file", type=str, help="bert model folder")
    parser.add_argument("--sentence_file", type=str, help="file containing the sentences")