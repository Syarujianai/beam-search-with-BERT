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
    return dict, pos_index, set(dict.keys())


def train(function_dict, pos_index, args, pos_set):
    action_number = len(function_dict)
    state_dim = args.bert_dim + args.position_dim * 2
    a2c_agent = A2CAgent(action_number, state_dim, args.learning_rate, args.training_device)
    env = FunctionalEnvironment(function_dict, pos_set, args.bert_file, position_dim=args.position_dim,
                                bert_feature_device=args.bert_feature_device,
                                bert_reward_device=args.bert_reward_device)
    print("start training samples....")
    with open(args.sentence_file, "r") as sentences:
        lines = sentences.readlines()
        for line in lines:
            line = line.replace("]", " ]")
            state = env.reset(line)
            if state is None:
                print("neglect sentence", line)
                continue
            done = False
            state_pool = []
            reward_pool = []
            action_pool = []
            while not done:
                state_emb, pos = state
                action = a2c_agent.get_action(state_emb, pos, pos_index)
                next_state, reward, done = env.step(action)

                state_pool.append(state_emb)
                action_pool.append(action)
                reward_pool.append(reward)

                state = next_state

        accumulate = 0
        for i in reversed(range(len(reward_pool))):
            accumulate = accumulate * args.gamma + reward_pool[i]
            reward_pool[i] = accumulate

        for i in range(len(reward_pool)):
            a2c_agent.optimize(state_pool[i], action_pool[i], reward_pool[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_folder", type=str, help="folder containing dict files")
    parser.add_argument("--bert_file", type=str, help="bert model folder")
    parser.add_argument("--sentence_file", type=str, help="file containing the sentences")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--bert_reward_device", type=int, default=0)
    parser.add_argument("--bert_feature_device", type=int, default=1)
    parser.add_argument("--training_device", type=int, default=2)
    parser.add_argument("--bert_dim", type=int, default=768)
    parser.add_argument("--position_dim", type=int, default=256)
    args = parser.parse_args()

    print("start build dict....")
    function_dict, pos_index, pos_set = build_dict(args.dict_folder)
    print("build dict done...")
    train(function_dict, pos_index, args, pos_set)