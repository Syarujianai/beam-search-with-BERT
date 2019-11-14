import torch
import torch.nn.functional as F
import math
from transformers import BertTokenizer, BertModel, BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained("./bert/models/")
# print(tokenized_text)
model = BertForMaskedLM.from_pretrained("./bert/models/")
model.eval()
model.to('cuda')


def get_sentence_probability(sentence):
    input_sequence = "[CLS]"
    result = 0
    for i in range(len(sentence)):
        token = sentence[i]
        token_id = tokenizer.convert_tokens_to_ids([token])[0]
        input_sequence_with_mask = input_sequence + "[MASK]"
        tokenized_text = tokenizer.tokenize(input_sequence_with_mask)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [0 for _ in range(len(tokenized_text))]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segment_ids])

        tokens_tensor = tokens_tensor.to("cuda")
        segments_tensors = segments_tensors.to("cuda")

        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = F.softmax(outputs[0], dim=-1)
            # print(predictions.shape)
            target_probability = math.log(predictions[0, len(tokenized_text) - 1, token_id].item())
            # print(target_probability)
            result += target_probability
        input_sequence += token
    return result


original_sentence = "他把人给杀了"
revised_sentence = "他被人给杀了"
original_likelihood = get_sentence_probability(original_sentence)
revised_likelihood = get_sentence_probability(revised_sentence)
print("original likelihood", original_likelihood)
print("revised likelihood", revised_likelihood)