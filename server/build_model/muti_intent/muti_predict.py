"""
Created by xiedong
@Date: 2023/5/27 19:31
"""
import numpy as np
import torch
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer
from muti_config import Args
from muti_process import get_word_list


class Predictor:
    def __init__(self, model):
        self.args = Args()
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_dir)
        self.model = model

    # Predict function
    def predict(self, input_text):
        self.model.eval()
        with torch.no_grad():
            # Tokenize and convert to input IDs
            # tokens = list(jieba.cut(input_text))
            tokens = get_word_list(input_text)
            # input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # tokenizer = BertTokenizer.from_pretrained(self.args.bert_dir)
            inputs = self.tokenizer.encode_plus(
                text=tokens,
                max_length=self.args.max_len,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            input_ids = torch.tensor(inputs['input_ids'])

            input_ids = torch.tensor(input_ids).unsqueeze(0)

            # Create attention mask
            attention_mask = [1] * len(input_ids)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)

            outputs = self.model(input_ids, attention_mask)
            intent_logits, slot_logits = outputs

            intent_probs = torch.sigmoid(intent_logits).squeeze(0).tolist()
            slot_probs = torch.softmax(slot_logits, dim=2).squeeze(0).tolist()

            intent_result = [(self.args.id2seqlabel[index], value) for index, value in enumerate(intent_probs) if
                             value > self.args.muti_intent_threshold]
            # filtered_values = [value for value in my_list if value > threshold]

            # intent_idx = np.where(intent_probs > 0.5)[0]
            # intent_result = [(args.id2seqlabel[idx], intent_probs[idx]) for idx in intent_idx]

            token_output = np.argmax(slot_probs, -1)

            token_output = token_output[1:len(input_text) - 1]
            token_output = [self.args.id2tokenlabel[i] for i in token_output]

            # intent = self.config.id2seqlabel[seq_output]
            # slots = str([(i[0], text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)])
            slots = [(i[0], input_text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)]
            print('意图：', intent_result)
            # print('意图强度：', intent_confidence)
            print('槽位：', str(slots))

            # return intent_probs, slot_probs
            return intent_result, slots
