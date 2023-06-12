"""
Created by xiedong
@Date: 2023/6/7 12:35
"""
import numpy as np
import torch
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer
from muti_server.models.muti_config import ModelConfig
from muti_server.models.muti_process import get_word_list
from muti_server.models.muti_model import MultiJointModel
from muti_server.utils.logger_conf import my_log
from transformers import logging

log = my_log.logger

logging.set_verbosity_error()


class MutiPredictWrapper:
    def __init__(self, args):
        self.args = args
        # 加载模型
        self.model = MultiJointModel(self.args.seq_num_labels, self.args.token_num_labels)
        # 是否加载本地模型
        self.model.load_state_dict(torch.load(self.args.load_dir))

        #
        self.predictor = Predictor(self.model)

    def predict(self, text):
        log.info("[model] predict text:{}".format(text))
        return self.predictor.predict(text)


class Predictor:
    def __init__(self, model):
        self.model_config = ModelConfig()
        self.tokenizer = BertTokenizer.from_pretrained(self.model_config.bert_dir)
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
                max_length=self.model_config.max_len,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            input_ids = torch.tensor(inputs['input_ids'])

            # input_ids = torch.tensor(input_ids).unsqueeze(0)
            input_ids = input_ids.unsqueeze(0)

            # Create attention mask
            attention_mask = [1] * len(input_ids)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)

            outputs = self.model(input_ids, attention_mask)
            intent_logits, slot_logits = outputs

            intent_probs = torch.sigmoid(intent_logits).squeeze(0).tolist()
            slot_probs = torch.softmax(slot_logits, dim=2).squeeze(0).tolist()

            intent_result = [(self.model_config.id2seqlabel[index], value) for index, value in enumerate(intent_probs)
                             if
                             value > self.model_config.muti_intent_threshold]

            intent_result = sorted(intent_result, key=lambda x: x[1], reverse=True)
            # filtered_values = [value for value in my_list if value > threshold]

            # intent_idx = np.where(intent_probs > 0.5)[0]
            # intent_result = [(args.id2seqlabel[idx], intent_probs[idx]) for idx in intent_idx]

            token_output = np.argmax(slot_probs, -1)

            token_output = token_output[1:len(input_text) - 1]
            token_output = [self.model_config.id2tokenlabel[i] for i in token_output]

            # intent = self.config.id2seqlabel[seq_output]
            # slots = str([(i[0], text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)])
            slots = [(i[0], input_text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)]
            # print('意图：', intent_result)
            # print('意图强度：', intent_confidence)
            # print('槽位：', str(slots))

            # return intent_probs, slot_probs
            return intent_result, slots


if __name__ == '__main__':
    args = ModelConfig()
    predictor = MutiPredictWrapper(args)
    input_text = "请问二型糖尿病的临床表现是什么,需要吃什么药啊"
    intent_probs, slot_probs = predictor.predict(input_text)

    print("Intent probabilities:", sorted(intent_probs, key=lambda x: x[1]))
    print("Slot probabilities:", slot_probs)
