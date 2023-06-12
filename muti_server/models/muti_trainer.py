"""
Created by xiedong
@Date: 2023/6/12 20:53
"""
import datetime
import time
import os
from muti_server.models.muti_model import MultiJointModel
from seqeval.metrics.sequence_labeling import get_entities
from muti_server.models.muti_process import get_word_list
import torch
import numpy as np
from transformers import BertTokenizer


class Trainer:
    def __init__(self, model_config):
        self.model_config = model_config
        self.num_intents = self.model_config.seq_num_labels
        self.num_slots = self.model_config.token_num_labels
        self.device = self.model_config.device
        self.model = MultiJointModel(num_intents=self.num_intents, num_slots=self.num_slots)
        if self.model_config.load_model:
            self.model.load_state_dict(torch.load(self.model_config.load_dir))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config.lr)
        self.num_epochs = self.model_config.epoch
        self.tokenizer = BertTokenizer.from_pretrained(self.model_config.bert_dir)

    # 获取时间函数，把当前时间格式化为str类型nowdate.strftime('%Y-%m-%d %H:%M:%S')
    def get_last_date(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def train(self, train_loader):
        global_step = 0
        total_step = len(train_loader) * self.num_epochs
        self.model.train()
        for epoch in range(self.num_epochs):
            train_loss = 0

            print(f"time:{self.get_last_date()} Epoch {epoch + 1}/{self.num_epochs} start")
            # for b, train_batch in enumerate(train_loader):
            # 遍历训练数据加载器中的每个批次
            for b, train_batch in enumerate(train_loader):
                for key in train_batch.keys():
                    train_batch[key] = train_batch[key].to(self.device)

                input_ids = train_batch['input_ids']
                attention_mask = train_batch['attention_mask']
                token_type_ids = train_batch['token_type_ids']
                seq_label_ids = train_batch['seq_label_ids']
                token_label_ids = train_batch['token_label_ids']

                outputs = self.model(input_ids, attention_mask)
                seq_output, token_output = outputs
                # total_loss = intent_loss + slot_loss

                active_loss = attention_mask.view(-1) == 1
                active_logits = token_output.view(-1, token_output.shape[2])[active_loss]
                active_labels = token_label_ids.view(-1)[active_loss]

                total_loss = self.model.calculate_loss(seq_output, active_logits, seq_label_ids, active_labels)

                # optimizer.zero_grad()
                # 梯度清零：因为在每次反向传播时，梯度值会累积，如果不清零，梯度值将会一直累加，导致参数更新不正确
                self.model.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                train_loss = total_loss.item()

                print(f'[train] epoch:{epoch + 1} {global_step}/{total_step} loss:{total_loss.item()}')
                global_step += 1

            print(f"time:{self.get_last_date()} Epoch {epoch + 1}/{self.num_epochs} train end")
            print(f"time:{self.get_last_date()} Train Loss: {train_loss:.4f}")
            print()
            if self.model_config.do_save:
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_config.save_dir,
                                        str(int(time.time())) + '_' + str(epoch) + '_muti_model.pt'))

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
            print('意图：', intent_result)
            print('槽位：', str(slots))

            # return intent_probs, slot_probs
            return intent_result, slots
