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
import torch.nn as nn


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
        self.criterion_intent = nn.BCELoss()
        self.criterion_slot = nn.CrossEntropyLoss()

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
                intent_labels = train_batch['seq_label_ids']
                slot_labels = train_batch['token_label_ids']

                outputs = self.model(input_ids, attention_mask)
                intent_output, slot_output = outputs

                # 计算意图分类损失
                intent_loss = self.criterion_intent(intent_output, intent_labels.float())

                # 计算Slot填充损失
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_output.view(-1, slot_output.shape[2])[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]
                slot_loss = self.criterion_slot(active_logits, active_labels)

                total_loss = intent_loss + slot_loss

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
            if self.model_config.do_predict:
                self.predict("你好")
                self.predict("再见")
                self.predict("不是")
                self.predict("请问糖尿病的临床表现是什么")
                self.predict("请问糖尿病的临床表现是什么，需要吃什么药")

    def predict(self, input_text):
        self.model.eval()
        with torch.no_grad():
            tokens = get_word_list(input_text)
            inputs = self.tokenizer.encode_plus(
                text=tokens,
                max_length=self.model_config.max_len,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,
                return_token_type_ids=True,
            )

            input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(self.device)

            intent_output, slot_output = self.model(input_ids, attention_mask)
            intent_probs = intent_output.squeeze().tolist()

            intent_result = [(self.model_config.id2seqlabel[index], value) for index, value in enumerate(intent_probs)
                             if value > self.model_config.muti_intent_threshold]

            intent_result = sorted(intent_result, key=lambda x: x[1], reverse=True)

            # 处理槽位
            token_output = slot_output.detach().cpu().numpy()
            token_output = np.argmax(token_output, -1)
            # token_output = token_output[0, 1:len(input_text) - 1]
            token_output = token_output[0, 1:len(tokens) - 1]
            token_output = [self.model_config.id2tokenlabel[i] for i in token_output]

            # intent = self.config.id2seqlabel[seq_output]
            # slots = str([(i[0], text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)])
            slots = [(i[0], input_text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)]
            print(input_text + '：意图：', intent_result)
            print(input_text + '槽位：', str(slots))

            # return intent_probs, slot_probs
            return intent_result, slots

    def test(self, test_loader):
        self.model.eval()
        seq_preds = []
        seq_trues = []
        token_preds = []
        token_trues = []
        with torch.no_grad():
            for step, test_batch in enumerate(test_loader):
                for key in test_batch.keys():
                    test_batch[key] = test_batch[key].to(self.device)
                input_ids = test_batch['input_ids']
                attention_mask = test_batch['attention_mask']
                token_type_ids = test_batch['token_type_ids']
                seq_label_ids = test_batch['seq_label_ids']
                token_label_ids = test_batch['token_label_ids']
                seq_output, token_output = self.model(
                    input_ids,
                    attention_mask
                )
                seq_output = seq_output.detach().cpu().numpy()
                seq_output = np.argmax(seq_output, -1)
                seq_label_ids = seq_label_ids.detach().cpu().numpy()
                seq_label_ids = seq_label_ids.reshape(-1)
                seq_preds.extend(seq_output)
                seq_trues.extend(seq_label_ids)

                token_output = token_output.detach().cpu().numpy()
                token_label_ids = token_label_ids.detach().cpu().numpy()
                token_output = np.argmax(token_output, -1)
                active_len = torch.sum(attention_mask, -1).view(-1)
                for length, t_output, t_label in zip(active_len, token_output, token_label_ids):
                    t_output = t_output[1:length - 1]
                    t_label = t_label[1:length - 1]
                    t_ouput = [self.model_config.id2tokenlabel[i] for i in t_output]
                    t_label = [self.model_config.id2tokenlabel[i] for i in t_label]
                    token_preds.append(t_ouput)
                    token_trues.append(t_label)

        # acc, precision, recall, f1 = self.get_metrices(seq_trues, seq_preds, 'cls')
        # report = self.get_report(seq_trues, seq_preds, 'cls')
        ner_acc, ner_precision, ner_recall, ner_f1 = self.get_metrices(token_trues, token_preds, 'ner')
        ner_report = self.get_report(token_trues, token_preds, 'ner')
        # print('意图识别：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
        #     acc, precision, recall, f1
        # ))
        # print(report)
        print('槽位填充：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            ner_acc, ner_precision, ner_recall, ner_f1
        ))
        print(ner_report)

    def get_metrices(self, trues, preds, mode):
        if mode == 'cls':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            precision = precision_score(trues, preds, average='micro')
            recall = recall_score(trues, preds, average='micro')
            f1 = f1_score(trues, preds, average='micro')
            return acc, precision, recall, f1
        elif mode == 'ner':
            from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            precision = precision_score(trues, preds)
            recall = recall_score(trues, preds)
            f1 = f1_score(trues, preds)
            return acc, precision, recall, f1

    def get_report(self, trues, preds, mode):
        if mode == 'cls':
            from sklearn.metrics import classification_report
            report = classification_report(trues, preds)
            return report
        elif mode == 'ner':
            from seqeval.metrics import classification_report
            report = classification_report(trues, preds)
            return report
