"""
Created by xiedong
@Date: 2023/4/20 15:28
"""
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

import torch.nn.functional as F
from transformers import logging

logging.set_verbosity_error()
import time


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)
        self.epoch = config.epoch
        self.device = config.device
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)

    def train(self, train_loader):
        time_start = time.time()
        global_step = 0
        total_step = len(train_loader) * self.epoch
        self.model.train()
        for epoch in range(self.epoch):
            for step, train_batch in enumerate(train_loader):
                for key in train_batch.keys():
                    train_batch[key] = train_batch[key].to(self.device)
                input_ids = train_batch['input_ids']
                attention_mask = train_batch['attention_mask']
                token_type_ids = train_batch['token_type_ids']
                seq_label_ids = train_batch['seq_label_ids']
                token_label_ids = train_batch['token_label_ids']
                seq_output, token_output = self.model(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )

                active_loss = attention_mask.view(-1) == 1
                active_logits = token_output.view(-1, token_output.shape[2])[active_loss]
                active_labels = token_label_ids.view(-1)[active_loss]

                seq_loss = self.criterion(seq_output, seq_label_ids)
                token_loss = self.criterion(active_logits, active_labels)
                loss = seq_loss + token_loss
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f'[train] epoch:{epoch + 1} {global_step}/{total_step} loss:{loss.item()}')
                global_step += 1

            if self.config.do_save:
                self.save(self.config.save_dir, str(int(time.time())) + 'model.pt')

        time_end = time.time()
        print('time cost', time_end - time_start, 's')

    def save(self, save_path, save_name):
        torch.save(self.model.state_dict(), os.path.join(save_path, save_name))

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            tmp_text = [i for i in text]
            inputs = self.tokenizer.encode_plus(
                text=tmp_text,
                max_length=self.config.max_len,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']
            seq_output, token_output = self.model(
                input_ids,
                attention_mask,
                token_type_ids,
            )

            softmaxSeq = F.softmax(seq_output, dim=1)
            numpySeq = softmaxSeq.numpy()[0]
            # 意图强度
            intent_confidence = numpySeq.max()

            seq_output = seq_output.detach().cpu().numpy()
            token_output = token_output.detach().cpu().numpy()

            seq_output = np.argmax(seq_output, -1)
            token_output = np.argmax(token_output, -1)
            print(seq_output, token_output)
            seq_output = seq_output[0]
            token_output = token_output[0][1:len(text) - 1]
            token_output = [self.config.id2tokenlabel[i] for i in token_output]

            intent = self.config.id2seqlabel[seq_output]
            # slots = str([(i[0], text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)])
            slots = [(i[0], text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)]
            print('意图：', intent)
            print('意图强度：', intent_confidence)
            print('槽位：', str(slots))

            return intent, float(intent_confidence), slots

    def get_metrices(self, trues, preds, mode):
        if mode == 'cls':
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            acc = accuracy_score(trues, preds)
            precision = precision_score(trues, preds, average='micro')
            recall = recall_score(trues, preds, average='micro')
            f1 = f1_score(trues, preds, average='micro')
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
            # 将 trues 中的标签范围从0-21转换为1-22
            trues_mapped = [label + 1 for label in trues]
            preds_mapped = [label + 1 for label in preds]

            # report = classification_report(trues, preds)
            sorted_keys = sorted(self.config.seqlabel2id, key=self.config.seqlabel2id.get)
            report = classification_report(trues_mapped, preds_mapped, target_names=sorted_keys)
            # report = classification_report(trues, preds)
        elif mode == 'ner':
            from seqeval.metrics import classification_report
            report = classification_report(trues, preds)
        return report

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
                    attention_mask,
                    token_type_ids,
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
                    t_ouput = [self.config.id2tokenlabel[i] for i in t_output]
                    t_label = [self.config.id2tokenlabel[i] for i in t_label]
                    token_preds.append(t_ouput)
                    token_trues.append(t_label)

        acc, precision, recall, f1 = self.get_metrices(seq_trues, seq_preds, 'cls')
        report = self.get_report(seq_trues, seq_preds, 'cls')
        ner_acc, ner_precision, ner_recall, ner_f1 = self.get_metrices(token_trues, token_preds, 'ner')
        ner_report = self.get_report(token_trues, token_preds, 'ner')
        print('意图识别：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            acc, precision, recall, f1
        ))
        print(report)
        print('槽位填充：\naccuracy:{}\nprecision:{}\nrecall:{}\nf1:{}'.format(
            ner_acc, ner_precision, ner_recall, ner_f1
        ))
        print(ner_report)
