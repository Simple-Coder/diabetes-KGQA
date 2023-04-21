"""
Created by xiedong
@Date: 2023/4/20 15:28
"""
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer
import numpy as np


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
            self.save(self.config.save_dir, 'model.pt')

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
            seq_output = seq_output.detach().cpu().numpy()
            token_output = token_output.detach().cpu().numpy()
            seq_output = np.argmax(seq_output, -1)
            token_output = np.argmax(token_output, -1)
            print(seq_output, token_output)
            seq_output = seq_output[0]
            token_output = token_output[0][1:len(text) - 1]
            # token_output = [self.config.id2nerlabel[i] for i in token_output]
            print('意图：', self.config.id2seqlabel[seq_output])
            # print('槽位：', str([(i[0], text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)]))
