import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import logging

from muti_config import Args
from muti_dataset import BertDataset
from muti_model import MutiJointModel
from muti_predict import Predictor

logging.set_verbosity_error()

# Test the model
# def test_model(model, input_ids, attention_mask, intent_labels, slot_labels, criterion_intent, criterion_slot):
#     model.eval()
#     outputs = model(input_ids, attention_mask)
#     intent_logits, slot_logits = outputs
#
#     intent_loss = criterion_intent(intent_logits, intent_labels.float())
#     slot_loss = criterion_slot(slot_logits.view(-1, model.slot_filler.out_features), slot_labels.view(-1))
#
#     total_loss = intent_loss + slot_loss
#
#     return total_loss.item()
import datetime


# 获取时间函数，把当前时间格式化为str类型nowdate.strftime('%Y-%m-%d %H:%M:%S')
def getLastDate():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    #
    args = Args()
    device = args.device
    num_intents = args.seq_num_labels
    num_slots = args.token_num_labels

    model = MutiJointModel(num_intents=num_intents, num_slots=num_slots).to(device)
    criterion_intent = nn.BCEWithLogitsLoss()
    criterion_slot = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training and testing
    num_epochs = args.epoch

    train_dataset = BertDataset('train')
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    for epoch in range(num_epochs):
        train_loss = 0
        print(f"time:{getLastDate()} Epoch {epoch + 1}/{num_epochs} start")
        for b, train_batch in enumerate(train_loader):
            input_ids = train_batch['input_ids'].to(device)
            attention_mask = train_batch['attention_mask']
            token_type_ids = train_batch['token_type_ids']
            seq_label_ids = train_batch['seq_label_ids']
            token_label_ids = train_batch['token_label_ids']

            outputs = model(input_ids, attention_mask)
            intent_logits, slot_logits = outputs

            intent_loss = criterion_intent(intent_logits, seq_label_ids.float())
            # slot_loss = criterion_slot(slot_logits.view(-1, model.slot_filler.out_features), token_label_ids.view(-1))

            active_loss = attention_mask.view(-1) == 1
            active_logits = slot_logits.view(-1, slot_logits.shape[2])[active_loss]
            active_labels = token_label_ids.view(-1)[active_loss]

            slot_loss = criterion_slot(active_logits, active_labels)

            total_loss = intent_loss + slot_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss = total_loss.item()

        print(f"time:{getLastDate()} Epoch {epoch + 1}/{num_epochs}")
        print(f"time:{getLastDate()} Train Loss: {train_loss:.4f}")
        print()

        if args.do_save:
            torch.save(model.state_dict(), os.path.join(args.save_dir, str(int(time.time())) + 'muti_model.pt'))

        # Predict
        predictor = Predictor(model)
        input_text = "请问二型糖尿病的临床表现是什么,需要吃什么药啊"
        intent_probs, slot_probs = predictor.predict(input_text)

        print("Intent probabilities:", intent_probs)
        print("Slot probabilities:", slot_probs)

        input_text = "你好"
        intent_probs, slot_probs = predictor.predict(input_text)

        print(input_text + "Intent probabilities:", intent_probs)
        print(input_text + "Slot probabilities:", slot_probs)

        input_text = "再见"
        intent_probs, slot_probs = predictor.predict(input_text)

        print(input_text + "Intent probabilities:", intent_probs)
        print(input_text + "Slot probabilities:", slot_probs)
