import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import logging
import torch.optim as optim

from muti_server.models.muti_config import ModelConfig
from muti_dataset import BertDataset, SLUDataset
from muti_server.models.muti_model import MultiJointModel
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


def train_joint_bert():
    global device, epoch, input_ids, attention_mask, token_type_ids, seq_label_ids, token_label_ids
    #
    args = ModelConfig()
    device = args.device
    num_intents = args.seq_num_labels
    num_slots = args.token_num_labels
    model = MultiJointModel(num_intents=num_intents, num_slots=num_slots)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_dir))
    model.to(device)
    # criterion_intent = nn.BCEWithLogitsLoss()
    criterion_intent = nn.BCELoss()
    criterion_slot = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Training and testing
    num_epochs = args.epoch
    train_dataset = BertDataset('train')
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    global_step = 0
    total_step = len(train_loader) * num_epochs
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0

        print(f"time:{getLastDate()} Epoch {epoch + 1}/{num_epochs} start")
        # for b, train_batch in enumerate(train_loader):
        for b, train_batch in enumerate(train_loader):
            for key in train_batch.keys():
                train_batch[key] = train_batch[key].to(device)

            input_ids = train_batch['input_ids']
            attention_mask = train_batch['attention_mask']
            token_type_ids = train_batch['token_type_ids']
            seq_label_ids = train_batch['seq_label_ids']
            token_label_ids = train_batch['token_label_ids']

            outputs = model(input_ids, attention_mask)
            seq_output, token_output = outputs

            # intent_loss = criterion_intent(intent_logits, seq_label_ids.float())
            # if args.use_crf:
            #     slot_loss = model.crf(slot_logits, token_label_ids, mask=attention_mask.byte(), reduction='mean')
            #     slot_loss = -1 * slot_loss  # negative log-likelihood
            # else:
            #     # slot_loss = criterion_slot(slot_logits.view(-1, model.slot_filler.out_features),
            #     #                            token_label_ids.view(-1))
            #
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = slot_logits.view(-1, slot_logits.shape[2])[active_loss]
            #     active_labels = token_label_ids.view(-1)[active_loss]
            #     #
            #     slot_loss = criterion_slot(active_logits, active_labels)
            #
            # # active_loss = attention_mask.view(-1) == 1
            # # active_logits = slot_logits.view(-1, slot_logits.shape[2])[active_loss]
            # # active_labels = token_label_ids.view(-1)[active_loss]
            # #
            # # slot_loss = criterion_slot(active_logits, active_labels)
            #
            # total_loss = intent_loss + slot_loss

            active_loss = attention_mask.view(-1) == 1
            active_logits = token_output.view(-1, token_output.shape[2])[active_loss]
            active_labels = token_label_ids.view(-1)[active_loss]

            total_loss = model.calculate_loss(seq_output, active_logits, seq_label_ids, active_labels)

            # optimizer.zero_grad()
            model.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss = total_loss.item()

            print(f'[train] epoch:{epoch + 1} {global_step}/{total_step} loss:{total_loss.item()}')
            global_step += 1

        print(f"time:{getLastDate()} Epoch {epoch + 1}/{num_epochs} train end")
        print(f"time:{getLastDate()} Train Loss: {train_loss:.4f}")
        print()

        if args.do_save:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, str(int(time.time())) + '_' + str(epoch) + '_muti_model.pt'))

        # Predict
        predictor = Predictor(model)
        input_text = "请问二型糖尿病的临床表现是什么,需要吃什么药啊"
        intent_probs, slot_probs = predictor.predict(input_text)

        print("Intent probabilities:", sorted(intent_probs, key=lambda x: x[1]))
        print("Slot probabilities:", slot_probs)

        input_text = "你好"
        intent_probs, slot_probs = predictor.predict(input_text)

        print(input_text + "Intent probabilities:", sorted(intent_probs, key=lambda x: x[1]))
        print(input_text + "Slot probabilities:", slot_probs)

        input_text = "再见"
        intent_probs, slot_probs = predictor.predict(input_text)

        print(input_text + "Intent probabilities:", sorted(intent_probs, key=lambda x: x[1]))
        print(input_text + "Slot probabilities:", slot_probs)


# def train_jointslu():
#     global epoch, input_ids, attention_mask, token_type_ids, seq_label_ids, token_label_ids
#     args = ModelConfig()
#     # 数据预处理
#     # data = [[1, 2, 3, 4], [2, 3, 5, 6]]
#     # intent_labels = [[0, 1, 0], [1, 0, 1]]  # 多标签意图
#     # slot_labels = [[1, 2, 3, 4], [5, 6, 0, 0]]
#     train_data = SLUDataset()
#     train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
#     # 定义模型参数
#     num_intents = args.seq_num_labels
#     num_slots = args.token_num_labels
#     embedding_dim = 100
#     hidden_dim = 200
#     num_embeddings = 10000
#     # 创建模型实例
#     model = JointSLU(num_intents, num_slots, embedding_dim, hidden_dim, num_embeddings)
#     # 定义损失函数和优化器
#     criterion_intent = nn.BCEWithLogitsLoss()  # 多标签意图使用 BCEWithLogitsLoss
#     criterion_slot = nn.CrossEntropyLoss(ignore_index=0)
#     optimizer = optim.Adam(model.parameters())
#     # 训练模型
#     for epoch in range(10):
#         for b, train_batch in enumerate(train_loader):
#             input_ids = train_batch['input_ids']
#             attention_mask = train_batch['attention_mask']
#             token_type_ids = train_batch['token_type_ids']
#             seq_label_ids = train_batch['seq_label_ids']
#             token_label_ids = train_batch['token_label_ids']
#
#             optimizer.zero_grad()
#
#             intent_logits, slot_logits = model(input_ids)
#
#             intent_loss = criterion_intent(intent_logits, seq_label_ids.float())
#             slot_loss = criterion_slot(slot_logits.view(-1, num_slots), token_label_ids.view(-1))
#             total_loss = intent_loss + slot_loss
#
#             total_loss.backward()
#             optimizer.step()
#
#             print(f'[train] epoch:{epoch + 1} loss:{total_loss.item()}')
# 预测代码示例
# input_seq = torch.tensor([[1, 2, 3, 4]])
# intent_logits, slot_logits = model(input_seq)
# predicted_intent = torch.sigmoid(intent_logits) > 0.5  # 根据阈值进行判断
# _, predicted_slot = torch.max(slot_logits, 2)
# print("Predicted Intent:", predicted_intent.squeeze().tolist())
# print("Predicted Slot:", predicted_slot.squeeze().tolist())


if __name__ == '__main__':
    train_joint_bert()
    # train_jointslu()
