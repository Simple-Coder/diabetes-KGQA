import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from muti_config import Args
from muti_model import MutiJointModel
from muti_dataset import BertDataset
from muti_process import get_word_list
from transformers import BertTokenizer
from transformers import logging
from seqeval.metrics.sequence_labeling import get_entities

logging.set_verbosity_error()


# Test the model
def test_model(model, input_ids, attention_mask, intent_labels, slot_labels, criterion_intent, criterion_slot):
    model.eval()
    outputs = model(input_ids, attention_mask)
    intent_logits, slot_logits = outputs

    intent_loss = criterion_intent(intent_logits, intent_labels.float())
    slot_loss = criterion_slot(slot_logits.view(-1, model.slot_filler.out_features), slot_labels.view(-1))

    total_loss = intent_loss + slot_loss

    return total_loss.item()


# Predict function
def predict(model, input_text):
    model.eval()
    with torch.no_grad():
        args = Args()
        # Tokenize and convert to input IDs
        # tokens = list(jieba.cut(input_text))
        tokens = get_word_list(input_text)
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        inputs = tokenizer.encode_plus(
            text=tokens,
            max_length=args.max_len,
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

        outputs = model(input_ids, attention_mask)
        intent_logits, slot_logits = outputs

        intent_probs = torch.sigmoid(intent_logits).squeeze(0).tolist()
        slot_probs = torch.softmax(slot_logits, dim=2).squeeze(0).tolist()

        filtered_indices = [(args.id2seqlabel[index],value) for index, value in enumerate(intent_probs) if value > 0.5]
        # filtered_values = [value for value in my_list if value > threshold]



        # intent_idx = np.where(intent_probs > 0.5)[0]
        # intent_result = [(args.id2seqlabel[idx], intent_probs[idx]) for idx in intent_idx]

        token_output = np.argmax(slot_probs, -1)

        token_output = token_output[1:len(input_text) - 1]
        token_output = [args.id2tokenlabel[i] for i in token_output]

        # intent = self.config.id2seqlabel[seq_output]
        # slots = str([(i[0], text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)])
        slots = [(i[0], input_text[i[1]:i[2] + 1], i[1], i[2]) for i in get_entities(token_output)]
        print('意图：', filtered_indices)
        # print('意图强度：', intent_confidence)
        print('槽位：', str(slots))

        return intent_probs, slot_probs


if __name__ == '__main__':
    args = Args()
    device = args.device
    num_intents = args.seq_num_labels
    num_slots = args.token_num_labels

    model = MutiJointModel(num_intents=num_intents, num_slots=num_slots).to(device)
    criterion_intent = nn.BCEWithLogitsLoss()
    criterion_slot = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training and testing
    num_epochs = 100

    train_dataset = BertDataset('train')
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    for epoch in range(num_epochs):
        train_loss = 0
        for b, train_batch in enumerate(train_loader):
            input_ids = train_batch['input_ids']
            attention_mask = train_batch['attention_mask']
            token_type_ids = train_batch['token_type_ids']
            seq_label_ids = train_batch['seq_label_ids']
            token_label_ids = train_batch['token_label_ids']

            outputs = model(input_ids, attention_mask)
            intent_logits, slot_logits = outputs

            intent_loss = criterion_intent(intent_logits, seq_label_ids.float())
            slot_loss = criterion_slot(slot_logits.view(-1, model.slot_filler.out_features), token_label_ids.view(-1))

            total_loss = intent_loss + slot_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss = total_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print()

    # Predict
    input_text = "请问二型糖尿病的临床表现是什么,需要吃什么药啊"
    intent_probs, slot_probs = predict(model, input_text)

    print("Intent probabilities:", intent_probs)
    print("Slot probabilities:", slot_probs)
