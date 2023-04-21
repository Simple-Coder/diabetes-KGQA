"""
Created by xiedong
@Date: 2023/4/20 15:28
"""
from config import Args
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertForIntentClassificationAndSlotFilling
from trainer import Trainer
from process import Processor, get_features
from dataset import BertDataset

if __name__ == '__main__':

    args = Args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

    model = BertForIntentClassificationAndSlotFilling(args)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_dir))

    model.to(args.device)
    trainer = Trainer(model, args)

    if args.do_train:
        raw_examples = Processor.get_examples(args.train_texts, args.train_intents, args.train_slots, 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        train_dataset = BertDataset(train_features)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        trainer.train(train_loader)
