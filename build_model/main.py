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
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
if __name__ == '__main__':
    # 全局配置参数
    args = Args()
    # 加载bert编码使用
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    # 加载模型
    model = BertForIntentClassificationAndSlotFilling(args)
    # 是否加载本地模型
    if args.load_model:
        model.load_state_dict(torch.load(args.load_dir))

    # CPU or GPUC
    model.to(args.device)
    # 训练器实例
    trainer = Trainer(model, args)

    # 是否训练
    if args.do_train:
        raw_examples = Processor.get_examples(Processor.read_file(args.train_texts),
                                              Processor.read_file(args.train_intents),
                                              Processor.read_file(args.train_slots), 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        train_dataset = BertDataset(train_features)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        trainer.train(train_loader)

    # 是否测试
    if args.do_test:
        # trainer.test(test_loader)
        raw_examples = Processor.get_examples(Processor.read_file(args.train_texts),
                                              Processor.read_file(args.train_intents),
                                              Processor.read_file(args.train_slots), 'train')
        train_features = get_features(raw_examples, tokenizer, args)
        train_dataset = BertDataset(train_features)
        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        trainer.test(train_loader)

    # 是否预测
    if args.do_predict:
        trainer.predict('请问二型糖尿病的临床表现是什么')
