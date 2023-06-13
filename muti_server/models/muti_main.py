from torch.utils.data import DataLoader
from transformers import logging

from muti_dataset import BertDataset
from muti_server.models.muti_config import ModelConfig
from muti_server.models.muti_trainer import Trainer

logging.set_verbosity_error()


def get_metrices(trues, preds, mode):
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


def get_report(trues, preds, mode):
    if mode == 'cls':
        from sklearn.metrics import classification_report
        report = classification_report(trues, preds)
        return report
    elif mode == 'ner':
        from seqeval.metrics import classification_report
        report = classification_report(trues, preds)
        return report


if __name__ == '__main__':
    model_config = ModelConfig()
    trainer = Trainer(model_config)
    if model_config.do_train:
        train_dataset = BertDataset('train')
        train_loader = DataLoader(train_dataset, batch_size=model_config.batchsize, shuffle=True)
        trainer.train(train_loader)

    if model_config.do_predict:
        trainer.predict("请问二型糖尿病的临床表现是什么,需要吃什么药啊")
        trainer.predict("请问二型糖尿病的临床表现是什么")
        trainer.predict("你好")
        trainer.predict("再见")
        trainer.predict("是的")
        trainer.predict("不是")
