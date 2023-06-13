from torch.utils.data import DataLoader
from transformers import logging

from muti_dataset import BertDataset
from muti_server.models.muti_config import ModelConfig
from muti_server.models.muti_trainer import Trainer

logging.set_verbosity_error()

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
