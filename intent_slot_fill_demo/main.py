"""
Created by xiedong
@Date: 2023/4/9 16:37
"""
import argparse
from intent_slot_fill_demo.utils import init_logger, set_seed, load_tokenizer, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples
from trainer import Trainer

def main(args):
    # 初始化日志记录
    init_logger()
    # 设置随机数种子
    set_seed(args)

    # 加载tokenizer
    tokenizer = load_tokenizer(args)

    # 加载对应数据集
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    # 训练
    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")
    print()


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()

    # 用于区分是什么数据集
    parser.add_argument("--task", default=None, required=True, type=str, help="区分数据集")
    # 加载的模型文件
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="模型保存与加载路径")
    # 输入的数据入口文件
    parser.add_argument("--data_dir", default="./data", type=str, help="输入文件路径")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="意图标签文件")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="槽位标签文件")
    parser.add_argument("--model_type", default="bert", type=str, help="模型类型，默认使用bert")
    # 随机种子
    parser.add_argument('--seed', type=int, default=1234, help="随机种子")
    parser.add_argument("--train_batch_size", default=32, type=int, help="训练数据批处理大小")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="评估时批数据大小")
    parser.add_argument("--max_seq_len", default=50, type=int, help="最大句子长度")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="初始学习率")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="训练轮数")

    # 权重延迟
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str,
                        help="Pad token for slot label pad (to be ignore when calculate loss)")

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
