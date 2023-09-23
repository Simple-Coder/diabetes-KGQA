import argparse
import math


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-a', '--attn_dim', type=int, default=100)
    argparser.add_argument('-wr', '--wrong_reward', type=float, default=-0.05)
    argparser.add_argument('-ur', '--useless_reward', type=float, default=0.01)
    argparser.add_argument('-tr', '--teacher_reward', type=float, default=1.0)
    argparser.add_argument('-gw', '--global_reward_weight', type=float, default=0.1)
    argparser.add_argument('-lw', '--length_reward_weight', type=float, default=0.8)
    argparser.add_argument('-dw', '--diverse_reward_weight', type=float, default=0.1)
    argparser.add_argument('-np', '--num_episodes', type=int, default=500)
    argparser.add_argument('-wd', '--weight_decay', type=float, default=0.005)
    argparser.add_argument('-d2', '--dropout2', type=float, default=0.3)
    argparser.add_argument('-adr', '--action_dropout_rate', type=float, default=0.3)
    argparser.add_argument('-eb', '--exp_base', type=float, default=math.e)
    argparser.add_argument('-r', '--relation', type=str, default='concept_agentbelongstoorganization')
    argparser.add_argument('-t', '--task', type=str, default='retrain')
    argparser.add_argument('-hue', '--hidden_update_everytime', type=int, default=0)
    argparser.add_argument('-mo', '--model', type=str, default="TransD")
    argparser.add_argument('-remo', '--reward_shaping_model', type=str, default="ConvE")

    return argparser.parse_args()
    # args = argparser.parse_args()
    # print("Relation: ", args.relation)


def main():
    # 1、获取启动参数
    args = get_args()
    print("Relation: ", args.relation)


if __name__ == '__main__':
    main()
