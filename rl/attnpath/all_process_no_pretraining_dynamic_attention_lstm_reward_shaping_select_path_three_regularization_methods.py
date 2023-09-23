import argparse
import datetime
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


def get_save_file_header(args):
    time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    save_file_header = '_'.join([
        time_str,
        'a', str(args.attn_dim),
        'wr', str(args.wrong_reward),
        'ur', str(args.useless_reward),
        'tr', str(args.teacher_reward),
        'gw', str(args.global_reward_weight),
        'lw', str(args.length_reward_weight),
        'dw', str(args.diverse_reward_weight),
        'np', str(args.num_episodes),
        'wd', str(args.weight_decay),
        'd2', str(args.dropout2),
        'adr', str(args.action_dropout_rate),
        'eb', "%.4f" % args.exp_base,
        'hue', str(args.hidden_update_everytime),
        'mo', str(args.model),
        'remo', str(args.reward_shaping_model),
        'no_pretraining_dynamic_attention_lstm_reward_shaping_select_path_three_regularization_methods', ])

    return save_file_header


def main():
    # 1、获取启动参数
    args = get_args()
    print("Relation: ", args.relation)


if __name__ == '__main__':
    main()
