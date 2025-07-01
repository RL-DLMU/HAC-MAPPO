import csv

import env1
import torch
import sys

from config import get_config
from shared.env_runner import EnvRunner as Runner
def parse_args(args, parser):

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    env = env1.OilDeliveryEnv1()
    station_num = env.agent_num
    config = {
        "all_args": all_args,
        "envs": env,
        "num_agents": station_num,
        "device": device,
    }
    check_path = r'D:\BaiduSyncdisk\a最新大规模\LCMAPPO工作日2\envs\model_checkpoint_180.pth'    # check_path = 'D:\\BaiduSyncdisk\\对比算法\\mappo对比算法\\envs\\model_checkpoint_20000.pth'
    #check_path = r'D:\BaiduSyncdisk\a最新大规模\LCMAPPO节假日3\envs\model_checkpoint_160.pth'    # check_path = 'D:\\BaiduSyncdisk\\对比算法\\mappo对比算法\\envs\\model_checkpoint_20000.pth'

    # check_path = 'D:\\BaiduSyncdisk\\验证实验\\LCPPO改油耗数据实验室 无disentropy\envs\\model_checkpoint_4000.pth'
    # check_path = 'D:\\BaiduSyncdisk\\对比算法\\mappo对比算法\\envs\\model_checkpoint_20000.pth'
    runner = Runner(config)
    runner.run()
    #runner.eval_news(1,check_path)
    #eval_avg_cost=runner.eval_1(1, check_path)

    # 保存 eval_cost 到文件
    # cost_file_path = 'LCMAPPO3-1.csv'
    # with open(cost_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['eval_cost'])  # 列名
    #     for cost in eval_avg_cost:
    #         writer.writerow([cost])
    #
    # print(f"数据已分别保存到  {cost_file_path}")
    #
    # post process
    env.close()

if __name__ == '__main__':
        main(sys.argv[1:])
