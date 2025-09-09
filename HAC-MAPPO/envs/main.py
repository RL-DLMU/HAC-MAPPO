import csv
import numpy as np
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
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    env = env1.OilDeliveryEnv1()
    station_num = env.agent_num
    config = {
        "all_args": all_args,
        "envs": env,
        "num_agents": station_num,
        "device": device,
    }
   

  
    runner = Runner(config)
    runner.run()

    env.close()

if __name__ == '__main__':
        main(sys.argv[1:])
