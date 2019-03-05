from config import cfg
from modeling.afm import AFM

import argparse
import os
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch Line Segment Detection Training')

    parser.add_argument("--config-file",
        metavar = "FILE",
        help = "path to config file",
        type=str,
        required=True,
    )

    parser.add_argument("--gpu", type=int, default = 0)

    parser.add_argument("--start_epoch", dest="epoch", default=-1, type=int)


    parser.add_argument("opts",
        help="Modify config options using the command-line",
        default = None,
        nargs = argparse.REMAINDER
    )

    

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)



    system = AFM(cfg)
    system.train(cfg, args.epoch)

    
