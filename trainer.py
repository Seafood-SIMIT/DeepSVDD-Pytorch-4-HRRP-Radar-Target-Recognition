

from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader
from utils.deepSVDD import DeepSVDD

import os
import time
import logging
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='',help="Root directory of run")
    parser.add_argument('-c', '--config', type=str, required=True,help="yaml file for configuration")
    parser.add_argument('--checkpoint_path', type=str, default=None,help="path of checkpoint pt file")
    parser.add_argument('-m', '--model', type=str, required=True,
                       help="Name of the model. Used for both logging and saving checkpoints.")
    args = parser.parse_args()

    hp = HParam(args.config)

    pt_dir = os.path.join(args.base_dir, hp.log.chkpt_dir, args.model)
    os.makedirs(pt_dir, exist_ok=True)

    log_dir = os.path.join(args.base_dir, hp.log.log_dir, args.model)
    if os.path.exists(log_dir):
        del_list = os.listdir(log_dir)
        for del_f in del_list:
            file_path = os.path.join(log_dir, del_f)
            os.remove(file_path)
        
    else:
        os.makedirs(log_dir, exist_ok=True)

    chkpt_pth = args.checkpoint_path if args.checkpoint_path is not None else None
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
            ]
        )
    logger = logging.getLogger()

    #输入路径检查
    if hp.data.train_dir == '':
        logger.error("train_dir, test_dir cannot be empty.")
        raise Exception("Please specify directories of data in %s" % args.config)

    writer = MyWriter(hp, log_dir)

    # Load data
    trainloader, testloader = create_dataloader(hp,logger,args)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(hp.deepsvdd.objective, hp.deepsvdd.nu)
    deep_SVDD.set_network()

    train(args, pt_dir, chkpt_pth, trainloader, testloader, writer, logger, hp, deep_SVDD)