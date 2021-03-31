import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np 
import time
import random
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# from apex import amp
from tqdm import tqdm
from tqdm import trange
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
# from dataset import *
# from model import *

def log_loss(step_record, loss_record):
    if not os.path.exists("../img"):
        os.mkdir("../img")
    plt.plot(step_record, loss_record, lw=2)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join("../img", 'loss_curve.png'))
    plt.close()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str, default="4", help="gpu id")
    parser.add_argument("--lr", dest="lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, default=32, help="batch size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, default=3, help="max epoch number")
    
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.3, help="true entity(not 'BLANK') proportion")
    parser.add_argument("--model", dest="model", type=str, default="", help="{MTB, CP}")
    parser.add_argument("--train_sample", action="store_true", help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int, default=64, help="max sentence length")
    parser.add_argument("--bag_size", dest="bag_size", type=int, default=2, help="bag size")
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.05, help="temperature for NTXent loss")
    parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=768, help="hidden size for mlp")

    parser.add_argument("--weight_decay", dest="weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int, default=500, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float, default=1, help="max grad norm")

    parser.add_argument("--save_step", dest="save_step", type=int, default=10000, help="step to save")
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="", help="ckpt dir to save")

    parser.add_argument("--seed", dest="seed", type=int, default=42, help="seed for network")
    parser.add_argument("--local_rank", dest="local_rank", type=int, default=-1, help="local rank")
    args = parser.parse_args()

    print('-------args-------')
    for i in list(vars(args).keys()):
        print(f"{k}: {vars(args)[i]}")
    print('-------args-------\n')
