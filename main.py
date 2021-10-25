import argparse

import torch
import wandb
wandb.login()

from dataloader import get_dataloaders
from utils import get_model
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='[c10, c100, svhn]')
parser.add_argument('--model', required=True, help='[mlp_mixer, ]')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--eval-batch-size', type=int, default=1024)
parser.add_argument('--num-workers', type=int, default=4)

parser.add_argument('--patch-size', type=int, default=4)
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--hidden-c', type=int, default=2048)
parser.add_argument('--hidden-s', type=int, default=256)
parser.add_argument('--num-layers', type=int, default=8)
parser.add_argument('--drop-p', type=int, default=0.)

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--off-nesterov', action='store_true')
parser.add_argument('--label-smoothing', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
args = parser.parse_args()
args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.nesterov = not args.off_nesterov



if __name__=='__main__':
    with wandb.init(project='mlp_mixer', config=args):
        train_dl, test_dl = get_dataloaders(args)
        model = get_model(args)
        trainer = Trainer(model, args)
        trainer.fit(train_dl, test_dl)