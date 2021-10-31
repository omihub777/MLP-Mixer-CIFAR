import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import warmup_scheduler
from sam import SAM


class Trainer(object):
    def __init__(self, model, args):
        wandb.config.update(args)
        self.device = args.device
        self.clip_grad = args.clip_grad
        self.model = model
        self.sam = args.sam
        if self.sam:
            if args.optimizer=='sgd':
                self.base_optimizer = optim.SGD
                self.optimizer = SAM(self.model.parameters(), self.base_optimizer, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer=='adam':
                self.base_optimizer = optim.Adam
                self.optimizer = SAM(self.model.parameters(), self.base_optimizer, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        else:
            if args.optimizer=='sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer=='adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)


        if args.scheduler=='step':
            self.base_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=args.gamma)
        elif args.scheduler=='cosine':
            self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        else:
            raise ValueError(f"No such scheduler: {self.scheduler}")


        if args.warmup_epoch:
            self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=args.warmup_epoch, after_scheduler=self.base_scheduler)
        else:
            self.scheduler = self.base_scheduler

        self.epochs = args.epochs
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        self.num_steps = 0
        self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
    
    def _train_one_step(self, batch):
        self.model.train()
        img, label = batch
        self.num_steps += 1
        img, label = img.to(self.device), label.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(img)
        loss = self.criterion(out, label)
        loss.backward()
        if self.clip_grad:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        if self.sam:
            self.optimizer.first_step(zero_grad=True)
            self.criterion(self.model(img), label).backward()
            self.optimizer.second_step(zero_grad=True)
        else:
            self.optimizer.step()

        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)
        wandb.log({
            'loss':loss,
            'acc':acc
        }, step=self.num_steps)


    def _test_one_step(self, batch):
        self.model.eval()
        img, label = batch
        img, label = img.to(self.device), label.to(self.device)

        with torch.no_grad():
            out = self.model(img)
            loss = self.criterion(out, label)

        self.epoch_loss += loss * img.size(0)
        self.epoch_corr += out.argmax(dim=-1).eq(label).sum(-1)


    def fit(self, train_dl, test_dl):
        for epoch in range(1, self.epochs+1):
            for batch in train_dl:
                self._train_one_step(batch)
            wandb.log({
                'epoch': epoch, 
                'lr':self.optimizer.param_groups[0]["lr"]
                }, step=self.num_steps
            )
            self.scheduler.step()

            
            num_imgs = 0.
            self.epoch_loss, self.epoch_corr, self.epoch_acc = 0., 0., 0.
            for batch in test_dl:
                self._test_one_step(batch)
                num_imgs += batch[0].size(0)
            self.epoch_loss /= num_imgs
            self.epoch_acc = self.epoch_corr / num_imgs
            wandb.log({
                'val_loss': self.epoch_loss,
                'val_acc': self.epoch_acc
                }, step=self.num_steps
            )
