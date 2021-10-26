import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

class Trainer(object):
    def __init__(self, model, args):
        wandb.config.update(args)
        self.device = args.device
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.epochs//2, 3*args.epochs//4], gamma=args.gamma)
        self.scaler = torch.cuda.amp.GradScaler()

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
        with torch.cuda.amp.autocast():
            out = self.model(img)
            loss = self.criterion(out, label)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        acc = out.argmax(dim=-1).eq(label).sum(-1)/img.size(0)
        wandb.log({
            'loss':loss,
            'acc':acc
        }, step=self.num_steps)


    # @torch.no_grad
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
                'lr': self.scheduler.get_last_lr()
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
