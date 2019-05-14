import torch
import torch.nn.functional as func
import numpy as np


class Trainer:

    def __init__(self, batch_size, data_size, model, loader, optimizer, lr_decay=False, lr_decay_speed=0.8,
                 threshold_decay=False, threshold_decay_speed=0.8):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr_decay = lr_decay
        self.lr_decay_speed = pow(lr_decay_speed, self.batch_size / data_size)
        self.threshold_decay = threshold_decay
        self.threshold_decay_speed = pow(threshold_decay_speed, self.batch_size / data_size)

    def train(self, epoch, show_frq, accu_list, loss_list):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        for batch, (data, _) in enumerate(self.loader):
            self.optimizer.zero_grad()
            loss, accu = self.model(data[0], data[1], data[2:])
            total_loss += loss.item()
            total_accuracy += accu.item()
            loss.backward()
            self.optimizer.step()
            if self.threshold_decay:
                self._th_deacy(self.threshold_decay_speed)
            if self.lr_decay:
                self._lr_decay(self.lr_decay_speed)
            if (batch + 1) % show_frq == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.4f}\tAverage Accuracy: {:.4f}%'
                      .format(epoch, (batch + 1) * self.batch_size,
                              len(self.loader.dataset),
                              100. * (batch + 1) * self.batch_size / len(self.loader.dataset),
                              total_loss / show_frq,
                              total_accuracy / show_frq * 100))
                loss_list.append(total_loss / show_frq)
                accu_list.append(total_accuracy / show_frq)
                total_loss = 0
                total_accuracy = 0

    def _th_deacy(self, rate):
        self.model.simler.threshold = 2 - ((2 - self.model.simler.threshold) * rate)

    def _lr_decay(self, rate):
        lr = self.optimizer.param_groups[0]['lr']
        lr = lr * rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
