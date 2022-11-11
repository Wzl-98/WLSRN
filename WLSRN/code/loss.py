import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F



class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        
        # SR loss
        DS_weight = args.alpha
        DWT_weight = 1-args.alpha

        self.loss.append({'type': "DS", 'weight': DS_weight, 'function': nn.L1Loss()})
        self.loss.append({'type': "DWT", 'weight': DWT_weight, 'function': nn.L1Loss()})
        self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
        
        
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.log = torch.Tensor()

        if args.resume == 1: 
            self.load(ckp.dir, cpu=args.cpu)



    def forward(self, student_sr, student_DWT, teacher_DWT, hr):
        # DS Loss
        DS_loss = self.loss[0]['function'](student_sr, hr) * self.loss[0]['weight']
        self.log[-1, 0] += DS_loss.item()

        # DWT Loss
        DWT_loss = self.loss[1]['function'](student_DWT, teacher_DWT) * self.loss[1]['weight']
        self.log[-1, 1] += DWT_loss.item()

        # Total Loss
        loss_sum = DWT_loss + DS_loss
        self.log[-1, -1] += loss_sum.item()
        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = numpy.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)


    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))