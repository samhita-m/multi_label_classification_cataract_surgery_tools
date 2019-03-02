#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:53:04 2017

@author: siplab
"""

from __future__ import print_function, division

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:22:43 2017

@author: siplab
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 21:10:08 2017

@author: SAMHITA M
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import torch.utils.data as data_utils
from random import randint
import h5py
import csv

train_files = ['random_train1.h5','random_train1.h5']
train_labels = ['random_train1.csv','random_train2.csv']
for i in range(0,2):

    filename = train_files[i]
    f = h5py.File(filename, 'r')

    a_group_key = f.keys()[0]

    # Get the data
    data = np.array(f[a_group_key])
    data = data.swapaxes(0,3)
    data = data.swapaxes(1,2)
    data = data.swapaxes(2,3)
    
    data_tensor = torch.from_numpy(data)
    
    
    tr_label = np.genfromtxt('random_train1.csv', delimiter=',')
        
    train_label = torch.from_numpy(tr_label)
    
    
    filename = 'random_test1.h5'
    f = h5py.File(filename, 'r')
    
    a_group_key = f.keys()[0]
    
    # Get the data
    data1 = np.array(f[a_group_key])
    data1 = data1.swapaxes(0,3)
    data1 = data1.swapaxes(1,2)
    data1 = data1.swapaxes(2,3)
    test_tensor = torch.from_numpy(data1)
    
    te_label = np.genfromtxt('random_test1.csv', delimiter=',')
        
    test_label  = torch.from_numpy(te_label)
    
        
    
    
    
    train = data_utils.TensorDataset(data_tensor, train_label)
    train_loader = data_utils.DataLoader(train, batch_size=1, shuffle=True)
    val = data_utils.TensorDataset(test_tensor, test_label)
    test_loader = data_utils.DataLoader(val, batch_size=1, shuffle=True)
    
    
    use_gpu = torch.cuda.is_available()
    
    
    
    def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
        since = time.time()
    
        best_model = model
      #  best_acc = 0.0
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    optimizer = lr_scheduler(optimizer, epoch)
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode
    
                running_loss = 0.0
              #  running_corrects = 0
    
                # Iterate over data.
                if phase == 'train':
                    
                    for data in train_loader:
                        # get the inputs
                        inputs, labels = data
    
                        # wrap them in Variable
                        if use_gpu:
                            inputs, labels = Variable(inputs.float().cuda()), \
                             Variable(labels.float().cuda())
                        else:
                            inputs, labels = Variable(inputs), Variable(labels)
    
                        # zero the parameter gradients
                        optimizer.zero_grad()
    
                        # forward
                        outputs = model(inputs)
    
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
    
                        # backward + optimize only if in training phase
                    
                        loss.backward()
                        optimizer.step()
    
                        # statistics
                        running_loss += loss.data[0]
                       
              
    #                    running_corrects += torch.sum(preds.float() == labels.data)
    
                        epoch_loss = running_loss / len(train_loader)
                        print(epoch_loss)
                        print(preds)
                    #    epoch_acc = running_corrects / len(train_loader)
    
     #                   print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                 #       phase, epoch_loss, epoch_acc))
                            # Iterate over data.
                if phase == 'val':
                    
                    for data in test_loader:
                        # get the inputs
                        inputs, labels = data
    
                        # wrap them in Variable
                        if use_gpu:
                            inputs, labels = Variable(inputs.float().cuda()), \
                             Variable(labels.float().cuda())
                        else:
                            inputs, labels = Variable(inputs), Variable(labels)
    
                        # zero the parameter gradients
                        optimizer.zero_grad()
    
                        # forward
                        outputs = model(inputs)
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
    
    
                        # statistics
                        running_loss += loss.data[0]
                 #       running_corrects += torch.sum(preds == labels.data)
    
                        epoch_loss = running_loss / len(test_loader)
                        print(preds)
                  #      epoch_acc = running_corrects / len(test_loader)
    
    #                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
                # deep copy the model
          #      if phase == 'val' and epoch_acc > best_acc:
           #         best_acc = epoch_acc
            #        best_model = copy.deepcopy(model)
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
       # print('Best val Acc: {:4f}'.format(best_acc))
        return best_model
        
        
        
        
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    
        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        return optimizer
    
  
    
                    
    model_ft = models.densenet121(pretrained=True)
    model_ft.classifier = nn.Linear(1024, 22)
  #  model_ft.classifier = nn.Linear(4096,22)

    
    if use_gpu:
        model_ft = model_ft.cuda()
    
    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    #(weight=None, size_average=True)
    
    
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=5)
  #  visualize_model(model_ft)              
                    
torch.save(model_ft.state_dict(), '/home/siplab/samhita/model_ft')               
#the_model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))
            

