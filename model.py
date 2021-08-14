# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:24:16 2021

@author: saki
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from hierarchical_generalization.make_datasets import generate_phase_train_test_data
from hierarchical_generalization.make_datasets import generate_taskset_test_data

def onehot(data):
    """
    input: trials x color x shape
    output: trials x onehotcoding of color and shape
    """
    num_trials = data.shape[0]
    length = data.shape[1] + data.shape[2]
    one_hot = torch.zeros([num_trials, length])
    
    for i in range(num_trials):
        dim0 = data[i].sum(axis=0).argmax()
        dim1 = data[i].sum(axis=1).argmax()
        one_hot[i][dim0] = 1
        one_hot[i][data.shape[1] - 1 + dim1] = 1
        
    return one_hot

def phase_data(phase, batch_size=120):
    train_data, test_data = generate_phase_train_test_data()
    train, test = train_data[phase]
    train = onehot(train)
    test = torch.from_numpy(test)
    return train[:batch_size], test[:batch_size]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=2)
    
    def init_hidden(self, batch_size):
        #h0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        #c0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return [h0, c0]
    
    def forward(self, x):
        hidden = self.init_hidden(x.size(1))
        steps = x.size(0) # squence length
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        #out = self.softmax(self.fc(out))
        return out
    
def plot_loss(phase, all_losses):
    """
    all_losses must be an array
    """
    f, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel('Blocks')
    ax.set_ylabel('Loss')
    ax.set_title(phase)
    for i in range(len(all_losses)):
        if all_losses[i] < 0.1:
            ax.text(i*0.9, all_losses[i]*1.4, s=(i, str(all_losses[i])[:4]))
            break
    ax.plot(all_losses)
    #ax.text(len(all_losses)*0.9, all_losses[-1]*1.2, s=str(all_losses[-1])[:5], alpha=0.9)
    
def plot_accuracy(phase, all_accus):
    """
    all_losses can be an array or a matrix
    """
    phases = ['Phase A', 'Phase B', 'Phase C']
    f, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel('Blocks')
    ax.set_ylabel('Accuracy')
    ax.set_title(phase)
    ax.set_ylim([0, 1.1])
    ax.set_yticks(np.arange(0,1.1,0.2))
    ax.spines['top'].set_visible(False)  #去掉上边框
    ax.spines['right'].set_visible(False) #去掉右边框
    for j in range(all_accus.shape[-1]):
        all_accu = all_accus[:, j]
        for i in range(len(all_accu)):
            if all_accu[i] > 0.9:
                ax.text(i, all_accu[i], s=i)
                break
        ax.plot(all_accu, label=phases[j])
        ax.legend() #loc='upper right'
        
def test_allphase(plot=False, ifprint=False):
    phases = ['Phase A', 'Phase B', 'Phase C']
    train_data, test_data = generate_phase_train_test_data()
    all_accu = []
    
    for phase in phases:
        test_input, test_target = test_data[phase]
        test_input = onehot(test_input)
        test_target = torch.from_numpy(test_target)

        with torch.no_grad():
            test_input = torch.unsqueeze(test_input, 0)
            output = lstm(test_input)
            output = torch.squeeze(output, 0)
            accuracy = (output.argmax(dim=1) == test_target.argmax(dim=1)).float()
            accuracy = accuracy.mean()
            all_accu.append(accuracy)
            if ifprint:
                print(f"Accuracy in {phase} is: {accuracy.item():.2%}")
        
    if plot:
        f, ax = plt.subplots(figsize=(4, 4))
        ax.bar(phases, all_accu, width=0.5)
        for i in range(3):
            ax.text(i, all_accu[i]*1.05, '%.3f' % all_accu[i], ha='center', va= 'bottom')
    
    return all_accu

def phase_test(phase, batch_size=120):
    ts_test_data = generate_taskset_test_data()
    train, target = ts_test_data[phase]
    train = onehot(train)
    target = torch.from_numpy(target)
    return train[:batch_size], target[:batch_size]


def test_C0C1vsC2(phase, plot=False, ifprint=False):
    """
    phase: 'Phase A' or 'Phase B'
    """
    TS1_train, TS1_target = phase_test('TS 1 '+phase)
    TS2_train, TS2_target = phase_test('TS 2 '+phase)

    TS1_accu = []
    TS2_accu = []
    
    #TS1_target = torch.from_numpy(TS1_target)
    #TS2_target = torch.from_numpy(TS2_target)
    
    for i in range(TS1_target.size(0)):
        with torch.no_grad():
            ###train = torch.tile(TS1_PhaseA_train, (4,1))
            train = TS1_train[i:i+1]
            train = torch.unsqueeze(train, 0)
            output = lstm(train)
            output = torch.squeeze(output, 0)
            accuracy = (output.argmax(dim=1) == TS1_target[i:i+1].argmax(dim=1)).float()
            TS1_accu.append([i.item() for i in accuracy])
            

            train = TS2_train[i:i+1]
            train = torch.unsqueeze(train, 0)
            output = lstm(train)
            output = torch.squeeze(output, 0)
            accuracy = (output.argmax(dim=1) == TS2_target[i:i+1].argmax(dim=1)).float()
            TS2_accu.append([i.item() for i in accuracy])
            
    return (np.mean(TS1_accu), np.mean(TS2_accu))

BATCH_SIZE = 1

train, test = phase_data('Phase A', batch_size=BATCH_SIZE) # batch x features; batch x actions
input_size = train.size(1)
output_size = test.size(1)
hidden_size = 64

num_layers = 1
length = 1

lstm = LSTM(input_size, hidden_size, num_layers, output_size)

optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train_model(phase, transfer=True):
    current_loss = 0
    all_losses = []
    all_accu = []
    n_iters = 100
    T1_all_accu = []
    T2_all_accu = []

    for i in range(n_iters):
        train, test = phase_data(phase, batch_size=BATCH_SIZE)
        train = torch.unsqueeze(train, 0)

        output = lstm(train)
        output = torch.squeeze(output, 0)
        loss = criterion(output, test.argmax(axis=1))
        
        accuracy = test_allphase()
        all_accu.append([i.item() for i in accuracy])

        TS1_accu, TS2_accu = test_C0C1vsC2(phase)
        T1_all_accu.append(T1_all_accu)
        T2_all_accu.append(T2_all_accu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
        if all_losses[-1] <= 0.01:
            if transfer or i >= 100:
                break
    print(i)
    return all_losses, np.array(all_accu), np.array(T1_all_accu), np.array(T2_all_accu)

all_lossesA, all_accuA, T1_all_accuA, T2_all_accuA = train_model('Phase A', transfer=False)
plot_loss('Phase A', all_lossesA)
plot_accuracy('Phase A', all_accuA)



