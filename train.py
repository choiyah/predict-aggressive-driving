#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:11:14 2023

@author: choeyeong-a
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm
import pickle
import numpy as np



class predictLSTM(nn.Module):
    def __init__(self, n_features=12, embedding_dim=3):
        super(predictLSTM, self).__init__()
        self.n_features = n_features
        self.embedding_dim=embedding_dim*n_features
        
        self.rnn = nn.LSTM(
            input_size=self.n_features,
            hidden_size = self.embedding_dim,
            batch_first = True
            )
        
        self.output_layer = nn.Linear(self.embedding_dim, self.n_features)
    
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.output_layer(x[:,-1,:])
        return  x

class Predict_Model:
    def __init__(self,train_dataloader: DataLoader, test_dataloader: DataLoader,
                 lr: float = 1e-4, n_features=12, window_size=25, batch_size=500):
        mps_condition = torch.backends.mps.is_available()
        self.device = torch.device("mps" if mps_condition else "cpu")
        
        # Initialize the BERT Language Model, with BERT model
        self.model = predictLSTM(n_features).to(self.device)
        
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        
        # Setting the Adam optimizer with hyper-param
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        
        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.MSELoss()
        
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_data)
    
    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)
    
    def validate(self, data, filename):
        val_list = list()

        for seq_batch, tar_batch in data:
            pred = self.model.forward(seq_batch.to(self.device))
            val_loss = self.criterion(pred, tar_batch.to(self.device))
        
            val_list.append(val_loss.item())
            
        val_list = np.asarray(val_list)
        
        pickle.dump(val_list, open(filename, 'wb'))
        print(' Save', filename)
        
    
    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{bar:10}{r_bar}")
        
        avg_loss = 0.0
        
        for i, data in data_iter:
            # 1. forward the seq_batch_train
            predictions = self.model.forward(data[0].to(self.device))   # seq_batch_train
            
            # 2. MSELoss of predicting tar_batch_train
            loss = self.criterion(predictions, data[1].to(self.device)) # tar_batch_train
            
            # 3. backward and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            avg_loss += loss.item()
        
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), end='\n')
        
    def save(self, epochs, filename):
        torch.save(self, filename)
        print("EP:%d Model Saved on:" % epochs, filename)
        return filename