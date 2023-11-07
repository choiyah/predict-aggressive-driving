#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:34:35 2023

@author: choeyeong-a
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle

def split_data(seq, tar):
    train_seq, val_seq, train_tar, val_tar = train_test_split(seq, tar, test_size=0.2, random_state=42, shuffle=True)
    train_seq, test_seq, train_tar, test_tar = train_test_split(train_seq, train_tar, test_size=0.125, random_state=42, shuffle=True)

    train_seq = torch.Tensor(train_seq)
    train_tar = torch.Tensor(train_tar)
    test_seq = torch.Tensor(test_seq)
    test_tar = torch.Tensor(test_tar)
    val_seq = torch.Tensor(val_seq)
    val_tar = torch.Tensor(val_tar)

    return (train_seq, train_tar, test_seq, test_tar, val_seq, val_tar)

def load_data(f=None, data=None, batch_size=1, train=True):
    if train:
        print('>>Load train dataset file...')
        seq, tar = pickle.load(open(f, 'rb'))
        train_seq, train_tar, test_seq, test_tar, val_seq, val_tar = split_data(seq, tar)
        
        train_ds = TensorDataset(train_seq, train_tar)
        train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=train)

        test_ds = TensorDataset(test_seq, test_tar)
        test_ds = DataLoader(test_ds, batch_size=32, shuffle=train)
        
        val_ds = TensorDataset(val_seq, val_tar)
        val_ds = DataLoader(val_ds, batch_size=1, shuffle=train)
        
        return (train_ds, test_ds, val_ds)
    else:
        print('>>Load validation dataset file...')
        seq, tar = pickle.load(open(f, 'rb'))
        ds = TensorDataset(torch.Tensor(seq), torch.Tensor(tar))
        ds = DataLoader(ds, batch_size=batch_size, shuffle=train)

        return ds