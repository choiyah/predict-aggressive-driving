#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:33:43 2023

@author: choeyeong-a
"""
from sklearn import metrics
import pickle
import numpy as np

import matplotlib.pyplot as plt

def roc_auc_curve(path, window_size, label):
    # c. safe 라벨: np.zeros(len(validation_safe))
    val_safe_list = pickle.load(open(path+f'win{window_size}_val_safe_error.pkl', 'rb'))
    print('[Validatioin] Average Normal losses: ', val_safe_list.mean())

    safe_label = np.zeros(len(val_safe_list))
    
    # d. 난폭 라벨: np.ones(len(validation_not_safe))
    val_aggressive_list = pickle.load(open(path+f'win{window_size}_val_{label}_error.pkl', 'rb'))
    print(f'Average loss of {label}:', val_aggressive_list.mean())

    aggressive_label = np.ones(len(val_aggressive_list))

    # AUC 계산
    pred_vals_loss = np.concatenate((val_safe_list, val_aggressive_list), axis=0)    # 정상/비정상 val 계산으로 도출된 loss값
    pred_vals = np.concatenate((safe_label, aggressive_label), axis=0)    # 정상 0, 비정상 1

    fpr, tpr, thresholds = metrics.roc_curve(pred_vals, pred_vals_loss)        # 정상/비정상과 그에 대한 score (loss)
    auc = metrics.auc(fpr, tpr)
    print(f"ROC-AUC score of {label}:", auc)
    
    graph_curve(fpr, tpr, label)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Threshold of {label}:", optimal_threshold, end='\n\n')
    
    return val_safe_list, val_aggressive_list, auc, optimal_threshold

def graph_curve(fpr, tpr, label):
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, marker='.', label=label)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    