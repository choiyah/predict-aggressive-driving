#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:42:26 2023

@author: choeyeong-a
"""
import os
import torch

import dataset
from train import Predict_Model
import validation


if __name__=='__main__':
    
    dataset_path = './exploiting_processed_hz_50/[3] normal_dataset/'   # 학습용 정상 데이터
    val_N_path = './exploiting_processed_hz_50/[4] feature_dataset/'    # 검증용 난폭운전 데이터
    error_path = './exploiting_processed_hz_50/[5] error/'
    auc_path = './exploiting_processed_hz_50/[6] auc/'
    model_path='./exploiting_processed_hz_50/[7] model/'

    if not os.path.exists(error_path):os.makedirs(error_path)
    if not os.path.exists(auc_path):os.makedirs(auc_path)
    if not os.path.exists(model_path):os.makedirs(model_path)
    
    epochs=300
    window_size=200
    batch_size=500
    labels = [
        'non_aggressive',
        'aggressive_right_turn',
        'aggressive_left_turn',
        'aggressive_right_lane_change',
        'aggressive_left_lane_change',
        'aggressive_break',
        'aggressive_acceleration'
        ]
    
    normal_fname = [i for i in os.listdir(dataset_path) if str(window_size) in i][0]
    print('Normal dataset: ', normal_fname)
    train_data, test_data, val_data = dataset.load_data(f=dataset_path+normal_fname, batch_size=batch_size)
    
    model = Predict_Model(train_data, test_data)
    
    # Step 1. Train model
    for epoch in range(epochs):
        model.train(epoch)
        model.test(epoch)
    
    model_file=model_path+f'model_win{window_size}_batch{batch_size}_epoch{epochs}.pt'
    # model.save(epochs, model_file)
    
    # Load model
    model=torch.load(model_file, map_location=torch.device('mps'))
        
    # Step 2. Validate: 난폭 dataset 가져오기 (filenames_val)
    aggressive_fnames = [i for i in os.listdir(val_N_path) if str(window_size) in i]
        
    model.validate(val_data, error_path+f'win{window_size}_val_safe_error.pkl')  # a. safe -> validation error 구함 (val_safe_error.pkl)
    
    print('>>> Start Validation on Dangerous States..')
    for i,label in enumerate(labels[1:]):    # 난폭 label (normal:0)
        val_name = [name for name in aggressive_fnames if f'_{i+1}_win{window_size}' in name][0]
        print('Aggressive dataset: ', val_name)
        val_aggressive_data = dataset.load_data(f=val_N_path+val_name, train=False)

        # b. 난폭 -> validation error 구함 (val_feature_N_error.pkl)
        model.validate(val_aggressive_data, error_path+f'win{window_size}_val_{label}_error.pkl')

    # Step 3. Validation: ROC-AUC score
    print('>>> Start AUC..')
    
    # d. 난폭 라벨: np.ones(len(validation_not_safe))
    for label in labels[1:]:
        normal_loss_list, aggressive_loss_list, auc, th = validation.roc_auc_curve(error_path, window_size, label)






