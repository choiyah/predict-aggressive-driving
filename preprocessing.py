# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pickle


def make_normalScaler(df, filepath):    # 정상 데이터 (Label 0)에 맞춰 MinMaxScaler 저장
    # MinMax scaling with Label 0 (normal)
    normal = df[df['label']==0].drop('label', axis=1)
    
    scaler = MinMaxScaler()
    
    scaler.fit(normal)
    for i, col in enumerate(df.columns[:-1]):
        print('>> {:<20} Normal MAX:{:>10}'.format(col, round(scaler.data_max_[i],4)))
        print('>> {:<20} Normal MIN:{:>10}'.format(col, round(scaler.data_min_[i],4)))
    
    # Save scaler
    scaler_file = filepath.replace('.csv', '_scaler.pkl')
    joblib.dump(scaler, open(scaler_file, 'wb'))
    print(f'Save {filename} scaler.pkl')
        
def window_sliding(data, filepath, window_size_list):
    print('Start Window Sliding...')
    
    for window_size in window_size_list:
        total_container = list()
        
        # Window Sliding (keep label)
        loop = len(data) - window_size - 1
        for window_idx in range(loop):
            start_idx = window_idx
            end_idx = window_idx + window_size
            
            selected_data = data.iloc[start_idx:end_idx].values     # input vectors
            selected_label = data.iloc[end_idx].values              # predict vector
            
            container = dict()
            container['features'] = selected_data
            container['label'] = selected_label
            
            total_container.append(container)   # len(total_container) = len(data_scaling) - window_size - 1
            print(f'\r[{window_idx+1}/{loop}] Append container', end='')
        
        sliding_path = path+'[1] window_sliding/'
        if not os.path.exists(sliding_path):os.makedirs(sliding_path)

        sliding_name = filepath.split('/')[-1]
        sliding_name = sliding_name.replace('.csv', f'_sliding_win{window_size}.pkl')
        pickle.dump(total_container, open(sliding_path+sliding_name, 'wb'))
        print(f'\t>>> Save {sliding_name} File')

def make_windowSliding(df, filepath, window_size_list):
    # Load scaler
    scaler_file = filepath.replace('.csv', '_scaler.pkl')
    scaler = joblib.load(open(scaler_file, 'rb'))
    
    # MinMaxScaling (Normal)
    data = scaler.transform(df.drop('label', axis=1))    # numpy
    col_list = df.columns
    data = pd.DataFrame(data, columns=col_list[:-1])
    data = pd.concat([data, df['label']], axis=1)
    print('Complete MinMaxScaling', filename)
    
    # window sliding
    window_sliding(data, filepath, window_size_list)

def split_same_label(path, filenames, window_size_list, labels):
    for window_size in window_size_list:
        containers = list() # make 7 lists
        for i in range(labels):
            containers.append(list())
            
        print('\n================ win: {} ================'.format(window_size))
        
        for filename in filenames:
            print('\n>>> Processing {}...'.format(filename))
            sliding_path = path+'[1] window_sliding/'
            sliding_name = '{}_sliding_win{}.pkl'.format(filename.replace('.csv',''), window_size)
            total_container = pickle.load(open(sliding_path+sliding_name, 'rb'))
    
            # split container into label
            loop = len(total_container)
            pre_tar = int(total_container[0]['label'][-1])    # First Target Label
            print('[{}/{}] Start Target Label: {}'.format(0, loop, pre_tar))
            
            for i in range(loop):
                container = total_container[i]
                current_tar = int(container['label'][-1])
    
                # Drop label
                selected_data = container['features'][:,:-1]
                selected_label = container['label'][:-1]
                
                container = dict()
                container['features'] = selected_data
                container['label'] = selected_label
                
                # update target(label) value
                if not(current_tar == pre_tar):
                    print('[{}/{}] Start Target Label: {}'.format(i, loop, current_tar))
                    pre_tar = current_tar
    
                # Split container into continuous label
                containers[current_tar].append(container)

                    
        # Save container by target (label)
        print('\n---------------- Save PKL ---------------')
        for i in range(labels):
            feature_path = path+'[2] split_feature/'
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
                
            feature_name = 'feature_{}_win{}.pkl'.format(i, window_size)
            pickle.dump(containers[i], open(feature_path+feature_name, 'wb'))
            print('label_window: {}/{} Complete'.format(i, labels-1), end='\r')
        print('\n')
    
def validate_continuous_label(path, filenames, labels):
    for filename in filenames:
        print('\n================', filename, '================')
        filepath = os.path.join(path, filename)
        data = pd.read_csv(filepath)
        
        for i in range(labels):
            print(' {}:{}'.format(i, len(data[data['label']==i])), end='')
        
        print('\n Total: {}'.format(len(data)))
        print('---------------------------------------------------------------------------')
        
        # Split data into continuous label
        data_scaling = list()
        data_sameLabel = data[data.index==0]    # start row
        
        # First Label
        pre_label = int(data_sameLabel['label'])
        print('Start Idx of continous label_{}: {}'.format(pre_label, 0), end='')
        
        # row 하나씩 돌면서 라벨 체크
        for i in range(1, len(data)):
            row = data[data.index==i]
            
            row_label = int(row['label'])    # 현재 label
            
            if row_label == pre_label:
                data_sameLabel = pd.concat([data_sameLabel, row])
                
                if i == len(data)-1:
                    data_scaling.append(data_sameLabel)
                    print('\tNumber of continous label_{}: {}'.format(row_label, len(data_sameLabel)))    # final
            else:
                # 라벨 구간 변경
                data_scaling.append(data_sameLabel)
                print('\tNumber of continous label_{}: {}'.format(pre_label, len(data_sameLabel)))    # last data_sameLabel
                print('Start Idx of continous label_{}: {}'.format(row_label, i), end='')    # new data_sameLabel
                pre_label = row_label
                data_sameLabel = data[data.index==i]    # update start row

def ds2numpy(feature_path):
    dataset_path = './exploiting_processed_hz_50/[3] normal_dataset/'
    N_path = './exploiting_processed_hz_50/[4] feature_dataset/'
    
    if not os.path.exists(dataset_path):os.makedirs(dataset_path)
    if not os.path.exists(N_path):os.makedirs(N_path)
    
    filenames = [i for i in os.listdir(feature_path) if 'pkl' in i]
    for filename in filenames:
        total_container=pickle.load(open(feature_path+filename, 'rb'))
        print('filename:', filename)
        total = len(total_container)
        print('Total Container:', total)
        
        window_size = len(total_container[0]['features'])
        col_num = len(total_container[0]['label'])
        print('features:', window_size,'*', col_num)
        print('label:', col_num)
        
        seq = np.ndarray(shape=(total, window_size, col_num))
        tar = np.ndarray(shape=(total, col_num))
        
        for i in range(total):
            seq[i] = total_container[i]['features']
            tar[i] = total_container[i]['label']
        print('{}/{}'.format(total, total))
            
        pair = [seq, tar]
        
        if '_0_' in filename:
            pickle.dump(pair, open(dataset_path+'normal_{}.pkl'.format(window_size), 'wb'))
        else:
            pickle.dump(pair, open(N_path+'np_'+filename, 'wb'))
        print('---------------------------------------')

if __name__ == '__main__':
    labels = 7
    window_size_list = [25]
    
    path = './exploiting_processed_hz_50/'
    filenames = [i for i in os.listdir(path) if 'csv' in i]
    print(f'filenames: {filenames}')
    
    for filename in filenames:
        print('\n================', filename, '================')
        filepath = os.path.join(path, filename)
        df = pd.read_csv(filepath)
    
        make_normalScaler(df, filepath)                     # make normal scaler
        make_windowSliding(df, filepath, window_size_list)  # fit normal scaler and make window sliding

    split_same_label(path, filenames, window_size_list, labels) # Split to window_size by label
    
    # # validate Continuous Label
    # validate_continuous_label(path, filenames, labels)
    
    feature_path = path+'[2] split_feature/'
    ds2numpy(feature_path)
    
