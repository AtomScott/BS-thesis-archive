import os, re
from collections import defaultdict

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

class SLJDataset(Dataset):
    def __init__(
        self, pose_paths, grf_paths, info_path, transforms=[]
    ):
        self.pose_paths = pose_paths
        self.grf_paths = grf_paths
        self.info_path = info_path
        self.info_df = pd.read_excel(info_path)
        self.labels = self.encode_labels()
        self.transforms = transforms

        assert len(self.info_df) == len(self.pose_paths) == len(self.grf_paths), \
            f'Should be equal: {len(self.info_df)}, {len(self.pose_paths)}, {len(self.grf_paths)}'

    def __len__(self):
        return len(self.info_df)
    
    def __getitem__(self, idx):
        pose = self.get_pose(self.pose_paths[idx])
        grf = self.get_grf(self.grf_paths[idx])
        label = self.get_label(idx)

        sample = {'pose':pose, 'grf':grf, 'label':label}

        for transform in self.transforms:
            sample = transform(sample)

        return sample

    def get_maxlen(self):
        mx_len = {
            'pose': max(map(len, [self.get_pose(path) for path in self.pose_paths])),
            'grf': max(map(len, [self.get_grf(path) for path in self.grf_paths]))
        }
        return mx_len

    def get_pose(self, path):
        pose_df = pd.read_csv(path, skiprows=6, index_col='Frame').drop('Time (Seconds)', axis='columns')
        pose = pose_df.values
        pose = pose.reshape(pose.shape[0], pose.shape[1]//3, 3)
        return pose

    def get_grf(self, path):
        df = pd.read_csv(path, skiprows=6, header=None, names=['DataLabel','null','FX[1]','FY[1]','FZ[1]','AX[1]','AY[1]']).drop(['DataLabel', 'null'], axis='columns')
        return df.values

    def get_label(self, idx):
        # order -> ['miss', 'healthy', 'structural', 'subjective', 'recovered', 'prone'] 
        return self.labels[idx]

    def encode_labels(self):
        
        labels = defaultdict(lambda: {a:0 for a in ['miss', 'healthy', 'structural', 'subjective', 'recovered', 'prone']})

        # CODE HORROR!
        for idx, row in self.info_df.iterrows():
            if row.成功失敗 == '〇':
                labels[idx]['miss']=0
            elif row.成功失敗 == '×':
                labels[idx]['miss']=1
            else:
                raise KeyError(row.成功失敗)        
            
            if row.足関節不安定性の分類==1:
                labels[idx]['healthy']=1
            elif row.足関節不安定性の分類==2:
                labels[idx]['structural']=1
            elif row.足関節不安定性の分類==3:
                labels[idx]['subjective']=1
            elif row.足関節不安定性の分類==4:
                labels[idx]['prone']=1
            elif row.足関節不安定性の分類==5:
                labels[idx]['structural']=1
                labels[idx]['subjective']=1
            elif row.足関節不安定性の分類==6:
                labels[idx]['structural']=1
                labels[idx]['prone']=1
            elif row.足関節不安定性の分類==7:
                labels[idx]['subjective']=1
                labels[idx]['prone']=1
            elif row.足関節不安定性の分類==8:
                labels[idx]['structural']=1
                labels[idx]['subjective']=1
                labels[idx]['prone']=1        
            elif row.足関節不安定性の分類==9:
                labels[idx]['recovered']=1
            else:
                raise KeyError(row.足関節不安定性の分類)

        return pd.DataFrame(labels, index =['miss', 'healthy', 'structural', 'subjective', 'recovered', 'prone'] ).T.values

def edgepad(sample, mx_len):
    pose = sample['pose']
    grf = sample['grf']
    label = sample['label']
    
    diff = (mx_len['pose'] - len(pose))/2
    before_pad, after_pad= np.ceil(diff).astype(int), np.floor(diff).astype(int)
    pose = np.pad(pose, [[before_pad,after_pad],[0,0],[0,0]], mode='edge')
    
    diff = (mx_len['grf'] - len(grf))/2
    before_pad, after_pad= np.ceil(diff).astype(int), np.floor(diff).astype(int)
    grf = np.pad(grf, [[before_pad,after_pad],[0,0]], mode='edge')

    return {'pose':pose, 'grf':grf, 'label':label}