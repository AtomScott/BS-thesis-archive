import copy
import chainer
import chainercv
import glob
import os
import numpy as np
import pandas as pd

class JumpPoseDataset(chainer.dataset.DatasetMixin):
    """
    A simple dataset that that iterates over jump (xyzt) data and its label data.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """

    def __init__(
        self, pose_paths, info_path,
    ):
        self.pose_paths = pose_paths
        self.info_path = info_path
        self.info_df = pd.read_excel(info_path)

    def __len__(self):
        return len(self.pose_paths)

    def get_example(self, i):
        id = self.info_df.iloc[i].values[0]

        pose = self.get_pose_by_id(id)
        label = self.encode_label(self.info_df.iloc[i].values[1:])
        return pose, label

    def get_pose_by_id(self, id):
        pose_path = os.path.join(self.pose_paths, id+'.csv')
        pose_df = pd.read_csv(pose_path, skiprows=6, index_col='Frame')
        return pose_df.values

    def get_label_by_id(self, id):
        label = self.info_df[['ID'] == '172-A_001'].values[0, 1:]
        return label

    def encode_label(self, label):
        if label[0] == '〇':
            label[0] = 1.
        else:
            label[0] = 0.
        if label[1] == '〇':
            label[0] = 1.
        else:
            label[1] = 0.
        label = label.astype(np.float32)
        return label
