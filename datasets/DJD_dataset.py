import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import numpy as np
import torch.nn.functional as F
import pandas as pd
import pdb
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import class_weight

class DJD_Condyle(Dataset):
    def __init__(self,
                 data_file,
                 n_points=10000,
                 split='train',
                 category=0,
                 random_subsample=True,
                 points_mean=None,
                 points_std=None,
                 ):
        
        df = pd.read_csv(data_file)

        unique_classes = np.sort(np.unique(df['class']))
        self.unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df['class']))
        self.unique_class_weights = torch.from_numpy(self.unique_class_weights)

        df_classes = pd.Series(unique_classes)
        self.one_hot = pd.get_dummies(df_classes)
        
        self.split = split ## train - eval or test 
        self.n_points = n_points

        self.random_subsample = random_subsample
        self.input_dim = 3

        self.points = []
        self.cate_idx_lst = []

        for index, row in df.iterrows():
            try:
                point_cloud = np.load(row['surf'])

                ## add new point to the point cloud by using nearest neighbors
                if point_cloud.shape[0] < self.n_points:
                    num = self.n_points - point_cloud.shape[0]
                    
                    if point_cloud.shape[0] > 1000:
                        neigh = NearestNeighbors(n_neighbors=point_cloud.shape[1]+1)
                        neigh.fit(point_cloud)
                        NearestNeighbors(algorithm='auto')
                        indices = neigh.kneighbors(point_cloud, return_distance=False)
                        list_point = []
                        for i in range(int(num/2)+1):
                            idx_1, idx_2, idx_3 = indices[i][0], indices[i][1], indices[i][2]

                            new_point = (point_cloud[idx_1] + point_cloud[idx_2])/2
                            list_point.append(new_point)

                            new_point = (point_cloud[idx_1] + point_cloud[idx_3])/2
                            list_point.append(new_point)

                        new_pc = np.array(list_point)                        
                        point_cloud = np.concatenate((point_cloud, new_pc[:num]))

                        self.points.append(point_cloud[np.newaxis, ...])
                        self.cate_idx_lst.append(row['class'])
                        
                else:
                    indices = np.random.choice(point_cloud.shape[0], self.n_points, replace=False)
                    point_cloud = point_cloud[indices, :]
                
                    self.points.append(point_cloud[np.newaxis, ...])
                    self.cate_idx_lst.append(row['class'])

            except Exception as e:
                print(f"Could not load point cloud: {e}")
                continue


        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.points)))
        random.Random(38383).shuffle(self.shuffle_idx)

        self.points = [self.points[i] for i in self.shuffle_idx]
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]

        # Normalization
        self.points = np.concatenate(self.points)
        
        if points_mean is not None and points_std is not None:
            # using loaded dataset stats
            self.points_mean = points_mean
            self.points_std = points_std
        else:  # normalize across the dataset
            self.points_mean = self.points.reshape(-1, self.input_dim).mean(axis=0).reshape(1, 1, self.input_dim)
            self.points_std = self.points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        self.points = (self.points - self.points_mean) / self.points_std
       

        # n_train_sample = int(0.8* len(self.points))
        # if self.split == 'train':
        #     self.points = self.points[:n_train_sample]
        # elif self.split == 'test':
        #     self.points = self.points[n_train_sample:]
        # else:
        #     raise ValueError(f"Invalid split: {self.split}")

        self.display_axis_order = [0, 1, 2]
    
    def renormalize(self, mean, std):
        self.points = self.points * self.points_std + self.points_mean
        self.points_mean = mean
        self.points_std = std
        self.points = (self.points - self.points_mean) / self.points_std

        # n_train_sample = int(0.8* len(self.points))
        # if self.split == 'train':
        #     self.points = self.points[:n_train_sample]
        # elif self.split == 'test':
        #     self.points = self.points[n_train_sample:]
        # else:
        #     raise ValueError(f"Invalid split: {self.split}")
        


    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):

        point_cloud = self.points[idx]
        cate_idx = self.cate_idx_lst[idx]
        if self.random_subsample:
            indices = np.random.choice(point_cloud.shape[0], self.n_points, replace=False)
        else:
            indices = np.arange(self.n_points)
        point_cloud = point_cloud[indices, :]
        
        class_encoded = self.one_hot[cate_idx].to_numpy()

        m = self.points_mean.reshape(1,self.input_dim), 
        s = self.points_std.reshape(1,1)

        
        out = {
            'idx': idx,
            'points': torch.from_numpy(point_cloud).float(), 
            'class': cate_idx,
            'mean': m,
            'std': s,
        }
        return out