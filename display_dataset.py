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
from utils.file_utils import *
import matplotlib.pyplot as plt


def visualize_pointcloud_batch(path, pointclouds, pred_labels, labels, categories, vis_label=False, target=None,  elev=30, azim=225):
    batch_size = len(pointclouds)

    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)
    idx = 0
    for _, pc in enumerate(pointclouds):
        
        
        pc =  pc.reshape((2580,3))
        
        colour = 'g'

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=5)
        # ax.view_init(elev=elev, azim=azim)
        # ax.axis('off')
        idx+=1

    plt.savefig(path)
    plt.close(fig)
    return fig

data_file = '/CMF/data/lumargot/condyles_4classes_train.csv'

df = pd.read_csv(data_file)

unique_classes = np.sort(np.unique(df['class']))
unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df['class']))
unique_class_weights = torch.from_numpy(unique_class_weights)

n_points = 2580

random_subsample = True
input_dim = 3

all_points = []
cate_idx_lst = []

for index, row in df.iterrows():
    try:
        point_cloud = np.load(row['surf'])

        if point_cloud.shape[0] < n_points:
            ## add new point to the point cloud by using nearest neighbors
            num = n_points - point_cloud.shape[0]
            
            if point_cloud.shape[0] > 1000:
                neigh = NearestNeighbors(n_neighbors=point_cloud.shape[1]+1)
                neigh.fit(point_cloud)
                NearestNeighbors(algorithm='auto')
                indices = neigh.kneighbors(point_cloud, return_distance=False)
                list_point = []
                for i in range(int(num/2)+1):
                    idx_1 = indices[i][0]
                    idx_2 = indices[i][1]
                    idx_3 = indices[i][2]

                    new_point = (point_cloud[idx_1] + point_cloud[idx_2])/2
                    list_point.append(new_point)

                    new_point = (point_cloud[idx_1] + point_cloud[idx_3])/2
                    list_point.append(new_point)

                new_pc = np.array(list_point)
                
                point_cloud = np.concatenate((point_cloud, new_pc[:num]))

                all_points.append(point_cloud[np.newaxis, ...])
                cate_idx_lst.append(row['class'])
            # else:
            #     print("Ignoring point cloud: not enough point :", point_cloud.shape[0])
        else:
            indices = np.random.choice(point_cloud.shape[0], n_points, replace=False)
            point_cloud = point_cloud[indices, :]
        
            all_points.append(point_cloud[np.newaxis, ...])
            cate_idx_lst.append(row['class'])

    except Exception as e:
        print(f"Could not load point cloud: {e}")
        continue

array_points = np.array(all_points)
list_batch = []
for i in range(0,160,20):
    arr = array_points[i:i+20]
    
    arr = arr.reshape(20,-1,2580,3)    
    img_gt = visualize_pointcloud_batch(str(i)+'_'+str(i+20)+'.png', arr, None, None, None)

arr = array_points[160:-1]    
arr = arr.reshape(17,-1,2580,3)
img_gt = visualize_pointcloud_batch('160_end.png', arr, None, None, None)