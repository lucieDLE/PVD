from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.DJD_dataset import DJD_Condyle

from utils.file_utils import *
from utils.visualize import *
from model.pvcnn_generation import PVCNN2Base
import pandas as pd


datafile = '/CMF/data/lumargot/DCBIA_DJD/Deg_classification_aggregate_long_exists-2586.csv'

df = pd.read_csv(datafile)

count = 0
list_batch = []
for index, row in df.iterrows():
    try:
        point_cloud = np.load(row['surf'])
        # print(point_cloud.shape)
        count +=1
        list_batch.append(point_cloud)
    except Exception as e:
        print(f"Could not load point cloud: {e}")
        continue
    if count%20 == 0:
        list_batch = np.array(list_batch)
        # print(list_batch.shape)
        img = visualize_pointcloud_batch( 'tmp/'+ str(count)+ '.png', list_batch, 
                                    None, None, None)
        list_batch = []
    