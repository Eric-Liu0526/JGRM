import pandas as pd
import networkx as nx
from shapely import wkt
from shapely.geometry import LineString
import pickle as pkl
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from concurrent.futures import ThreadPoolExecutor, as_completed

data_name = "chengdu"
edge_weight_threshold = 0.8
with open(f'dataset/didi_{data_name}/{data_name}_1101_1115_data_sample10w.pkl', 'rb') as file:
# with open(f'dataset/didi_{data_name}/{data_name}_1101_1115_data_seq_evaluation.pkl', 'rb') as file:
    traj_df = pkl.load(file)
traj_df.reset_index(drop=True, inplace=True)
with open(f'dataset/didi_{data_name}/sub_g_traj_dict.pkl', 'rb') as f:
    sub_g_traj_dict = pkl.load(f)
with open(f'dataset/didi_{data_name}/road_subgraph_node_ids.pkl', 'rb') as f:
    sub_g_dict = pkl.load(f)

# 读取traj_df中指定index的轨迹数据
subg_id = 8
traj_ids = sub_g_traj_dict[subg_id]
road_set = set(sub_g_dict[subg_id])
road_with_trajs_dict = {key: set() for key in road_set}
traj_sub_df = traj_df.loc[traj_ids]
traj_num = traj_sub_df.shape[0]
print(f'sub_graph_{subg_id} has {traj_num} trajs')

# 创建一个有向图
sub_G = nx.Graph()

cpath_lists = traj_sub_df['cpath_list']

# 创建一个进度条对象
pbar = tqdm(total=len(cpath_lists) * (len(cpath_lists) - 1) // 2)

for i in range(len(cpath_lists)):
    for j in range(i+1, len(cpath_lists)):
        cpath_list_i = cpath_lists.iloc[i]
        cpath_list_j = cpath_lists.iloc[j]
        # 计算两个轨迹的路径列表的交集
        intersection = set(cpath_list_i).intersection(set(cpath_list_j))
        # 计算两个轨迹的路径列表的并集
        union = set(cpath_list_i).union(set(cpath_list_j))
        # 计算权重
        weight = len(intersection) / len(union)
        # 如果权重大于阈值，则添加边
        if weight > edge_weight_threshold:
            sub_G.add_edge(i, j, weight=weight)
        # 更新进度条
        pbar.update(1)

# 关闭进度条
pbar.close()
with open(f'dataset/didi_{data_name}/traj_subg_{subg_id}.pkl', 'wb') as f:
            pkl.dump(sub_G, f)