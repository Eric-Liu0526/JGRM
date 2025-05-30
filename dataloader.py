#!/usr/bin/python
import os

import numpy as np
import torch
import warnings
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import pickle
import pandas as pd
import json
from torch.utils.data import BatchSampler
import random
import networkx as nx
from typing import List, Iterator

class CustomBatchSampler(BatchSampler):
    def __init__(self, batch_indices_list):
        self.batch_indices = batch_indices_list

    def __iter__(self):
        for batch in self.batch_indices:
            yield batch

    def __len__(self):
        return len(self.batch_indices)
    
def extract_weekday_and_minute_from_list(timestamp_list):
    weekday_list = []
    minute_list = []
    for timestamp in timestamp_list:
        dt = datetime.fromtimestamp(timestamp)
        weekday = dt.weekday() + 1
        minute = dt.hour * 60 + dt.minute + 1
        weekday_list.append(weekday)
        minute_list.append(minute)
    return torch.tensor(weekday_list).long(), torch.tensor(minute_list).long()

def prepare_gps_data(df,mat_padding_value,data_padding_value,max_len):
    """

    Args:
        df: cpath_list,
        mat_padding_value: default num_nodes
        data_padding_value: default 0

    Returns:
        gps_data: (batch, gps_max_length, num_features)
        gps_assign_mat: (batch, gps_max_length)

    """

    # padding opath_list
    opath_list = [torch.tensor(opath_list, dtype=torch.float32) for opath_list in df['opath_list']]
    gps_assign_mat = rnn_utils.pad_sequence(opath_list, padding_value=mat_padding_value,
                                                               batch_first=True)
    # padding gps point data
    data_package = []
    for col in df.drop(columns='opath_list').columns:
        features = df[col].tolist()
        features = [torch.tensor(f, dtype=torch.float32) for f in features]
        features = rnn_utils.pad_sequence(features, padding_value=torch.nan, batch_first=True)
        features = features.unsqueeze(dim=2)
        data_package.append(features)

    gps_data = torch.cat(data_package, dim=2)


    # todo 临时处理的方式, 把时间戳那维特征置1
    gps_data[:, :, 0] = torch.ones_like(gps_data[:, :, 0])

    # 对除第一维特征进行标准化
    for i in range(1, gps_data.shape[2]):
        fea = gps_data[:, :, i]
        nozero_fea = torch.masked_select(fea, torch.isnan(fea).logical_not())  # 计算不为nan的值的fea的mean与std
        gps_data[:, :, i] = (gps_data[:, :, i] - torch.mean(nozero_fea)) / torch.std(nozero_fea)

    # 把因为数据没有前置节点因此无法计算，加速度等特征的nan置0
    gps_data = torch.where(torch.isnan(gps_data), torch.full_like(gps_data, data_padding_value), gps_data)

    return gps_data, gps_assign_mat

def prepare_route_data(df,mat_padding_value,data_padding_value,max_len):
    """

    Args:
        df: cpath_list,
        mat_padding_value: default num_nodes
        data_padding_value: default 0

    Returns:
        route_data: (batch, route_max_length, num_features)
        route_assign_mat: (batch, route_max_length)

    """
    # padding capath_list
    cpath_list = [torch.tensor(cpath_list, dtype=torch.float32) for cpath_list in df['cpath_list']]
    route_assign_mat = rnn_utils.pad_sequence(cpath_list, padding_value=mat_padding_value,
                                                               batch_first=True)

    # padding route data
    weekday_route_list, minute_route_list = zip(
        *df['road_timestamp'].apply(extract_weekday_and_minute_from_list))

    weekday_route_list = [torch.tensor(weekday[:-1]).long() for weekday in weekday_route_list] # road_timestamp 比 route 本身的长度多1，包含结束的时间戳
    minute_route_list = [torch.tensor(minute[:-1]).long() for minute in minute_route_list]# road_timestamp 比 route 本身的长度多1，包含结束的时间戳
    weekday_data = rnn_utils.pad_sequence(weekday_route_list, padding_value=0, batch_first=True)
    minute_data = rnn_utils.pad_sequence(minute_route_list, padding_value=0, batch_first=True)

    # 分箱编码时间值
    # interval = []
    # df['road_interval'].apply(lambda row: interval.extend(row))
    # interval = np.array(interval)[~np.isnan(interval)]
    #
    # cuts = np.percentile(interval, [0, 2.5, 16, 50, 84, 97.5, 100])
    # cuts[0] = -1
    #
    # new_road_interval = []
    # for interval_list in df['road_interval']:
    #     new_interval_list = pd.cut(interval_list, cuts, labels=[1, 2, 3, 4, 5, 6])
    #     new_road_interval.append(torch.Tensor(new_interval_list).long())

    new_road_interval = []
    for interval_list in df['road_interval']:
        new_road_interval.append(torch.Tensor(interval_list).long())

    delta_data = rnn_utils.pad_sequence(new_road_interval, padding_value=-1, batch_first=True)

    route_data = torch.cat([weekday_data.unsqueeze(dim=2), minute_data.unsqueeze(dim=2), delta_data.unsqueeze(dim=2)], dim=-1)# (batch_size,max_len,2)

    # 填充nan
    route_data = torch.where(torch.isnan(route_data), torch.full_like(route_data, data_padding_value), route_data)

    return route_data, route_assign_mat

class StaticDataset(Dataset):
    def __init__(self, data, mat_padding_value, data_padding_value,gps_max_len,route_max_len):
        # 仅包含gps轨迹和route轨迹，route中包含路段的特征
        # 不包含路段过去n个时间戳的流量数据
        self.data = data
        self.traj_idx = data.index
        gps_length = data['opath_list'].apply(lambda opath_list: self._split_duplicate_subseq(opath_list, data['route_length'].max())).tolist()
        self.gps_length = torch.tensor(gps_length, dtype=torch.int)
        gps_data, gps_assign_mat = prepare_gps_data(data[['opath_list', 'tm_list',\
                                                          'lng_list', 'lat_list',\
                                                          'speed', 'acceleration',\
                                                          'angle_delta', 'interval',\
                                                          'dist']], mat_padding_value, data_padding_value, gps_max_len)

        # gps 点的信息，从tm_list，traj_list，speed，acceleration，angle_delta，interval，dist生成，padding_value = 0
        self.gps_data = gps_data # gps point shape = (num_samples,gps_max_length,num_features)

        # 表示gps点数据属于哪个路段，从opath_list生成，padding_value = num_nodes
        self.gps_assign_mat = gps_assign_mat #  shape = (num_samples,gps_max_length)

        # todo 路段本身的属性特征怎么放进去
        route_data, route_assign_mat = prepare_route_data(data[['cpath_list', 'road_timestamp','road_interval']],\
                                                          mat_padding_value, data_padding_value, route_max_len)

        # route对应的信息，从road_interval生成，padding_value = 0
        self.route_data = route_data # shape = (num_samples,route_max_length,1)

        # 表示路段的序列信息，从cpath_list生成，padding_value = num_nodes
        self.route_assign_mat = route_assign_mat # shape = (num_samples,route_max_length)

    def _split_duplicate_subseq(self,opath_list,max_len):
        length_list = []
        subsequence = [opath_list[0]]
        for i in range(0, len(opath_list)-1):
            if opath_list[i] == opath_list[i+1]:
                subsequence.append(opath_list[i])
            else:
                length_list.append(len(subsequence))
                subsequence = [opath_list[i]]
        length_list.append(len(subsequence))
        return length_list + [0]*(max_len-len(length_list))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.gps_data[idx], self.gps_assign_mat[idx], self.route_data[idx], self.route_assign_mat[idx],
                self.gps_length[idx], self.traj_idx[idx])

# class DynamicDataset_(Dataset):
#     def __init__(self, data, edge_features, traffic_flow_history):
#         # 仅包含gps轨迹和route轨迹，route中包含路段的特征
#         # 不包含路段过去n个时间戳的流量数据
#         self.data = data
#
#         # 特征矩阵和指示矩阵
#         # gps 点的信息，从tm_list，traj_list，speed，acceleration，angle_delta，interval，dist生成，padding_value = 0
#         self.gps_data = # gps point shape = (num_samples,gps_max_length,features)
#
#         # 表示gps点数据属于哪个路段，从opath_list生成，padding_value = num_nodes
#         self.gps_assign_mat = #  shape = (num_samples,gps_max_length)
#
#         # # route features
#         # # cpath_list
#         # # road_interval
#         # # 关联道路特征图有其他特征
#
#         # route对应的信息，从road_interval 和 edge_features生成，包含静态特征和动态特征，静态特征为路段特征，动态特征为历史流量信息，padding_value = 0
#         self.route_data = # shape = (num_samples,route_max_length,num_features)
#         self.route_assign_mat = # shape = (num_samples,route_max_length)
#
#         self.data_lst = []
#         self.data.groupby('pairid').apply(lambda group: self.data_lst.append(torch.tensor(np.array(group.iloc[:,:-2]))))
#         self.x_mbr = pad_sequence(self.data_lst, batch_first=True) # set_count*max_len*fea_dim
#         self.x_c = self.x_mbr
#         self.label_idxs = self.data.groupby('pairid').apply(lambda group:find_label_idx(group)).values # label=1 ndarray
#         self.lengths = self.data.groupby('pairid')['label'].count().values # ndarray
#
#     def __len__(self):
#         return len(self.data_lst)
#
#     def __getitem__(self, idx):
#         return (self.x_mbr[idx],self.x_c[idx],self.label_idxs[idx],self.lengths[idx])

'''
def get_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed):

    dataset = pickle.load(open(data_path, 'rb'))
    print(dataset.columns)

    dataset['route_length'] = dataset['cpath_list'].map(len)
    dataset = dataset[
        (dataset['route_length'] > route_min_len) & (dataset['route_length'] < route_max_len)].reset_index(drop=True)

    dataset['gps_length'] = dataset['opath_list'].map(len)
    dataset = dataset[
        (dataset['gps_length'] > gps_min_len) & (dataset['gps_length'] < gps_max_len)].reset_index(drop=True)

    print(dataset.shape)
    assert dataset.shape[0] >= num_samples

    # 获取最大路段id
    uniuqe_path_list = []
    dataset['cpath_list'].apply(lambda cpath_list: uniuqe_path_list.extend(list(set(cpath_list))))
    uniuqe_path_list = list(set(uniuqe_path_list))

    mat_padding_value = max(uniuqe_path_list) + 1
    data_padding_value = 0.0

    dataset['flag'] = pd.to_datetime(dataset['start_time'], unit='s').dt.day
    # 前13天作为训练集，第14天作为测试集，第15天作为验证集
    train_data, test_data, val_data =dataset[dataset['flag']<14], dataset[dataset['flag']==14], dataset[dataset['flag']==15]
    # train_data = train_data.sample(n=num_samples, replace=False, random_state=seed)

    # notice: 一般情况下

    train_dataset = StaticDataset(train_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    test_dataset = StaticDataset(test_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    val_dataset = StaticDataset(val_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)

    batch_sampler = CustomBatchSampler(split_batches(batch_size))
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_worker)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)

    return train_loader, val_loader, test_loader
'''

def sample_batch(graph, batch_size):
    """
    从图中进行邻居采样，生成一个batch。

    Args:
        graph (networkx.Graph): 当前子图。
        batch_size (int): 每个批次需要采样的节点数量。

    Returns:
        sub_batch (list): 当前批次的节点ID列表。
        node_features (torch.Tensor): 当前批次节点的特征矩阵。
        adj_matrix (torch.Tensor): 当前批次的邻接矩阵。
    """
    # 1. 随机选择batch_size个节点
    nodes = list(graph.nodes)
    sampled_nodes = random.sample(nodes, batch_size)
    
    # 2. 为每个采样的节点选择邻居（你可以选择固定数量的邻居）
    #    这里我们简单地采样每个节点的所有邻居，也可以加一些采样技巧来限制邻居的数量
    subgraph_nodes = set(sampled_nodes)  # 初步选择的子图节点集合
    for node in sampled_nodes:
        neighbors = list(graph.neighbors(node))  # 获取邻居
        subgraph_nodes.update(neighbors)  # 将邻居加入子图节点集合

    # 3. 创建子图
    subgraph = graph.subgraph(subgraph_nodes)
    
    # 4. 获取子图的邻接矩阵
    adj_matrix = nx.to_numpy_matrix(subgraph)  # 邻接矩阵（numpy）
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)  # 转为Tensor
    
    # 5. 获取节点特征
    # 假设每个节点都有一个 'feature' 属性，返回每个节点的特征
    node_features = []
    for node in subgraph_nodes:
        feature = graph.nodes[node].get('feature', torch.zeros(1))  # 这里可以替换为节点的实际特征
        node_features.append(feature)
    
    node_features = torch.stack(node_features)  # 转为Tensor，维度 (num_nodes, feature_dim)
    
    return sampled_nodes, node_features, adj_matrix


from typing import List

def sample_by_degree(graph: nx.Graph, batch_size: int) -> List[List]:
    """
    根据节点度从大到小排序，按顺序选择节点及其邻居，生成批次。
    每次采样一个节点及其batch_size-1个邻居，邻居可以是多阶的，只有一阶邻居会被标记。
    
    参数:
        graph (nx.Graph): 输入的networkx图
        batch_size (int): 每个batch的节点数量
        
    返回:
        List[List]: 包含多个batch的列表，每个batch是节点ID的列表
    """
    # 获取图中所有节点并按度排序，度大的节点优先
    nodes = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
    
    # 创建一个已标记的节点集合，用于标记已采样的一阶邻居
    sampled = set()
    batches = []
    for node in nodes:
        if node not in sampled:  # 如果当前节点未被标记
            # 采样当前节点
            batch = [node]
            sampled.add(node)  # 标记当前节点为已采样
            
            # 当前节点的所有邻居
            neighbors = list(graph.neighbors(node))
            
            # 如果邻居数量不足，尝试从更高阶的邻居中采样
            extra_neighbors = []  # 用于存储更多阶的邻居
            level = 1
            while len(batch) < batch_size and extra_neighbors:
                # 获取当前level的邻居
                next_level_neighbors = []
                for neighbor in extra_neighbors:
                    next_level_neighbors.extend(graph.neighbors(neighbor))
                
                # 过滤掉已采样的节点
                next_level_neighbors = [n for n in next_level_neighbors if n not in sampled]
                
                # 还剩余需要的邻居
                extra_neighbors = next_level_neighbors
                batch.extend(extra_neighbors)
                level += 1  # Repeat


def random_batch_nodes(graph: nx.Graph, batch_size: int) -> List[List]:
    """
    将图的节点随机划分为多个batch，每个batch大小为batch_size，尽可能覆盖所有节点。
    
    参数:
        graph (nx.Graph): 输入的networkx图
        batch_size (int): 每个batch的节点数量
        
    返回:
        List[List]: 包含多个batch的列表，每个batch是节点ID的列表
    """
    # 获取图中所有节点的列表
    nodes = list(graph.nodes())
    np.random.shuffle(nodes)  # 随机打乱节点顺序
    batches = []
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]  # 切片获取当前batch
        if len(batch) == batch_size:
            batches.append(batch)
    return batches

'''
def split_batches(batch_size=32):
    batch2g_dict = dict()
    batches = []
    # 读取子图轨迹字典：{sub_g_id: traj_set}
    with open(f'dataset/didi_chengdu/sub_g_traj_dict.pkl', 'rb') as f:
        sub_g_traj_dict = pickle.load(f)
    # 通过子图采样划分训练batch
    for sub_g_id in sub_g_traj_dict.keys():
        graph = pickle.load(open(f'dataset/didi_chengdu/traj_subg_{sub_g_id}.pkl', 'rb'))
        # sub_batches = sample_by_degree(graph, batch_size)
        sub_batches = random_batch_nodes(graph, batch_size)
        for sub_batch in sub_batches:
            batch2g_dict[len(batch2g_dict)] = sub_g_id
            batches.append(sub_batch)
    return batch2g_dict, batches
'''
def k_hop_sampling(G, node, k, max_sample_size):
    """执行K-hop邻居采样"""
    sampled_nodes = {node}
    for _ in range(k):
        neighbors = set()
        for n in sampled_nodes:
            neighbors.update(G.neighbors(n))  # 添加该节点的邻居
        sampled_nodes.update(neighbors)
        if len(sampled_nodes) >= max_sample_size:
            break
    return list(sampled_nodes)[:max_sample_size]

def split_batches(batch_size=64, pos_ratio=0.5, k=2, seed=42):
    batch2g_dict = {}
    batches = []
    
    # 读取子图轨迹字典：{sub_g_id: traj_set}
    with open(f'dataset/didi_chengdu/sub_g_traj_dict.pkl', 'rb') as f:
        sub_g_traj_dict = pickle.load(f)
    
    graph_dict = dict()
    # 读取每个子图的图数据
    for sub_g_id in sub_g_traj_dict.keys():
        graph = pickle.load(open(f'dataset/didi_chengdu/traj_subg_{sub_g_id}.pkl', 'rb'))
        graph_dict[sub_g_id] = graph
    
    random.seed(seed)
    
    # 计算正负例的节点数
    pos_size = int(batch_size * pos_ratio)
    neg_size = batch_size - pos_size
    batch_id = 0

    for subg_id, G in graph_dict.items():
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        
        # 确保每个子图至少能生成一个batch
        num_batches = (num_nodes + batch_size - 1) // batch_size  # 向上取整
        
        print(f"子图 {subg_id} 有 {num_nodes} 个节点, 将生成 {num_batches} 个batch")
        
        for _ in range(num_batches):
            if len(nodes) < pos_size:
                continue  # 正例节点太少就跳过
            
            # 正例采样（K-hop采样确保拓扑关系）
            pos_sample = []
            for node in nodes:
                pos_sample.extend(k_hop_sampling(G, node, k, pos_size))
            pos_sample = list(set(pos_sample))  # 去重，避免重复
            if len(pos_sample) < pos_size:
                continue  # 如果正例不够，跳过

            # 负例采样
            other_subg_ids = list(set(graph_dict.keys()) - {subg_id})
            other_nodes = []
            for oid in other_subg_ids:
                other_nodes.extend(list(graph_dict[oid].nodes()))

            if len(other_nodes) < neg_size:
                continue  # 太少就跳过

            neg_sample = random.sample(other_nodes, neg_size)

            # 构建batch
            batch_nodes = pos_sample[:pos_size] + neg_sample
            random.shuffle(batch_nodes)

            # 保存batch和对应的子图信息
            batches.append(batch_nodes)
            batch2g_dict[batch_id] = subg_id
            batch_id += 1

    return batch2g_dict, batches



def get_train_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed):

    dataset = pickle.load(open(data_path, 'rb'))
    print(dataset.columns)

    dataset['route_length'] = dataset['cpath_list'].map(len)
    dataset = dataset[
        (dataset['route_length'] > route_min_len) & (dataset['route_length'] < route_max_len)].reset_index(drop=True)

    dataset['gps_length'] = dataset['opath_list'].map(len)
    dataset = dataset[
        (dataset['gps_length'] > gps_min_len) & (dataset['gps_length'] < gps_max_len)].reset_index(drop=True)
    

    print(dataset.shape)
    print(num_samples)
    assert dataset.shape[0] >= num_samples

    # 获取最大路段id
    uniuqe_path_list = []
    dataset['cpath_list'].apply(lambda cpath_list: uniuqe_path_list.extend(list(set(cpath_list))))
    uniuqe_path_list = list(set(uniuqe_path_list))

    mat_padding_value = max(uniuqe_path_list) + 1
    data_padding_value = 0.0

    # 前13天作为训练集，第14天作为测试集，第15天作为验证集，已经提前分好
    train_data = dataset
    # train_data = train_data.sample(n=num_samples, replace=False, random_state=seed)

    train_dataset = StaticDataset(train_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    batch2g_dict, batches = split_batches(batch_size)
    batch_sampler = CustomBatchSampler(batches)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=num_worker)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)

    return batch2g_dict, train_loader

def random_mask(gps_assign_mat, route_assign_mat, gps_length, mask_token, mask_length=1, mask_prob=0.2):

    # mask route
    col_num = int(route_assign_mat.shape[1] / mask_length) + 1
    batch_size = route_assign_mat.shape[0]

    # mask的位置和padding的位置有重合，但整体mask概率无影响
    route_mask_pos = torch.empty(
        (batch_size, col_num),
        dtype=torch.float32,
        device=route_assign_mat.device).uniform_(0, 1) < mask_prob

    route_mask_pos = torch.stack(sum([[col]*mask_length for col in route_mask_pos.t()], []), dim=1)

    # 截断
    if route_mask_pos.shape[1] > route_assign_mat.shape[1]:
        route_mask_pos = route_mask_pos[:, :route_assign_mat.shape[1]]

    masked_route_assign_mat = route_assign_mat.clone()
    masked_route_assign_mat[route_mask_pos] = mask_token

    # mask gps
    masked_gps_assign_mat = gps_assign_mat.clone()
    gps_mask_pos = []
    for idx, row in enumerate(gps_assign_mat):
        route_mask = route_mask_pos[idx]
        length_list = gps_length[idx]
        unpad_mask_pos_list = sum([[mask] * length_list[_idx].item() for _idx, mask in enumerate(route_mask)], [])
        pad_mask_pos_list = unpad_mask_pos_list + [torch.tensor(False)] * (
                    gps_assign_mat.shape[1] - len(unpad_mask_pos_list))
        pad_mask_pos = torch.stack(pad_mask_pos_list)
        gps_mask_pos.append(pad_mask_pos)
    gps_mask_pos = torch.stack(gps_mask_pos, dim=0)
    masked_gps_assign_mat[gps_mask_pos] = mask_token
    # 获得每个gps点对应路段的长度

    return masked_route_assign_mat, masked_gps_assign_mat
