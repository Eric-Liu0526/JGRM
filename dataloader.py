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
import pickle as pkl
import torch.nn.functional as F

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

        # 轨迹id
        self.tid_list = data['tid'].tolist()

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
                self.gps_length[idx], self.tid_list[idx])

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
    train_data = train_data.sample(n=num_samples, replace=False, random_state=seed)

    # notice: 一般情况下

    train_dataset = StaticDataset(train_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    test_dataset = StaticDataset(test_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)
    val_dataset = StaticDataset(val_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)

    return train_loader, val_loader, test_loader

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
    train_data = train_data.sample(n=num_samples, replace=False, random_state=seed)

    train_dataset = StaticDataset(train_data, mat_padding_value, data_padding_value,gps_max_len,route_max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)

    return train_loader

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

class DynamicAnchorMemory(Dataset):
    def __init__(self, data_path, feature_dim=256, temperature=0.07):
        """
        初始化动态记忆库
        
        参数:
            memory_size: 记忆库大小
            feature_dim: 特征维度
            temperature: 温度参数,用于计算相似度
        """
        self.memory_size = None
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # 初始化记忆库, 存储tid和对应的向量表征
        self.memory = dict()

        # 存储相似度矩阵和相似度排名
        self.similarity_matrix = None
        self.similarity_rank = None
        
        # 存储锚轨迹的tid和密度
        self.anchor_tids = None
        self.anchor_densities = None

        anchor_indices_path = f'{data_path}/anchor_indices_and_densities.pkl'
        similarity_matrix_path = f'{data_path}/similarity_matrix.pkl'
        similarity_rank_path = f'{data_path}/similarity_rank.pkl'
        self.set_anchor_indices(anchor_indices_path)
        self.set_similarity_matrix(similarity_matrix_path, similarity_rank_path)
        self.train_loader = None

    @staticmethod
    def get_anchor_loader(train_loader, anchor_indices_path):
        """
        从训练数据加载器中获取锚轨迹数据
        
        参数:
            train_loader: 训练数据加载器
            anchor_indices_path: 锚轨迹索引文件路径
            
        返回:
            锚轨迹数据加载器
        """
        # 从anchor_indices_path中获取锚轨迹的anchor_tids
        anchor_tids, _ = pkl.load(open(anchor_indices_path, 'rb'))
        # 获取anchor_tids在train_loader.dataset中的索引
        anchor_indices = [train_loader.dataset.tid_list.index(tid) for tid in anchor_tids]
        
        # 创建锚轨迹数据集
        anchor_dataset = torch.utils.data.Subset(train_loader.dataset, anchor_indices)
        
        # 创建数据加载器，保持与train_loader相同的参数
        anchor_loader = torch.utils.data.DataLoader(
            anchor_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return anchor_loader

    def get_similarity_based_loader(self, train_loader, batch_size, num_positive):
        """
        构建基于相似度的训练数据加载器
        
        参数:
            train_loader: 原始训练数据加载器
            batch_size: batch大小
            num_positive: 每个batch中正样本的数量
            
        返回:
            新的训练数据加载器
        """
        if self.similarity_matrix is None:
            raise ValueError("similarity_matrix is not set")
        if self.similarity_rank is None:
            raise ValueError("similarity_rank is not set")
        if self.anchor_tids is None:
            raise ValueError("anchor_tids is not set")
        
        # 创建新的数据集
        dataset = SimilarityBasedDataset(
            train_loader.dataset,
            self.similarity_rank,
            self.anchor_tids,
            batch_size,
            num_positive
        )
        
        # 创建数据加载器，保持与train_loader相同的参数
        new_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_loader.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.train_loader = new_loader
        return new_loader

    def set_anchor_indices(self, anchor_indices_path):
        self.anchor_tids, self.anchor_densities = pkl.load(open(anchor_indices_path, 'rb'))
        self.memory_size = len(self.anchor_tids)
    
    def update_memory(self, vector_dict):
        """
        更新记忆库中的表征
        
        参数:
            vector_dict: 包含锚轨迹tid和对应的向量表征的字典{tid: vector}
        """
        with torch.no_grad():
            # 更新记忆库中的表征
            for tid, vector in vector_dict.items():
                self.memory[tid] = vector
                
    def update_memory(self, tid_list, gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep):
        """
        更新记忆库中的表征
        
        参数:
            tid_list: 包含锚轨迹tid的列表
            gps_road_rep: 锚轨迹的gps_road_rep
            gps_traj_rep: 锚轨迹的gps_traj_rep
            route_road_rep: 锚轨迹的route_road_rep
            route_traj_rep: 锚轨迹的route_traj_rep
            gps_road_joint_rep: 锚轨迹的gps_road_joint_rep
            gps_traj_joint_rep: 锚轨迹的gps_traj_joint_rep
            route_road_joint_rep: 锚轨迹的route_road_joint_rep
        """
        with torch.no_grad():
            # 更新记忆库中的表征
            for tid, gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep in zip(tid_list, gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep):
                self.memory[tid] = {
                    'gps_road_rep': gps_road_rep,
                    'gps_traj_rep': gps_traj_rep,
                    'route_road_rep': route_road_rep,
                    'route_traj_rep': route_traj_rep,
                    'gps_road_joint_rep': gps_road_joint_rep,
                    'gps_traj_joint_rep': gps_traj_joint_rep,
                    'route_road_joint_rep': route_road_joint_rep,
                    'route_traj_joint_rep': route_traj_joint_rep,
                }
                
    def get_memory(self, tids=None):
        """
        获取记忆库中的表征
        
        参数:
            tids: 需要获取的tid,如果为None则返回所有表征
            
        返回:
            记忆库中的表征
        """
        if tids is None:
            return self.memory
        elif isinstance(tids, list):
            return {tid: self.memory[tid] for tid in tids}
        elif isinstance(tids, int):
            return self.memory[tids]
        else:
            raise ValueError(f"Invalid type for tids: {type(tids)}")
    
    def set_similarity_matrix(self, similarity_matrix_path, similarity_rank_path):
        """
        设置相似度矩阵和相似度排名
        
        参数:
            similarity_matrix_path: 相似度矩阵路径
            similarity_rank_path: 相似度排名路径
        返回:
            相似度矩阵
            相似度排名
        """
        similarity_matrix = pkl.load(open(similarity_matrix_path, 'rb'))
        similarity_rank = pkl.load(open(similarity_rank_path, 'rb'))
        self.similarity_matrix = similarity_matrix
        self.similarity_rank = similarity_rank
        return similarity_matrix, similarity_rank
    
    def knowledge_transfer(self, query_features, top_k=5):
        """
        基于相似度进行知识迁移
        
        参数:
            query_features: 查询特征
            top_k: 选择最相似的k个锚轨迹
            
        返回:
            迁移后的特征
        """
        # 计算相似度
        similarity = self.compute_similarity(query_features)
        
        # 选择top-k个最相似的锚轨迹
        top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=1)
        
        # 获取对应的记忆库表征
        memory_features = self.get_memory(top_k_indices)
        
        # 加权融合
        weights = F.softmax(top_k_values, dim=1).unsqueeze(-1)
        transferred_features = torch.sum(weights * memory_features, dim=1)
        
        return transferred_features
    
    def update_access_frequency(self, indices):
        """
        更新访问频率
        
        参数:
            indices: 被访问的锚轨迹索引
        """
        self.access_frequency[indices] += 1
        
    def get_least_frequent_indices(self, n):
        """
        获取访问频率最低的n个索引
        
        参数:
            n: 需要获取的索引数量
            
        返回:
            访问频率最低的n个索引
        """
        _, indices = torch.topk(-self.access_frequency, k=n)
        return indices
    
    def __len__(self):
        return len(self.anchor_indices)
    
    def __getitem__(self, idx):
        return self.memory[idx]

class SimilarityBasedDataset(Dataset):
    def __init__(self, original_dataset, similarity_rank, anchor_tids, batch_size, num_positive):
        """
        基于相似度的数据集
        
        参数:
            original_dataset: 原始数据集
            similarity_rank: 相似度排名矩阵，形状为 (n_anchors, n_trajectories)
            anchor_tids: 锚轨迹的tid列表
            batch_size: batch大小
            num_positive: 每个batch中正样本的数量
        """
        self.original_dataset = original_dataset
        self.similarity_rank = similarity_rank
        self.anchor_tids = anchor_tids
        self.batch_size = batch_size
        self.num_positive = num_positive
        self.num_negative = batch_size - num_positive - 1  # 减1是因为还要包含锚轨迹
        
        # 计算每个锚轨迹可以生成的batch数量
        self.num_all = len(original_dataset)
        num_anchor_batch_traj = self.num_all // len(self.anchor_tids) + 1
        self.num_anchor_batch = num_anchor_batch_traj // self.batch_size + 1
        
        # 记录每个轨迹被训练的次数
        self.train_count = [0] * self.num_all

        # 构建所有可能的batch
        self.batches = []
        for anchor_idx, anchor_tid in enumerate(self.anchor_tids):
            for i in range(self.num_anchor_batch):
                # 获取正样本索引
                pos_start_idx = i * self.num_positive
                pos_indices = self.similarity_rank[anchor_idx, pos_start_idx:pos_start_idx+self.num_positive]
                for index in pos_indices:
                    self.train_count[index] += 1
                # 获取负样本索引
                neg_start_idx = self.num_all - (self.num_anchor_batch - i) * self.num_negative
                neg_indices = self.similarity_rank[anchor_idx, neg_start_idx:neg_start_idx+self.num_negative]
                for index in neg_indices:
                    self.train_count[index] += 1
                
                # 获取锚轨迹索引
                anchor_idx_in_dataset = self.original_dataset.tid_list.index(anchor_tid)
                
                # 组合成一个batch
                batch_indices = np.concatenate([[anchor_idx_in_dataset], pos_indices, neg_indices])
                self.batches.append(batch_indices)
        # 将batches合成一个list
        self.batches = [item for sublist in self.batches for item in sublist]
        # 将train_count写入txt文件
        with open('train_count.txt', 'w') as f:
            for count in self.train_count:
                f.write(f"{count}\n")
            # 将train_count中的0的个数写入txt文件
            f.write(f"0的个数: {self.train_count.count(0)}\n")
    
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        # batch_indices = self.batches[idx]
        # # 获取batch中的所有数据
        # gps_data_list = []
        # gps_assign_mat_list = []
        # route_data_list = []
        # route_assign_mat_list = []
        # gps_length_list = []
        # tid_list = []
        
        # for index in batch_indices:
        #     gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length, tid = self.original_dataset[index]
        #     gps_data_list.append(gps_data)
        #     gps_assign_mat_list.append(gps_assign_mat)
        #     route_data_list.append(route_data)
        #     route_assign_mat_list.append(route_assign_mat)
        #     gps_length_list.append(gps_length)
        #     tid_list.append(tid)
        
        # # 将列表转换为张量
        # gps_data = torch.stack(gps_data_list)
        # gps_assign_mat = torch.stack(gps_assign_mat_list)
        # route_data = torch.stack(route_data_list)
        # route_assign_mat = torch.stack(route_assign_mat_list)
        # gps_length = torch.stack(gps_length_list)
        
        # return gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length, tid_list
        
        return self.original_dataset[self.batches[idx]]

class DynamicBatchDataset(Dataset):
    def __init__(self, original_dataset, similarity_matrix, anchor_tids, 
                 batch_anchor_size=32, num_positive=5, num_negative=10, 
                 neg_threshold=0.5, top_k=100):
        """
        动态批次划分数据集
        
        参数:
            original_dataset: 原始数据集
            similarity_matrix: 相似度矩阵，格式为 {anchor_tid: [(traj_tid, score), ...]}
            anchor_tids: 锚轨迹ID列表
            batch_anchor_size: 每个批次中的锚轨迹数量
            num_positive: 每个锚轨迹的正样本数量
            num_negative: 每个锚轨迹的负样本数量
            neg_threshold: 负样本的最大相似度阈值
            top_k: 每个锚轨迹保留的候选轨迹数量
        """
        self.original_dataset = original_dataset
        self.similarity_matrix = similarity_matrix
        self.anchor_tids = anchor_tids
        self.batch_anchor_size = batch_anchor_size
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.neg_threshold = neg_threshold
        self.top_k = top_k
        
        # 预处理：为每个锚轨迹保留top-K个候选轨迹
        self.anchor_to_topK = {}
        for anchor_tid in self.anchor_tids:
            # 获取相似度排序后的轨迹列表
            sorted_trajs = sorted(self.similarity_matrix[anchor_tid].items(), 
                                key=lambda x: x[1], reverse=True)
            # 只保留top-K个候选轨迹
            self.anchor_to_topK[anchor_tid] = [tid for tid, _ in sorted_trajs[:top_k]]
        
        # 获取所有轨迹ID
        self.all_traj_ids = original_dataset.tid_list
        
        # 初始化批次列表
        self.batches = []
        self.current_epoch = 0
        
        # 生成初始批次
        self._generate_batches()
    
    def _generate_batches(self):
        """生成新的批次"""
        self.batches = []
        
        # 随机打乱锚轨迹
        anchor_pool = self.anchor_tids.copy()
        np.random.shuffle(anchor_pool)
        
        # 按批次大小划分锚轨迹
        for i in range(0, len(anchor_pool), self.batch_anchor_size):
            batch_anchors = anchor_pool[i:i + self.batch_anchor_size]
            batch = []
            
            for anchor_tid in batch_anchors:
                # 正样本采样（滑动窗口）
                pos_pool = self.anchor_to_topK[anchor_tid]
                start_idx = self.current_epoch % (len(pos_pool) - self.num_positive + 1)
                positives = pos_pool[start_idx:start_idx + self.num_positive]
                
                # 负样本采样
                excluded = set(positives + [anchor_tid])
                # 从相似度矩阵中获取低相似度的候选轨迹
                neg_candidates = [
                    tid for tid, score in self.similarity_matrix[anchor_tid].items()
                    if score < self.neg_threshold and tid not in excluded
                ]
                
                # 如果负样本候选不足，从全局随机采样补充
                if len(neg_candidates) < self.num_negative:
                    backup = [tid for tid in self.all_traj_ids if tid not in excluded]
                    neg_candidates.extend(np.random.choice(backup, 
                                                         size=self.num_negative - len(neg_candidates),
                                                         replace=False))
                
                # 随机选择负样本
                negatives = np.random.choice(neg_candidates, 
                                           size=self.num_negative, 
                                           replace=False)
                
                # 将锚轨迹、正样本和负样本添加到批次中
                batch.append((anchor_tid, positives, negatives))
            
            self.batches.append(batch)
    
    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.current_epoch = epoch
        self._generate_batches()
    
    def __len__(self):
        """返回批次数量"""
        return len(self.batches)
    
    def __getitem__(self, idx):
        """获取指定批次的数据"""
        batch = self.batches[idx]
        
        # 收集批次中的所有轨迹ID
        all_tids = []
        for anchor_tid, positives, negatives in batch:
            all_tids.extend([anchor_tid] + positives.tolist() + negatives.tolist())
        
        # 获取所有轨迹的数据
        batch_data = []
        for tid in all_tids:
            # 在原始数据集中查找轨迹
            traj_idx = self.original_dataset.tid_list.index(tid)
            traj_data = self.original_dataset[traj_idx]
            batch_data.append(traj_data)
        
        # 将数据组织成批次格式
        gps_data = torch.stack([data[0] for data in batch_data])
        gps_assign_mat = torch.stack([data[1] for data in batch_data])
        route_data = torch.stack([data[2] for data in batch_data])
        route_assign_mat = torch.stack([data[3] for data in batch_data])
        gps_length = torch.stack([data[4] for data in batch_data])
        tid_list = [data[5] for data in batch_data]
        
        # 返回批次数据
        return gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length, tid_list

def get_dynamic_batch_loader(data_path, similarity_matrix_path, anchor_indices_path,
                           batch_anchor_size=32, num_positive=5, num_negative=10,
                           neg_threshold=0.5, top_k=100, num_workers=4):
    """
    获取动态批次数据加载器
    
    参数:
        data_path: 数据路径
        similarity_matrix_path: 相似度矩阵路径
        anchor_indices_path: 锚轨迹索引路径
        batch_anchor_size: 每个批次中的锚轨迹数量
        num_positive: 每个锚轨迹的正样本数量
        num_negative: 每个锚轨迹的负样本数量
        neg_threshold: 负样本的最大相似度阈值
        top_k: 每个锚轨迹保留的候选轨迹数量
        num_workers: 数据加载的工作进程数
    """
    # 加载数据
    dataset = pkl.load(open(data_path, 'rb'))
    similarity_matrix = pkl.load(open(similarity_matrix_path, 'rb'))
    anchor_tids, _ = pkl.load(open(anchor_indices_path, 'rb'))
    
    # 创建原始数据集
    original_dataset = StaticDataset(dataset, mat_padding_value=dataset['cpath_list'].max() + 1,
                                   data_padding_value=0.0, gps_max_len=dataset['opath_list'].map(len).max(),
                                   route_max_len=dataset['cpath_list'].map(len).max())
    
    # 创建动态批次数据集
    dynamic_dataset = DynamicBatchDataset(
        original_dataset=original_dataset,
        similarity_matrix=similarity_matrix,
        anchor_tids=anchor_tids,
        batch_anchor_size=batch_anchor_size,
        num_positive=num_positive,
        num_negative=num_negative,
        neg_threshold=neg_threshold,
        top_k=top_k
    )
    
    # 创建数据加载器
    loader = DataLoader(
        dynamic_dataset,
        batch_size=1,  # 因为批次已经在数据集中组织好了
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

