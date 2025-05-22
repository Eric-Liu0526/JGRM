import pandas as pd
import pickle as pkl
from datetime import datetime
from tqdm import tqdm
import numpy as np

def hot_roads(edge_feat_path, traj_data_path):
    """
    统计每个路段的时段通行量
    输入：
        edge_feat_path(str): 路段特征文件路径(csv)
        traj_data_path(str): 轨迹数据文件路径(pkl),pkl中为df数据
    """
    road_hour_score_dict = dict()
    road_od_score_dict = dict()
    traj_acceleration_dict = dict()
    # 获取路段id
    edge_feat = pd.read_csv(edge_feat_path)
    fids = edge_feat['fid'].tolist()
    for fid in fids:
        road_hour_score_dict[fid] = [0] * 24
        road_od_score_dict[fid] = (0, 0)
    # 获取轨迹数据
    traj_data = pkl.load(open(traj_data_path, 'rb'))
    # 统计每个路段的时段通行量
    for index, row in tqdm(traj_data.iterrows(), total=len(traj_data), desc='统计锚点得分'):
        # 1. 统计路段时段通行量
        cpath_list = row['cpath_list']
        road_timestamp = row['road_timestamp']
        for i in range(len(cpath_list)):
            timestamp = road_timestamp[i]
            hour = datetime.fromtimestamp(timestamp).hour
            road_hour_score_dict[cpath_list[i]][hour] += 1

        # 2. 统计路段od量
        start_fid = cpath_list[0]
        end_fid = cpath_list[-1]
        road_od_score_dict[start_fid] = (road_od_score_dict[start_fid][0] + 1, road_od_score_dict[start_fid][1])
        road_od_score_dict[end_fid] = (road_od_score_dict[end_fid][0], road_od_score_dict[end_fid][1] + 1)
        
        # 3. 统计轨迹加速度方差
        tid = row['tid']
        acceleration_list = row['acceleration']
        acceleration_list = acceleration_list[2:]
        # 计算加速度方差
        acceleration_var = np.var(acceleration_list)
        traj_acceleration_dict[tid] = acceleration_var

    # 保存结果
    with open(f'dataset/didi_{city_name}/road_hour_score_dict.pkl', 'wb') as f:
        pkl.dump(road_hour_score_dict, f)
    with open(f'dataset/didi_{city_name}/road_od_score_dict.pkl', 'wb') as f:
        pkl.dump(road_od_score_dict, f)
    with open(f'dataset/didi_{city_name}/traj_acceleration_dict.pkl', 'wb') as f:
        pkl.dump(traj_acceleration_dict, f)

def hour_trajs(traj_data_path):
    """
    统计每个时段的轨迹数量
    输入：
        traj_data_path(str): 轨迹数据文件路径(pkl),pkl中为df数据
    """
    traj_data = pkl.load(open(traj_data_path, 'rb'))
    hour_trajs_dict = dict()
    for i in range(24):
        hour_trajs_dict[i] = 0
    # 统计每个时段的轨迹数量
    for index, row in tqdm(traj_data.iterrows(), total=len(traj_data), desc='统计每个时段的轨迹数量'):
        start_time = row['start_time']
        hour = datetime.fromtimestamp(start_time).hour
        hour_trajs_dict[hour] += 1
    print(hour_trajs_dict)

def similarity_number(similarity_matrix_path, similarity_rank_path):
    """
    统计相似度矩阵中每个轨迹在不同锚轨迹中排名靠前10%出现的次数
    """
    similarity_rank = pkl.load(open(similarity_rank_path, 'rb'))
    traj_num = similarity_rank.shape[1]
    similarity_number = [0] * traj_num
    for traj_idx in tqdm(range(traj_num), desc='统计每个轨迹在不同锚轨迹中排名靠前10%出现的次数'):
        for anchor_idx in range(similarity_rank.shape[0]):
            if np.where(similarity_rank[anchor_idx] == traj_idx)[0] < similarity_rank.shape[1] * 0.1:
                similarity_number[traj_idx] += 1
    
    # 将similarity_number按从大到小排序
    similarity_number.sort(reverse=True)
    print(similarity_number)


if __name__ == '__main__':
    city_name = 'chengdu'
    edge_feat_path = f'dataset/didi_{city_name}/edge_features.csv'
    traj_data_path = f'dataset/didi_{city_name}/{city_name}_1101_1115_data_sample10w.pkl'
    # hot_roads(edge_feat_path, traj_data_path)
    # hour_trajs(traj_data_path)
    similarity_matrix_path = f'dataset/didi_{city_name}/similarity_matrix.pkl'
    similarity_rank_path = f'dataset/didi_{city_name}/similarity_rank.pkl'
    similarity_number(similarity_matrix_path, similarity_rank_path)
    