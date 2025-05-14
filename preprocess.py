import numpy as np
import networkx as nx
import pickle as pkl
import pandas as pd

# 文件路径
city_name = 'chengdu'
npy_file_path = f'dataset/didi_{city_name}/line_graph_edge_idx.npy'
edge_features_path = f'dataset/didi_{city_name}/edge_features.csv'
edge_geometry_path = f'dataset/didi_{city_name}/edge_geometry.csv'
traj_path = f'dataset/didi_{city_name}/{city_name}_1101_1115_data_sample10w.pkl'
transition_prob_mat_path = f'dataset/didi_{city_name}/transition_prob_mat.npy'

def build_roadcross_graph(edge_features_path, edge_geometry_path):
    """
    构建道路交叉口图

    参数：
        edge_features_path(str): 路段特征文件路径
        edge_geometry_path(str): 路段几何文件路径
    
    返回：
        G(networkx.Graph): 道路交叉口图，G=(V, E)，V为路段端点集合，E为边集合，E={(u, v, key, length)}
    """
    # 读取路段长度
    road_features_pd = pd.read_csv(edge_features_path)
    road_dict = {}
    for index, row in road_features_pd.iterrows():
        road_dict[row['fid']] = row['length']
    
    G = nx.DiGraph()
    # 读取路段几何
    road_geometry_pd = pd.read_csv(edge_geometry_path)
    for index, row in road_geometry_pd.iterrows():
        # 获取路段的两个端点
        geometry = row['geometry'].split('(')[1].split(')')[0].split(', ')
        start_node = geometry[0]
        end_node = geometry[-1]
        # 添加边
        G.add_edge(start_node, end_node, key=row['fid'], length=road_dict[row['fid']])
    return G

def build_road_graph(adj_matrix_path, transition_prob_mat_path):
    """
    构建路段图

    参数：
        adj_matrix_path(str): 邻接矩阵的.npy文件路径
        transition_prob_mat_path(str): 转移概率矩阵的.npy文件路径
    返回：
        G(networkx.Graph): 路段图，G=(V, E)，V为路段集合，E为路段转移集合
    """
    adj_matrix = np.load(adj_matrix_path)
    transition_prob_mat = np.load(transition_prob_mat_path)
    G = nx.DiGraph()
    for i in range(adj_matrix.shape[1]):
        u = adj_matrix[0, i]
        v = adj_matrix[1, i]
        prob = transition_prob_mat[u, v]
        G.add_edge(u, v, weight=prob)        
    return G

if __name__ == "__main__":
    # tasks:
    # 1. 构建道路交叉口图
    roadcross_G = build_roadcross_graph(edge_features_path, edge_geometry_path)
    pkl.dump(roadcross_G, open(f'dataset/didi_{city_name}/roadcross_G.pkl', 'wb'))
    # 2. 构建路段图
    road_G = build_road_graph(npy_file_path, transition_prob_mat_path)
    pkl.dump(road_G, open(f'dataset/didi_{city_name}/road_G.pkl', 'wb'))