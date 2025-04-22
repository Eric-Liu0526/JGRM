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

def load_road_vals(file_path):
    """_summary_: 加载路段的轨迹数量

    Args:
        file_path (str): 文件路径

    Returns:
        road_val_dict (dict): 路段id和轨迹数量的映射
    """
    road_val_dict = dict()
    df = pd.read_csv(file_path)
    for idx, row in df.iterrows():
        road_val_dict[row['fid']] = row['traj_count']
    return road_val_dict

def build_road_graph(road_geo_file_path, road_val_file_path):

    road_val_dict = load_road_vals(road_val_file_path)
    df = pd.read_csv(road_geo_file_path)
    # 创建一个有向图（也可以用无向图 nx.Graph()）
    G = nx.DiGraph()

    index2id = df['fid'].to_list()
    starts = list()
    ends = list()

    # 遍历每行数据，提取路段的起始点和终点
    for idx, row in df.iterrows():
        line: LineString = wkt.loads(row['geometry'])
        coords = list(line.coords)
        start = coords[0]
        end = coords[-1]
        starts.append(start)
        ends.append(end)

    # 添加边
    for index, id in enumerate(index2id):
        for s_index, start in enumerate(starts):
            # 如果起始点和当前边的终点相同
            if start == ends[index]:
                G.add_edge(id, index2id[s_index])

    # 为图中节点添加属性
    for node in G.nodes():
        G.nodes[node]['val'] = road_val_dict[node]

    # 输出一些基本信息
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")

    return G


def save_subgraphs_node_ids(G, labels, dataset="chengdu"):

    # 1. 将 labels 转换为 node -> label 映射
    node_labels = dict(zip(G.nodes(), labels))

    # 2. 找出每个 cluster 的节点
    clusters = {}
    for node, label in node_labels.items():
        clusters.setdefault(label, []).append(node)

    sub_g_dict = dict()
    # 3. 保存每个 cluster 的节点 ID
    for label, nodes in clusters.items():
        sub_g_dict[label] = nodes
    with open(f'dataset/didi_{dataset}/road_subgraph_node_ids.pkl', 'wb') as f:
        pkl.dump(sub_g_dict, f)

import pymetis

def metis_partition(G, k=3):
    # 将图转换为 METIS 所需格式（邻接表）
    node_list = list(G.nodes())
    node_index = {node: i for i, node in enumerate(node_list)}
    adj_list = [[] for _ in range(len(node_list))]
    
    for u, v in G.edges():
        i = node_index[u]
        j = node_index[v]
        adj_list[i].append(j)
        adj_list[j].append(i)  # 无向图

    # 执行划分
    n_cuts, membership = pymetis.part_graph(k, adjacency=adj_list)

    # 返回 node -> label 映射
    labels = [membership[node_index[node]] for node in G.nodes()]
    return labels

def get_boundary_nodes_with_vals(G, labels):
    """
    找出所有边界节点，以及它们的 'val' 值（轨迹数量）。
    边界节点定义：其邻居节点中有属于不同聚类标签的。
    """
    node_labels = dict(zip(G.nodes(), labels))
    boundary_nodes = []

    for node in G.nodes():
        node_label = node_labels[node]
        for neighbor in G.neighbors(node):
            if node_labels[neighbor] != node_label:
                boundary_nodes.append(node)
                break  # 有一个邻居是其他聚类就足够了

    # 返回边界节点及其轨迹值
    boundary_vals = [(node, G.nodes[node]['val']) for node in boundary_nodes]
    return boundary_vals

def main1():
    dataset_name = 'chengdu'
    road_geo_file_path = f'dataset/didi_{dataset_name}/edge_geometry.csv'
    road_val_file_path = f'logs/didi_chengdu_road_count.csv'
    G = build_road_graph(road_geo_file_path, road_val_file_path)
    G_undirected = G.to_undirected()
    # labels = spectral_clustering_split_weighted_balanced(G_undirected, n_clusters=9)
    labels = metis_partition(G_undirected, k=10)
    boundary_vals = get_boundary_nodes_with_vals(G_undirected, labels)
    boundary_vals_sorted = sorted(boundary_vals, key=lambda x: x[1])
    with open(f'boundary.txt', 'w') as f:
        for node, val in boundary_vals_sorted:
            f.write(f"边界节点 {node} 的轨迹数量：{val}\n")
        vals = [val for _, val in boundary_vals]
        print("边界节点轨迹数量的统计：")
        print(f"最小值：{np.min(vals)}")
        print(f"最大值：{np.max(vals)}")
        print(f"平均值：{np.mean(vals):.2f}")
        print(f"中位数：{np.median(vals)}")

    # with open(f'boundary.txt', 'w') as f:
    #     for node, val in boundary_vals:
    #         f.write(f'{node},{val}\n')
    #         vals = [val for _, val in boundary_vals]

    save_subgraphs_node_ids(G_undirected, labels, dataset=dataset_name)

def main2():
    dataset_name = 'chengdu'
    with open(f'dataset/didi_{dataset_name}/road_subgraph_node_ids.pkl', 'rb') as f:
        sub_g_dict = pkl.load(f)
    # with open(f'dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb') as file:
    with open(f'dataset/didi_chengdu/chengdu_1101_1115_data_seq_evaluation.pkl', 'rb') as file:
        traj_df = pkl.load(file)
    sub_g_traj_dict = {key: set() for key in sub_g_dict}
    # 遍历轨迹数据，统计每个轨迹在每个子图中的段数
    traj_df.reset_index(drop=True, inplace=True)
    for index, traj in tqdm(traj_df.iterrows(), total=len(traj_df), desc="Processing rows"):
        traj_id = traj['tid']
        opath_list = traj['opath_list']
        occur_dict = {key: 0 for key in sub_g_dict}
        # 统计轨迹在各个子图中的段数
        for road in opath_list:
            for sub_g_id, road_ids in sub_g_dict.items():
                if road in road_ids:
                    occur_dict[sub_g_id] += 1
        # 获取轨迹在哪个子图中段数最多
        max_occur_sub_g_id = max(occur_dict, key=occur_dict.get)
        sub_g_traj_dict[max_occur_sub_g_id].add(index)
    # 将结果保存到文件
    with open(f'dataset/didi_{dataset_name}/sub_g_traj_dict.pkl', 'wb') as f:
        pkl.dump(sub_g_traj_dict, f)

# '''
from concurrent.futures import ProcessPoolExecutor, as_completed
def process_single_traj(index_traj, sub_g_dict):
    index, traj = index_traj
    traj_id = traj['tid']
    opath_list = traj['opath_list']
    occur_dict = {key: 0 for key in sub_g_dict}

    for road in opath_list:
        for sub_g_id, road_ids in sub_g_dict.items():
            if road in road_ids:
                occur_dict[sub_g_id] += 1

    max_occur_sub_g_id = max(occur_dict, key=occur_dict.get)
    return max_occur_sub_g_id, index

def main2p():
    dataset_name = 'chengdu'
    
    with open(f'dataset/didi_{dataset_name}/road_subgraph_node_ids.pkl', 'rb') as f:
        sub_g_dict = pkl.load(f)

    with open(f'dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb') as file:
    # with open(f'dataset/didi_chengdu/chengdu_1101_1115_data_seq_evaluation.pkl', 'rb') as file:
        traj_df = pkl.load(file)
    traj_df.reset_index(drop=True, inplace=True)
    sub_g_traj_dict = {key: set() for key in sub_g_dict}

    num_workers = 20  # 设置线程数为4
    # 并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_traj, item, sub_g_dict) for item in traj_df.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing with multiprocessing"):
            sub_g_id, index = future.result()
            sub_g_traj_dict[sub_g_id].add(index)

    # 保存结果
    with open(f'dataset/didi_{dataset_name}/sub_g_traj_dict.pkl', 'wb') as f:
        pkl.dump(sub_g_traj_dict, f)
# '''

def build_traj_graph_sig(data_name, data_type, edge_weight_threshold=0.8):
    # 读取轨迹数据、划分后轨迹子图节点id、子图路段节点id
    with open(f'dataset/didi_{data_name}/{data_name}_1101_1115_data_sample10w.pkl', 'rb') as file:
    # with open(f'dataset/didi_{data_name}/{data_name}_1101_1115_data_seq_evaluation.pkl', 'rb') as file:
        traj_df = pkl.load(file)
    traj_df.reset_index(drop=True, inplace=True)
    with open(f'dataset/didi_{data_name}/sub_g_traj_dict.pkl', 'rb') as f:
        sub_g_traj_dict = pkl.load(f)
    with open(f'dataset/didi_{data_name}/road_subgraph_node_ids.pkl', 'rb') as f:
        sub_g_dict = pkl.load(f)

    # 读取traj_df中指定index的轨迹数据
    for subg_id, traj_ids in sub_g_traj_dict.items():
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


        '''
        # 遍历子图中的轨迹，统计road_with_trajs_dict: {road: set(traj_id)}
        for index, row in tqdm(traj_sub_df.iterrows(), total=traj_sub_df.shape[0], desc=f'Reading {data_type} Data'):
            cpath_list = row['cpath_list']
            for cpath in cpath_list:
                if cpath not in road_set:
                    road_set.add(cpath)
                    road_with_trajs_dict[cpath] = set()
                road_with_trajs_dict[cpath].add(index)

        # 构建轨迹图，边的权重为轨迹间重合路段数
        for index, row in tqdm(traj_sub_df.iterrows(), total=traj_sub_df.shape[0], desc='Build Trajectory Graph'):
            cpath_list = row['cpath_list']
            # 获取轨迹的路径列表
            for path_id in cpath_list:
                # 遍历路径中的每个轨迹
                for traj_index in road_with_trajs_dict[path_id]:
                    if index != traj_index:
                        if sub_G.has_edge(index, traj_index):
                            sub_G[index][traj_index]['weight'] += 1
                        else:
                            sub_G.add_edge(index, traj_index, weight=1)

        # 归一化权重：weight = 重合路段数 / 两个轨迹路段的并集
        edges_to_remove = []
        for u, v, d in sub_G.edges(data=True):
            d['weight'] = d['weight'] / len(set(traj_sub_df.loc[u, 'cpath_list']).union(set(traj_sub_df.loc[v, 'cpath_list'])))
            # 如果权重<=edge_weight_threshold，则删除边
            if d['weight'] <= edge_weight_threshold:
                edges_to_remove.append((u, v))
        sub_G.remove_edges_from(edges_to_remove)
        '''        
        with open(f'dataset/didi_{data_name}/traj_subg_{subg_id}.pkl', 'wb') as f:
            pkl.dump(sub_G, f)


        '''
        # 转为 PyG 格式
        edge_index = []
        edge_weight = []
        for u, v, d in sub_G.edges(data=True):
            edge_index.append([u, v])
            edge_weight.append(d['weight'])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # 构建 PyG 的 Data 对象（如果有节点特征可以加 x）
        data_pyg = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=traj_num)

        # 保存为 .pt 文件
        graph_path = f'dataset/didi_{data_name}/traj_subg_{key}.pt'
        torch.save(data_pyg, graph_path)
        '''

def main_build_graphs():
    data_name = 'chengdu'
    # build_traj_graph_parallel(data_name, data_type="", edge_weight_threshold=0.8, max_workers=10)
    build_traj_graph_sig(data_name, data_type="", edge_weight_threshold=0.8)

if __name__ == '__main__':
    # main1()
    # main2()
    main2p()
    main_build_graphs()