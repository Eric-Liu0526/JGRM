# anchor_selector.py

import numpy as np
import networkx as nx
from sklearn.cluster import OPTICS
from collections import defaultdict
import math
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from datetime import datetime
class Anchor:
    """通用锚点类：支持路段锚与轨迹片段锚"""
    def __init__(self, anchor_id, anchor_type, metadata=None):
        self.anchor_id = anchor_id
        self.anchor_type = anchor_type  # 'spatial' | 'temporal' | 'dynamic'
        # self.metadata = metadata or {}

    def __repr__(self):
        return f"<Anchor {self.anchor_id} | type={self.anchor_type}>"


class AnchorGenerator:
    """锚点选取主类"""

    def __init__(self, roadcross_graph: nx.Graph, road_graph: nx.Graph, traj_df: pd.DataFrame):
        """
        :param roadcross_graph: 道路交叉口图，G=(V, E)，V为路段端点集合，E为边集合，E={(u, v, key, length)}
        :param road_graph: 路段图，G=(V, E)，V为路段集合，E为边集合，E={(u, v, key, length)}
        :param traj_df: 轨迹数据集，pd.DataFrame
        """
        self.rcG = roadcross_graph
        self.rG = road_graph
        self.traj_df = traj_df
        self.spatial_anchors = []
        self.temporal_anchors = []
        self.dynamic_anchors = []
        self.anchor_trajs = dict()

    def select_spatial_anchors(self, top_k_percent=5):
        """选取空间锚点：根据边介数中心性与路段节点度数"""
        # 计算边介数中心性
        centrality = nx.edge_betweenness_centrality(self.rcG)

        # 将边介数中心性转换为边字典{key, betweenness}
        edge_betweenness_dict = {}
        for edge, betweenness in centrality.items():
            u, v = edge
            fid = self.rcG[u][v]['key']
            edge_betweenness_dict[fid] = betweenness
        centrality_values = list(centrality.values())
        centrality_threshold = np.percentile(centrality_values, 100 - top_k_percent)

        # 计算路段节点度数
        degree_dict = dict(self.rG.degree())

        # 遍历边介数中心性字典，筛选空间锚点
        for index, (fid, cent) in tqdm(enumerate(edge_betweenness_dict.items()), total=len(edge_betweenness_dict), desc='select spatial anchors'):
            if cent >= centrality_threshold and degree_dict[fid] >= 3:
                self.spatial_anchors.append(fid)

    def select_temporal_anchors(self, min_samples=5, xi=0.05, delta_ratio=0.2):
        """时序锚点提取：按小时聚类"""
        # 按小时划分轨迹集合
        hourly_trajectories = defaultdict(list)
        for index, row in self.traj_df.iterrows():
            tid = row['tid']
            start_unix_time = row['start_time']
            end_unix_time = row['road_timestamp'][-1]
            # 获取start_unix_time和end_unix_time转换成一般时间格式
            start_time = datetime.fromtimestamp(start_unix_time)
            end_time = datetime.fromtimestamp(end_unix_time)
            start_hour = start_time.hour
            end_hour = end_time.hour
            # 按小时划分轨迹集合
            # 时间在同一时段
            if start_hour == end_hour:
                hourly_trajectories[start_hour].append(tid)
            # 时间跨两个时段,todo: 需要优化处理
            elif start_hour == end_hour - 1:
                hourly_trajectories[start_hour].append(tid)
                hourly_trajectories[end_hour].append(tid)
        
        # 计算每个时段的阈值
        avg_count = np.mean([len(v) for v in hourly_trajectories.values()])
        delta_thresh = avg_count * delta_ratio

        # OPTICS聚类
        for hour, segments in hourly_trajectories.items():
            if len(segments) < delta_thresh:
                continue
            # 特征工程
            features = []
            valid_tids = []  # 记录有效tid（过滤异常值用）
            for tid in segments:
                row = self.traj_df[self.traj_df['tid'] == tid].iloc[0]
                try:
                    # 添加异常值过滤
                    speeds = row['speed'][1:-1]  # 去除首尾
                    if len(speeds) < 3: continue
                    
                    avg_speed = np.mean(speeds)
                    angle_var = np.var(row['angle_delta'][2:-1])
                    acc_std = np.std(row['acceleration'][2:-1])
                    
                    if not (np.isnan(avg_speed) or np.isinf(angle_var)):
                        features.append([avg_speed, angle_var, acc_std])
                        valid_tids.append(tid)
                except Exception as e:
                    print(f"Error processing tid {tid}: {str(e)}")
                    continue
            
            if len(features) < min_samples:
                continue

            # OPTICS聚类 + 核心点提取
            clustering = OPTICS(min_samples=min_samples, xi=xi, metric='euclidean')
            clustering.fit(features)
            
            # ==== 关键修改：手动计算核心点 ====
            # 1. 获取可达距离和排序
            reachability = clustering.reachability_[clustering.ordering_]
            ordering = clustering.ordering_
            
            # 2. 核心点条件：可达距离小于INF且至少有min_samples邻居
            core_mask = (reachability != np.inf)
            core_indices = ordering[core_mask]
            
            # 3. 获取原始数据索引
            core_tids = [valid_tids[i] for i in core_indices]
            
            # 4. 按簇统计核心点
            for label in set(clustering.labels_[core_mask]):
                if label == -1: continue  # 忽略噪声
                
                # 获取该簇所有核心点
                cluster_mask = (clustering.labels_[core_mask] == label)
                cluster_core_indices = core_indices[cluster_mask]
                
                # 计算簇特征中位数作为代表
                cluster_features = np.array(features)[cluster_core_indices]
                median_feature = np.median(cluster_features, axis=0)
                
                # 记录锚点信息
                self.temporal_anchors.append({
                    'hour': hour,
                    'representative_tids': [valid_tids[i] for i in cluster_core_indices],
                    'avg_speed': median_feature[0],
                    'angle_variance': median_feature[1],
                    'acc_std': median_feature[2],
                    'cluster_size': len(cluster_core_indices)
                })
    

    def select_dynamic_anchors(self, entropy_threshold=2.0):
        """基于转移熵筛选轨迹片段"""
        def compute_transition_entropy(segments):
            """简化熵计算方法：用邻接跳转概率模拟"""
            transitions = defaultdict(int)
            total = 0
            for i in range(len(segments) - 1):
                pair = (segments[i]['road_id'], segments[i+1]['road_id'])
                transitions[pair] += 1
                total += 1
            probs = [v / total for v in transitions.values()]
            return -sum(p * math.log(p + 1e-8) for p in probs)

        anchor_id = len(self.anchors)

        for traj in self.trajectories:
            segs = traj['road_segments']
            if len(segs) < 3:
                continue
            # 按固定窗口划分片段
            for i in range(0, len(segs) - 2):
                seg_window = segs[i:i + 3]
                entropy = compute_transition_entropy(seg_window)
                if entropy > entropy_threshold:
                    center = seg_window[1]['coords']
                    anchor = Anchor(anchor_id, center, 'dynamic', {
                        'entropy': entropy,
                        'traj_id': traj['traj_id']
                    })
                    self.anchors.append(anchor)
                    anchor_id += 1

    def _find_reachable_anchors(self, start_anchor, max_length):
        """使用BFS找到从起始锚点可达的其他锚点"""
        reachable = set()
        queue = [(start_anchor, 0)]  # (node, distance)
        visited = {start_anchor}
        
        while queue:
            current, dist = queue.pop(0)
            if dist >= max_length:
                continue
                
            for neighbor in self.rG[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor in self.spatial_anchors and neighbor != start_anchor:
                        reachable.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return reachable

    def _enumerate_paths(self, start, end, max_length, current_path=None, visited=None):
        """使用DFS枚举从start到end的所有可能路径"""
        if current_path is None:
            current_path = []
        if visited is None:
            visited = set()
            
        current_path.append(start)
        visited.add(start)
        
        if start == end:
            yield current_path[:]
        elif len(current_path) < max_length:
            for neighbor in self.rG[start]:
                if neighbor not in visited:
                    yield from self._enumerate_paths(neighbor, end, max_length, 
                                                   current_path, visited)
        
        current_path.pop()
        visited.remove(start)

    def _calculate_path_score(self, path, alpha=0.7):
        """计算路径得分"""
        # 计算路径中的锚点数量
        anchors_in_path = sum(1 for node in path if node in self.spatial_anchors)
        
        # 计算路径的转移概率和
        trans_prob_sum = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.rG.has_edge(u, v):
                trans_prob_sum += self.rG[u][v].get('weight', 0)
        
        # 计算最终得分
        return alpha * anchors_in_path + (1 - alpha) * trans_prob_sum

    def _is_path_redundant(self, path1, path2, threshold=0.8):
        """检查两条路径是否高度重合"""
        set1 = set(path1)
        set2 = set(path2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        similarity = len(intersection) / len(union)
        return similarity > threshold

    def generate_anchors(self, max_length=10, alpha=0.7):
        """生成锚轨迹
        给定一个空间锚点集合和路网拓扑图self.rG，从空间锚点出发，找到max_length内可达的轨迹片段

        Args:
            max_length: 最大路径长度
            alpha: 路径评分中的锚点数量权重
        """
        all_paths = []
        
        # 1. 对每个锚点，找到可达的其他锚点
        for start_anchor in tqdm(self.spatial_anchors, desc="Generating anchor trajectories"):
            reachable_anchors = self._find_reachable_anchors(start_anchor, max_length)
            
            # 2. 对每对可达锚点，枚举路径
            for end_anchor in reachable_anchors:
                paths = list(self._enumerate_paths(start_anchor, end_anchor, max_length))
                
                # 3. 计算路径得分并选择最佳路径
                if paths:
                    scored_paths = [(path, self._calculate_path_score(path, alpha)) 
                                  for path in paths]
                    best_path = max(scored_paths, key=lambda x: x[1])[0]
                    all_paths.append(best_path)
        
        # 4. 过滤冗余路径
        filtered_paths = []
        for path in all_paths:
            is_redundant = False
            for existing_path in filtered_paths:
                if self._is_path_redundant(path, existing_path):
                    is_redundant = True
                    break
            if not is_redundant:
                filtered_paths.append(path)
        
        # 5. 保存锚轨迹, 以index为key, 以轨迹片段列表为value
        for index,path in enumerate(filtered_paths):
            self.anchor_trajs[index] = path
        # 6. 保存锚轨迹
        pkl.dump(self.anchor_trajs, open(f'dataset/didi_{city_name}/anchor_trajs.pkl', 'wb'))

city_name = 'chengdu'
roadcross_G = pkl.load(open(f'dataset/didi_{city_name}/roadcross_G.pkl', 'rb'))
road_G = pkl.load(open(f'dataset/didi_{city_name}/road_G.pkl', 'rb'))
traj_df = pkl.load(open(f'dataset/didi_{city_name}/chengdu_1101_1115_data_sample10w.pkl', 'rb'))
anchor_generator = AnchorGenerator(roadcross_G, road_G, traj_df)
anchor_generator.select_spatial_anchors(top_k_percent=5)
anchor_generator.generate_anchors(max_length=10, alpha=0.7)
# anchor_generator.select_temporal_anchors(min_samples=5, xi=0.05, delta_ratio=0.2)