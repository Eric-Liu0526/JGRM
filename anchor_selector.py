import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Set
import pickle as pkl
import time
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tqdm import tqdm
import networkx as nx

class BaseAnchorSelector(ABC):
    """锚点选择器的基类"""
    
    def __init__(self):
        self.features = None
        self.scaler = StandardScaler()
    
    @abstractmethod
    def select_anchors(self, df: pd.DataFrame, n_anchors: int, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择锚点轨迹的抽象方法
        
        参数:
            df: 包含轨迹数据的DataFrame
            n_anchors: 需要选择的锚轨迹数量
            save_path: 保存结果的路径
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 返回锚轨迹的索引和对应的得分
        """
        pass
    
    def extract_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """
        从轨迹数据中提取特征
        
        参数:
            df: 包含轨迹数据的DataFrame
            fit_scaler: 是否重新拟合scaler，默认为True
            
        返回:
            特征矩阵
        """
        # 空间特征: 总长度、经纬度标准差、经纬度标准差乘积
        total_lengths = df['total_length'].values
        lat_lists = df['lat_list'].values
        lng_lists = df['lng_list'].values
        lat_stds = [np.std(lat_list) for lat_list in lat_lists]
        lng_stds = [np.std(lng_list) for lng_list in lng_lists]
        lat_lng_std_products = [lat_std * lng_std for lat_std, lng_std in zip(lat_stds, lng_stds)]
        
        # 时间特征: 开始时间、总时间
        start_times = df['start_time'].values
        total_times = df['total_time'].values

        # 动态特征: 速度中位数、加速度标准差、方向变化总量
        speed_lists = [df['speed'].values[i][1:-1] for i in range(len(df))]
        speed_medians = [np.median(speed_list) for speed_list in speed_lists]
        acceleration_lists = [df['acceleration'].values[i][2:-1] for i in range(len(df))]
        acceleration_vars = [np.var(acceleration_list) for acceleration_list in acceleration_lists]
        angle_delta_lists = [df['angle_delta'].values[i][2:-1] for i in range(len(df))]
        angle_delta_sums = [np.sum(angle_delta_list) for angle_delta_list in angle_delta_lists]
        
        # 语义特征: 道路多样性
        opath_lists = df['opath_list'].values
        road_diversities = [len(set(opath_list)) / len(opath_list) for opath_list in opath_lists]
        
        # 提取特征
        features = np.array([total_lengths, lat_lng_std_products, speed_medians, acceleration_vars, angle_delta_sums, road_diversities])
        features = features.T

        # 标准化特征
        if fit_scaler:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        return features_scaled

class DensityBasedAnchorSelector(BaseAnchorSelector):
    """基于密度的锚点选择器"""
    
    def __init__(self, bandwidth: float = 0.5, kernel: str = 'gaussian'):
        """
        初始化基于密度的锚点选择器
        
        参数:
            bandwidth: 核密度估计的带宽参数
            kernel: 核函数类型，可选 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = None
    
    def fit(self, df: pd.DataFrame) -> 'DensityBasedAnchorSelector':
        """
        使用轨迹数据拟合核密度估计模型
        
        参数:
            df: 包含轨迹数据的DataFrame
            
        返回:
            self: 返回实例本身，支持链式调用
        """
        self.features = self.extract_features(df)
        start_time = time.time()
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde.fit(self.features)
        end_time = time.time()
        print(f'拟合模型时间: {end_time - start_time} 秒')
        return self
    
    def select_anchors(self, df: pd.DataFrame, n_anchors: int, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于密度选择锚点轨迹
        
        参数:
            df: 包含轨迹数据的DataFrame
            n_anchors: 需要选择的锚轨迹数量
            save_path: 保存结果的路径
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 返回锚轨迹的索引和对应的密度值
        """
        if self.kde is None:
            raise ValueError("请先调用fit方法拟合模型")
            
        # 计算所有轨迹的密度值
        densities = np.exp(self.kde.score_samples(self.features))
        
        # 选择密度最高的n_anchors个轨迹作为锚点
        anchor_indices = np.argsort(densities)[-n_anchors:]
        anchor_densities = densities[anchor_indices]
        
        if save_path is not None:
            anchor_tids = df['tid'].iloc[anchor_indices]
            pkl.dump((anchor_tids, anchor_densities), open(save_path, 'wb'))
            
        return anchor_indices, anchor_densities
    
    def evaluate_anchors(self, df: pd.DataFrame, n_anchors: int, plot: bool = True) -> dict:
        """
        评估锚点选择的质量
        
        参数:
            df: 包含轨迹数据的DataFrame
            n_anchors: 锚点数量
            plot: 是否绘制评估图表
            
        返回:
            dict: 包含各项评估指标的字典
        """
        if self.kde is None:
            raise ValueError("请先调用fit方法拟合模型")
            
        # 选择锚点
        anchor_indices, anchor_densities = self.select_anchors(df, n_anchors)
        
        # 1. 密度评估
        all_densities = np.exp(self.kde.score_samples(self.features))
        density_ratio = np.mean(anchor_densities) / np.mean(all_densities)
        
        # 2. 空间分布评估
        anchor_features = self.features[anchor_indices]
        distances = euclidean_distances(anchor_features)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        # 3. 覆盖率评估
        all_distances = euclidean_distances(self.features, anchor_features)
        min_distances_to_anchors = np.min(all_distances, axis=1)
        coverage_threshold = np.percentile(min_distances_to_anchors, 90)
        coverage_ratio = np.mean(min_distances_to_anchors <= coverage_threshold)
        
        # 4. 密度分布评估
        density_percentile = np.percentile(all_densities, 100 * (1 - n_anchors/len(all_densities)))
        density_quality = np.mean(anchor_densities > density_percentile)
        
        metrics = {
            'density_ratio': density_ratio,
            'avg_min_distance': avg_min_distance,
            'coverage_ratio': coverage_ratio,
            'density_quality': density_quality,
            'anchor_densities': anchor_densities,
            'all_densities': all_densities
        }
        
        if plot:
            self._plot_evaluation_metrics(metrics)
            
        return metrics
    
    def _plot_evaluation_metrics(self, metrics: dict):
        """绘制评估指标的图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 密度分布图
        axes[0, 0].hist(metrics['all_densities'], bins=50, alpha=0.5, label='所有轨迹')
        axes[0, 0].hist(metrics['anchor_densities'], bins=50, alpha=0.5, label='锚点轨迹')
        axes[0, 0].set_title('密度分布对比')
        axes[0, 0].legend()
        
        # 2. 密度箱线图
        axes[0, 1].boxplot([metrics['all_densities'], metrics['anchor_densities']], 
                          labels=['所有轨迹', '锚点轨迹'])
        axes[0, 1].set_title('密度分布箱线图')
        
        # 3. 评估指标条形图
        metrics_to_plot = {
            '密度比值': metrics['density_ratio'],
            '平均最小距离': metrics['avg_min_distance'],
            '覆盖率': metrics['coverage_ratio'],
            '密度质量': metrics['density_quality']
        }
        axes[1, 0].bar(metrics_to_plot.keys(), metrics_to_plot.values())
        axes[1, 0].set_title('评估指标')
        
        # 4. 密度散点图
        axes[1, 1].scatter(range(len(metrics['all_densities'])), 
                          sorted(metrics['all_densities']), 
                          alpha=0.5, label='所有轨迹')
        axes[1, 1].scatter(range(len(metrics['anchor_densities'])), 
                          sorted(metrics['anchor_densities']), 
                          alpha=0.5, label='锚点轨迹')
        axes[1, 1].set_title('密度排序散点图')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('anchor_evaluation.png')

class SpatiotemporalAnchorSelector(BaseAnchorSelector):
    """基于时空分布的锚点选择器"""
    
    def __init__(self, alpha: float = 0.5):
        """
        初始化基于时空分布的锚点选择器
        
        参数:
            alpha: 时间权重因子
        """
        super().__init__()
        self.alpha = alpha
    
    def extract_feature_vector(self, traj: pd.Series) -> np.ndarray:
        """
        提取轨迹的特征向量
        
        参数:
            traj: 单条轨迹数据
            
        返回:
            np.ndarray: 特征向量
        """
        # 行为特征
        speed_list = traj['speed'][1:]
        acc_list = traj['acceleration'][2:]
        angle_list = traj['angle_delta'][2:]
        
        mean_speed = np.mean(speed_list)
        std_speed = np.std(speed_list)
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        mean_angle = np.mean(angle_list)
        std_angle = np.std(angle_list)
        
        # 空间特征（起终点）
        start_lat, start_lng = traj['lat_list'][0], traj['lng_list'][0]
        end_lat, end_lng = traj['lat_list'][-1], traj['lng_list'][-1]
        
        # 时间特征（转为正余弦编码）
        start_time = pd.to_datetime(traj['start_time'])
        hour = start_time.hour
        time_sin = np.sin(hour * (2 * np.pi / 24))
        time_cos = np.cos(hour * (2 * np.pi / 24))
        time_slot = np.array([time_sin, time_cos])
        
        # 尺度特征
        route_len = traj['total_length']
        duration = traj['total_time']
        
        # 拼接所有特征
        feat_vec = np.concatenate([
            np.array([mean_speed, std_speed]),
            np.array([mean_acc, std_acc]),
            np.array([mean_angle, std_angle]),
            np.array([start_lat, start_lng]),
            np.array([end_lat, end_lng]),
            np.array([route_len, duration]),
            time_slot
        ])
        
        return feat_vec
    
    def calculate_similarity_matrix_vectorized(self, df: pd.DataFrame, anchor_indices: np.ndarray, 
                                            save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用向量化方法计算相似度矩阵，结合特征向量相似度和轨迹序列Jaccard相似度
        
        参数:
            df: 包含轨迹数据的DataFrame
            anchor_indices: 锚点轨迹的索引数组
            save_path: 保存结果的路径
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 相似度矩阵和相似度排名矩阵
        """
        # 提取所有轨迹的特征向量
        all_features = np.array([self.extract_feature_vector(df.iloc[i]) for i in range(len(df))])
        anchor_features = all_features[anchor_indices]
        
        # 标准化特征
        all_features = self.scaler.fit_transform(all_features)
        anchor_features = self.scaler.transform(anchor_features)
        
        # 计算余弦相似度矩阵
        cosine_similarity = np.dot(all_features, anchor_features.T)
        
        # 计算轨迹序列的Jaccard相似度
        all_paths = df['cpath_list'].values
        anchor_paths = df.iloc[anchor_indices]['cpath_list'].values
        
        # 将路径列表转换为集合
        all_path_sets = [set(path) for path in all_paths]
        anchor_path_sets = [set(path) for path in anchor_paths]
        
        # 计算Jaccard相似度矩阵
        jaccard_similarity = np.zeros((len(df), len(anchor_indices)))
        for i in range(len(df)):
            for j in range(len(anchor_indices)):
                intersection = len(all_path_sets[i] & anchor_path_sets[j])
                union = len(all_path_sets[i] | anchor_path_sets[j])
                jaccard_similarity[i, j] = intersection / union if union > 0 else 0
        
        # 结合余弦相似度和Jaccard相似度（使用加权平均）
        # alpha = 0.5  # 余弦相似度权重
        # beta = 0.5   # Jaccard相似度权重
        similarity_matrix = cosine_similarity * jaccard_similarity
        
        # 计算相似度排名
        similarity_rank = np.argsort(similarity_matrix, axis=1)
        
        if save_path is not None:
            pkl.dump((similarity_matrix, similarity_rank), open(save_path, 'wb'))
            
        return similarity_matrix, similarity_rank
    
    def select_anchors(self, df: pd.DataFrame, min_coverage: float = 0.95, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于时空分布选择锚点轨迹，直到达到指定的覆盖量
        
        参数:
            df: 包含轨迹数据的DataFrame
            min_coverage: 最小覆盖率要求，范围[0,1]
            save_path: 保存结果的路径
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 返回锚轨迹的索引和对应的增益值
        """
        # 初始化数据结构
        T = set(df['tid'].values)  # 所有轨迹ID集合
        cpath_dict = {tid: set(df[df['tid'] == tid]['opath_list'].iloc[0]) for tid in T}  # 轨迹ID到道路段集合的映射
        time_dict = {tid: pd.to_datetime(df[df['tid'] == tid]['start_time'].iloc[0]).hour for tid in T}  # 轨迹ID到时间段的映射
        
        # 计算总道路段数量
        all_segments = set()
        for seg_set in cpath_dict.values():
            all_segments.update(seg_set)
        total_segments = len(all_segments)
        
        selected_anchors = []
        covered_segments = set()
        covered_time_buckets = set()
        anchor_gains = []
        
        pbar = tqdm(desc='选择锚点轨迹')
        while True:
            best_tid = None
            max_gain = -1
            
            for tid in tqdm(T, desc='遍历轨迹', leave=False):
                if tid in selected_anchors:
                    continue
                    
                seg_set = cpath_dict[tid]
                time_bucket = time_dict[tid]
                
                # 计算增量增益
                new_segments = seg_set - covered_segments
                new_time = 0 if time_bucket in covered_time_buckets else 1
                
                gain = len(new_segments) + self.alpha * new_time
                
                if gain > max_gain:
                    max_gain = gain
                    best_tid = tid
            
            if best_tid is None:
                break  # 没有更多有增益的锚点
                
            selected_anchors.append(best_tid)
            anchor_gains.append(max_gain)
            covered_segments.update(cpath_dict[best_tid])
            covered_time_buckets.add(time_dict[best_tid])
            
            # 计算当前覆盖率
            current_coverage = len(covered_segments) / total_segments
            pbar.update(1)
            pbar.set_description(f'选择锚点轨迹 (覆盖率: {current_coverage:.2%})')
            
            # 检查是否达到目标覆盖率
            if current_coverage >= min_coverage:
                break
        
        pbar.close()
                
        # 将tid转换为索引
        anchor_indices = [df[df['tid'] == tid].index[0] for tid in selected_anchors]
        
        if save_path is not None:
            pkl.dump((selected_anchors, anchor_gains), open(save_path, 'wb'))
            
        return np.array(anchor_indices), np.array(anchor_gains)
    
class RoadAnchorSelector(BaseAnchorSelector):
    """路段锚点选择器"""
    
    def __init__(self):
        super().__init__()
        self.transition_prob_mat = np.load('dataset/didi_chengdu/transition_prob_mat.npy')
        self.edge_features = pd.read_csv('dataset/didi_chengdu/edge_features.csv')
        self.line_graph_edge_idx = np.load('dataset/didi_chengdu/line_graph_edge_idx.npy')
        self.traj_df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))

    def select_anchors(self, df: pd.DataFrame = None, n_anchors: int = None, save_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        实现基类的抽象方法，选择路段锚点
        
        参数:
            df: 不使用，保持接口一致
            n_anchors: 不使用，保持接口一致
            save_path: 保存结果的路径
            
        返回:
            Tuple[np.ndarray, np.ndarray]: 返回锚点索引和对应的得分
        """
        anchor_indices = self.select_road_anchors()
        if save_path is not None:
            pkl.dump(anchor_indices, open(save_path, 'wb'))
        return anchor_indices, np.ones_like(anchor_indices)  # 返回等权重得分

    def select_road_anchors(self):
        """
        构建有向路网拓扑图，基于最小点覆盖贪心选择路段锚点。
        返回：
            anchor_indices: 被选为锚点的路段索引列表
            anchor_scores: 对应的出度权重和
        """
        # 1. 构建有向图
        G = nx.DiGraph()
        num_nodes = self.transition_prob_mat.shape[0]
        for i in range(num_nodes):
            for j in range(num_nodes):
                prob = self.transition_prob_mat[i, j]
                if prob > 0:
                    G.add_edge(i, j, weight=prob)
        
        # 2. 计算每个节点的出度权重和
        out_weight_sum = {node: sum([d['weight'] for _, _, d in G.out_edges(node, data=True)]) for node in G.nodes}
        
        # 3. 贪心选择锚点
        candidate_nodes = set(G.nodes)
        covered_nodes = set()
        anchor_indices = []
        # anchor_scores = []
        
        while candidate_nodes:
            # 只在候选节点中选最大出度权重和
            node_scores = {node: out_weight_sum[node] for node in candidate_nodes}
            best_node = max(node_scores, key=node_scores.get)
            anchor_indices.append(best_node)
            # anchor_scores.append(out_weight_sum[best_node])
            # 其一阶邻居（出边指向的节点）
            neighbors = set([v for _, v in G.out_edges(best_node)])
            # 移除自身和一阶邻居
            remove_nodes = neighbors | {best_node}
            candidate_nodes -= remove_nodes
            covered_nodes |= remove_nodes
        
        return np.array(anchor_indices)
    
    def link_enhancement(self, anchor_indices):
        """
        对锚点及其二阶邻居进行连边，生成新的line_graph_edge_idx
        
        参数:
            anchor_indices: 锚点索引数组
            
        返回:
            new_edge_idx: 新的line_graph_edge_idx
        """
        import networkx as nx
        
        # 1. 构建原始有向图
        G = nx.DiGraph()
        num_nodes = self.transition_prob_mat.shape[0]
        for i in range(num_nodes):
            for j in range(num_nodes):
                prob = self.transition_prob_mat[i, j]
                if prob > 0:
                    G.add_edge(i, j, weight=prob)
        
        # 2. 获取每个锚点的二阶邻居
        second_order_neighbors = {}
        for anchor in anchor_indices:
            # 获取一阶邻居
            first_order = set([v for _, v in G.out_edges(anchor)])
            # 获取二阶邻居
            second_order = set()
            for neighbor in first_order:
                second_order.update([v for _, v in G.out_edges(neighbor)])
            second_order_neighbors[anchor] = second_order
        
        # 3. 构建新的边集合
        new_edges = list()
        starts = list()
        ends = list()
        
        # 3.1 保留原始边
        for i, j in G.edges():
            starts.append(i)
            ends.append(j)
        
        # 3.2 添加锚点之间的边
        # for i in range(len(anchor_indices)):
        #     for j in range(i + 1, len(anchor_indices)):
        #         starts.append(anchor_indices[i])
        
        # 3.3 添加锚点与其二阶邻居之间的边
        for anchor in anchor_indices:
            for neighbor in second_order_neighbors[anchor]:
                starts.append(anchor)
                ends.append(neighbor)
                # new_edges.add((anchor, neighbor))
                # new_edges.add((neighbor, anchor))
        
        starts = np.array(starts)
        ends = np.array(ends)
        new_edges = np.stack((starts, ends), axis=0)

        # 4. 转换为numpy数组格式
        new_edge_idx = np.array(new_edges)
        
        # 5. 保存新的边索引
        np.save('dataset/didi_chengdu/enhanced_line_graph_edge_idx.npy', new_edge_idx)
        
        return new_edge_idx
    
    def enhance_road_anchors(self):
        """
        对路段锚点进行链接增强
        返回：
            anchor_indices: 被选为锚点的路段索引列表
        """
        anchor_indices = self.select_road_anchors()
        new_edge_idx = self.link_enhancement(anchor_indices)

def select_anchors_by_density(bandwidth=0.5, kernel='gaussian', n_anchors=1000):
    """使用基于密度的方法选择锚点"""
    df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))
    save_path = 'dataset/didi_chengdu/anchor_indices_and_densities.pkl'
    anchor_selector = DensityBasedAnchorSelector(bandwidth=bandwidth, kernel=kernel)
    anchor_selector.fit(df)
    anchor_indices, anchor_densities = anchor_selector.select_anchors(df, n_anchors, save_path)
    return anchor_indices, anchor_densities

def select_anchors_by_spatiotemporal(alpha=0.5, min_coverage=0.95, lambda_weight=0.5, tau=2.0):
    """使用基于时空分布的方法选择锚点"""
    df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))
    df = df.reset_index(drop=True)
    save_path = 'dataset/didi_chengdu/spatiotemporal_anchors.pkl'
    similarity_save_path = 'dataset/didi_chengdu/spatiotemporal_similarity.pkl'
    
    # 选择锚点
    anchor_selector = SpatiotemporalAnchorSelector(alpha=alpha)
    anchor_indices, anchor_gains = anchor_selector.select_anchors(df, min_coverage, save_path)
    # 直接读取pkl文件
    # tids, anchor_gains = pkl.load(open(save_path, 'rb'))
    # 将tids转换为索引
    # anchor_indices = [df['tid'].tolist().index(tid) for tid in tids]
    
    # 计算相似度矩阵
    similarity_matrix = anchor_selector.calculate_similarity_matrix_vectorized(
        df, anchor_indices, similarity_save_path
    )
    
    # 转换为数组格式
    similarity_array, similarity_rank = similarity_matrix
    
    # 保存数组格式的相似度矩阵
    pkl.dump((similarity_array, similarity_rank), 
             open('dataset/didi_chengdu/spatiotemporal_similarity_array.pkl', 'wb'))
    
    return anchor_indices, anchor_gains, similarity_array, similarity_rank

def cal_similarity_matrix(anchor_indices_path='dataset/didi_chengdu/anchor_indices_and_densities.pkl'):
    """
    使用已保存的锚点索引计算相似度矩阵
    
    参数:
        anchor_indices_path: 保存的锚点索引和密度文件路径
        
    返回:
        np.ndarray: 相似度矩阵
    """
    df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))

    # 加载已保存的锚点tid和密度
    anchor_tids, anchor_densities = pkl.load(open(anchor_indices_path, 'rb'))
    anchor_tids = anchor_tids.tolist()
    # 获取锚点tid对应索引
    anchor_indices = [df['tid'].tolist().index(tid) for tid in anchor_tids]

    # 获取锚轨迹特征
    anchor_df = df.iloc[anchor_indices].copy()
    all_df = df.copy()
    
    # 提取特征向量
    anchor_selector = BaseAnchorSelector()
    # 先对所有数据进行标准化
    all_features = anchor_selector.extract_features(all_df, fit_scaler=True)
    # 使用相同的标准化参数处理锚点特征
    anchor_features = anchor_selector.extract_features(anchor_df, fit_scaler=False)
    
    # 计算相似度
    similarity_matrix, similarity_rank = anchor_selector.cal_similarity(anchor_features, all_features)

    # 保存相似度矩阵
    pkl.dump(similarity_matrix, open('dataset/didi_chengdu/similarity_matrix.pkl', 'wb'))
    pkl.dump(similarity_rank, open('dataset/didi_chengdu/similarity_rank.pkl', 'wb'))

if __name__ == "__main__":
    # 使用基于密度的方法选择锚点
    # select_anchors_by_density(0.5, 'gaussian', 1024)
    
    # 使用基于时空分布的方法选择锚点
    select_anchors_by_spatiotemporal(alpha=0.5, min_coverage=1, lambda_weight=0.5, tau=2.0)
    
    # 计算相似度矩阵
    # cal_similarity_matrix()

    '''
    # 使用路段锚点选择器
    road_anchor_selector = RoadAnchorSelector()
    road_anchor_selector.enhance_road_anchors()
    '''