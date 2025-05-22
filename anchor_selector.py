import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import pickle as pkl
import time
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

class AnchorSelector:
    def __init__(self, bandwidth: float = 0.5, kernel: str = 'gaussian'):
        """
        初始化锚轨迹选择器
        
        参数:
            bandwidth: 核密度估计的带宽参数
            kernel: 核函数类型，可选 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde = None
        self.features = None
        self.scaler = StandardScaler()
        
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
        
    def fit(self, df: pd.DataFrame) -> 'AnchorSelector':
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
    
    def select_anchors(self, n_anchors: int, save_path: str = None, df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        选择锚轨迹
        
        参数:
            n_anchors: 需要选择的锚轨迹数量
            
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
        # 将anchor_indices转为tid
        anchor_tids = df['tid'].iloc[anchor_indices]
        if save_path is not None:
            pkl.dump((anchor_tids, anchor_densities), open(save_path, 'wb'))
        return anchor_indices, anchor_densities
    
    def get_density_scores(self) -> np.ndarray:
        """
        获取所有轨迹的密度得分
        
        返回:
            np.ndarray: 所有轨迹的密度得分
        """
        if self.kde is None:
            raise ValueError("请先调用fit方法拟合模型")
            
        return np.exp(self.kde.score_samples(self.features))
    
    def cal_similarity(self, anchor_features, all_features):
        """
        计算锚点与所有轨迹的相似度
        
        参数:
            anchor_features: 锚点特征，形状为 (n_anchors, n_features)
            all_features: 所有轨迹特征，形状为 (n_trajectories, n_features)
            
        返回:
            np.ndarray: 所有轨迹的相似度得分，形状为 (n_trajectories, n_anchors)
        """
        # 将所有特征合并后一起归一化
        combined_features = np.vstack([anchor_features, all_features])
        combined_features_norm = combined_features / np.linalg.norm(combined_features, axis=1, keepdims=True)
        
        # 分离回锚点特征和所有特征
        n_anchors = anchor_features.shape[0]
        anchor_features_norm = combined_features_norm[:n_anchors]
        all_features_norm = combined_features_norm[n_anchors:]
        
        # 计算余弦相似度
        similarity = np.dot(all_features_norm, anchor_features_norm.T)
        # 获得每个锚点与所有轨迹的相似度
        similarity = similarity.T

        # 转化为相似度排名,按照相似度从小到大排序
        similarity_rank = np.argsort(similarity, axis=1)

        # 归一化相似度得分
        # similarity = similarity / np.sum(similarity, axis=1, keepdims=True)

        return similarity, similarity_rank

    def evaluate_anchors(self, n_anchors: int, plot: bool = True) -> dict:
        """
        评估锚点选择的质量
        
        参数:
            n_anchors: 锚点数量
            plot: 是否绘制评估图表
            
        返回:
            dict: 包含各项评估指标的字典
        """
        if self.kde is None:
            raise ValueError("请先调用fit方法拟合模型")
            
        # 选择锚点
        anchor_indices, anchor_densities = self.select_anchors(n_anchors)
        
        # 1. 密度评估
        all_densities = self.get_density_scores()
        density_ratio = np.mean(anchor_densities) / np.mean(all_densities)
        
        # 2. 空间分布评估
        anchor_features = self.features[anchor_indices]
        distances = euclidean_distances(anchor_features)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        # 3. 覆盖率评估
        # 计算每个原始轨迹到最近锚点的距离
        all_distances = euclidean_distances(self.features, anchor_features)
        min_distances_to_anchors = np.min(all_distances, axis=1)
        coverage_threshold = np.percentile(min_distances_to_anchors, 90)  # 90%的轨迹都在这个距离内
        coverage_ratio = np.mean(min_distances_to_anchors <= coverage_threshold)
        
        # 4. 密度分布评估
        density_percentile = np.percentile(all_densities, 100 * (1 - n_anchors/len(all_densities)))
        density_quality = np.mean(anchor_densities > density_percentile)
        
        metrics = {
            'density_ratio': density_ratio,  # 锚点平均密度与整体平均密度的比值
            'avg_min_distance': avg_min_distance,  # 锚点之间的平均最小距离
            'coverage_ratio': coverage_ratio,  # 覆盖率
            'density_quality': density_quality,  # 密度质量
            'anchor_densities': anchor_densities,  # 锚点密度值
            'all_densities': all_densities  # 所有轨迹的密度值
        }
        
        if plot:
            self._plot_evaluation_metrics(metrics)
            
        return metrics
    
    def _plot_evaluation_metrics(self, metrics: dict):
        """
        绘制评估指标的图表
        """
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

def select_anchors(bandwidth=0.5, kernel='gaussian', n_anchors=1000):
    df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))
    save_path = 'dataset/didi_chengdu/anchor_indices_and_densities.pkl'
    anchor_selector = AnchorSelector(bandwidth=bandwidth, kernel=kernel)
    anchor_selector.fit(df)
    anchor_indices, anchor_densities = anchor_selector.select_anchors(n_anchors, save_path, df)
    return anchor_indices, anchor_densities

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
    anchor_selector = AnchorSelector()
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
    # select_anchors(0.5, 'gaussian', 1024)
    cal_similarity_matrix()
    # df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))
    # start_time = time.time()
    # anchor_selector = AnchorSelector(bandwidth=0.5, kernel='gaussian')
    # anchor_selector.fit(df)
    # # anchor_indices, anchor_densities = anchor_selector.select_anchors(1000, 'dataset/didi_chengdu/anchor_indices_and_densities.pkl')
    
    # # 评估锚点质量
    # metrics = anchor_selector.evaluate_anchors(1000)
    # print("\n评估指标:")
    # print(f"密度比值: {metrics['density_ratio']:.2f}")
    # print(f"平均最小距离: {metrics['avg_min_distance']:.2f}")
    # print(f"覆盖率: {metrics['coverage_ratio']:.2f}")
    # print(f"密度质量: {metrics['density_quality']:.2f}")
    
    # end_time = time.time()
    # print(f'\n总运行时间: {end_time - start_time} 秒')
