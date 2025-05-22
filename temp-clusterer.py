import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from typing import List, Dict, Any
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class TrajectoryClusterer:
    def __init__(self, min_samples: int = 5, xi: float = 0.05, min_cluster_size: float = 0.001):
        """
        初始化轨迹聚类器
        
        参数:
            min_samples: OPTICS算法的最小样本数
            xi: OPTICS算法的xi参数，用于确定聚类边界
            min_cluster_size: 最小聚类大小（相对于总样本数的比例）
        """
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.scaler = StandardScaler()
        self.clusterer = None
        print(f"初始化聚类器，min_samples: {min_samples}, xi: {xi}, min_cluster_size: {min_cluster_size}")
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        从轨迹数据中提取特征
        
        参数:
            df: 包含轨迹数据的DataFrame
            
        返回:
            特征矩阵
        """
        # 空间特征: 总长度、经纬度标准差、经纬度标准差乘积
        total_lengths = df['total_length'].values
        lat_lists = df['lat_list'].values
        lng_lists = df['lng_list'].values
        lat_stds = [np.std(lat_list) for lat_list in lat_lists]
        lng_stds = [np.std(lng_list) for lng_list in lng_lists]
        lat_lng_std_products = [lat_std * lng_std for lat_std, lng_std in zip(lat_stds, lng_stds)] # 计算经纬度标准差乘积
        
        # 时间特征: 开始时间、总时间
        start_times = df['start_time'].values
        total_times = df['total_time'].values

        # 动态特征: 速度中位数、加速度标准差、方向变化总量
        speed_lists = [df['speed'].values[i][1:-1] for i in range(len(df))]
        speed_medians = [np.median(speed_list) for speed_list in speed_lists]    # 计算速度中位数
        acceleration_lists = [df['acceleration'].values[i][2:-1] for i in range(len(df))]
        acceleration_vars = [np.var(acceleration_list) for acceleration_list in acceleration_lists]    # 计算加速度标准差
        angle_delta_lists = [df['angle_delta'].values[i][2:-1] for i in range(len(df))]
        angle_delta_sums = [np.sum(angle_delta_list) for angle_delta_list in angle_delta_lists]  # 计算方向变化总量
        
        # 语义特征: 道路多样性
        opath_lists = df['opath_list'].values
        road_diversities = [len(set(opath_list)) / len(opath_list) for opath_list in opath_lists]  # 计算道路多样性

        
        # 提取特征
        features = np.array([lat_lng_std_products, speed_medians, acceleration_vars, angle_delta_sums])
        features = features.T

        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        对轨迹数据进行聚类
        
        参数:
            df: 包含轨迹数据的DataFrame
        """
        # 提取特征
        start_time = time.time()
        features = self.extract_features(df)
        end_time = time.time()
        print(f"提取特征过程的时间: {end_time - start_time} 秒")
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        plt.figure(figsize=(10, 6))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], s=2)
        plt.title("PCA of Trajectory Features")
        plt.savefig("pca2.png")
        print(f'图片已保存')

        # 初始化OPTICS聚类器
        self.clusterer = OPTICS(
            min_samples=self.min_samples,
            xi=self.xi,
            min_cluster_size=self.min_cluster_size
        )
        
        # 执行聚类
        start_time = time.time()
        self.clusterer.fit(features)
        end_time = time.time()
        print(f"聚类过程的时间: {end_time - start_time} 秒")
    
    def get_clusters(self) -> np.ndarray:
        """
        获取聚类结果
        
        返回:
            聚类标签数组，-1表示噪声点
        """
        if self.clusterer is None:
            raise ValueError("请先调用fit方法进行聚类")
        return self.clusterer.labels_
    
    def get_cluster_stats(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        获取每个聚类的统计信息
        
        参数:
            df: 原始轨迹数据DataFrame
            
        返回:
            包含每个聚类统计信息的字典
        """
        if self.clusterer is None:
            raise ValueError("请先调用fit方法进行聚类")
            
        labels = self.clusterer.labels_
        cluster_stats = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # 跳过噪声点
                continue
                
            # 获取属于该聚类的轨迹
            cluster_trajectories = df[labels == cluster_id]
            
            # 计算统计信息
            stats = {
                'size': len(cluster_trajectories),
                'avg_speed': cluster_trajectories['speed'].mean(),
                'avg_length': cluster_trajectories['total_length'].mean(),
                'avg_time': cluster_trajectories['total_time'].mean(),
                'trajectory_ids': cluster_trajectories['tid'].tolist()
            }
            
            cluster_stats[cluster_id] = stats
            
        return cluster_stats

# 测试
if __name__ == "__main__":
    # 读取数据
    df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))
    # 初始化聚类器
    clusterer = TrajectoryClusterer()
    # 拟合数据
    clusterer.fit(df)
    # 获取聚类结果
    labels = clusterer.get_clusters()
    # 获取labels中聚类数量
    cluster_num = len(set(labels))
    # 打印聚类结果
    # print(labels)
    print(cluster_num)
    