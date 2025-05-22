import pickle
import numpy as np
from typing import Dict, List, Tuple

class AnchorSelector:
    def __init__(self, road_hour_score_dict_path: str, road_od_score_dict_path: str, traj_acceleration_dict_path: str):
        self.road_hour_score_dict_path = road_hour_score_dict_path
        self.road_od_score_dict_path = road_od_score_dict_path
        self.traj_acceleration_dict_path = traj_acceleration_dict_path
        
        # 加载评分数据
        self.road_hour_scores = self._load_pickle(road_hour_score_dict_path)
        self.road_od_scores = self._load_pickle(road_od_score_dict_path)
        self.traj_accelerations = self._load_pickle(traj_acceleration_dict_path)
        
    def _load_pickle(self, file_path: str) -> Dict:
        """加载pickle文件"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def calculate_trajectory_score(self, trajectory: List[str], hour: int, 
                                 od_pair: Tuple[str, str]) -> float:
        """
        计算单个轨迹的综合得分
        
        Args:
            trajectory: 轨迹路段列表
            hour: 时间（小时）
            od_pair: 起终点对 (origin, destination)
            
        Returns:
            float: 综合得分
        """
        # 1. 计算路段热门度得分
        road_score = np.mean([self.road_hour_scores.get((road, hour), 0) for road in trajectory])
        
        # 2. 计算加速度平稳性得分（加速度越小越好）
        acc_score = 1.0 / (1.0 + np.mean([self.traj_accelerations.get(road, 0) for road in trajectory]))
        
        # 3. 计算OD点对重复度得分
        od_score = self.road_od_scores.get(od_pair, 0)
        
        # 综合得分（可以调整权重）
        total_score = 0.4 * road_score + 0.4 * acc_score + 0.2 * od_score
        
        return total_score
    
    def select_anchor_trajectories(self, trajectories: List[List[str]], 
                                 hours: List[int],
                                 od_pairs: List[Tuple[str, str]],
                                 top_k: int = 10) -> List[Tuple[List[str], float]]:
        """
        选择得分最高的轨迹作为锚轨迹
        
        Args:
            trajectories: 轨迹列表
            hours: 对应的时间列表
            od_pairs: 对应的OD点对列表
            top_k: 选择的锚轨迹数量
            
        Returns:
            List[Tuple[List[str], float]]: 选中的锚轨迹及其得分
        """
        # 计算所有轨迹的得分
        scores = []
        for traj, hour, od_pair in zip(trajectories, hours, od_pairs):
            score = self.calculate_trajectory_score(traj, hour, od_pair)
            scores.append((traj, score))
        
        # 按得分排序并返回前top_k个
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
        
        