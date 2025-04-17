import pickle
from tqdm import tqdm
import pandas as pd

def load_data(file_path):
    # 打开并加载 .pkl 文件
    with open(file_path, 'rb') as file:
        df = pickle.load(file)
    return df

def load_road_ids(file_path):
    df = pd.read_csv(file_path)
    road_ids = df['fid'].tolist()
    return road_ids

def count_road_with_trajs(df, road_ids):
    """统计每一个路段有多少条轨迹经过

    Args:
        df (DataFrame): 轨迹数据
        road_ids (list): 路段ID列表
    """
    road_count_dict = dict()

    # 遍历 DataFrame 中的每一行
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        # 获取轨迹经过的路段列表
        road_list = row['cpath_list']
        # 对每个路段的轨迹数量加 1
        for road in road_list:
            if road in road_count_dict:
                road_count_dict[road] += 1
            else:
                road_count_dict[road] = 1 

    # 打印每个路段的轨迹数量
    # for i, count in enumerate(road_count):
    #     print(f'Road {i}: {count} trajs')
    
    with open(f'logs/didi_{dataset_name}_road_count.csv', 'w') as file:
        file.write('fid,traj_count\n')
        for road_id in road_ids:
            if road_id in road_count_dict:
                file.write(f'{road_id},{road_count_dict[road_id]}\n')
            else:
                file.write(f'{road_id},0\n')


dataset_name = 'chengdu'
traj_file_path = f'dataset/didi_{dataset_name}/{dataset_name}_1101_1115_data_sample10w.pkl'
road_feat_file_path = f'dataset/didi_{dataset_name}/edge_features.csv'
data = load_data(traj_file_path)
count_road_with_trajs(data, load_road_ids(road_feat_file_path))