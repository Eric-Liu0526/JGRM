import pandas as pd
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # 读取.pkl文件的pd.DataFrame，将标题和第一行内容保存为csv文件
# def pkl_to_csv(pkl_file_path, csv_file_path):
#     df = pkl.load(open(pkl_file_path, 'rb'))
#     first_row = df.iloc[[0]]
#     first_row.to_csv(csv_file_path, header=True, index=False)

# pkl_to_csv('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'temp/example.csv')

'''
def cal_od_pairs(traj_data_path):
    df = pkl.load(open(traj_data_path, 'rb'))
    od_pairs = dict()
    for index, row in df.iterrows():
        cpath_list = row['cpath_list']
        oid = cpath_list[0]
        did = cpath_list[-1]
        od_pairs[(oid, did)] = od_pairs.get((oid, did), 0) + 1
    # 对od_pairs进行排序
    od_pairs = sorted(od_pairs.items(), key=lambda x: x[1], reverse=True)
    return od_pairs

od_pairs = cal_od_pairs('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl')
print(od_pairs)
'''

# 读取anchor的pkl文件
# anchor_indices_path = 'dataset/didi_chengdu/anchor_indices_and_densities.pkl'
# anchor_indices, anchor_densities = pkl.load(open(anchor_indices_path, 'rb'))
# print(anchor_indices)
# print(anchor_densities)

# 读取spatiotemporal_anchors.pkl文件
# spatiotemporal_anchors_path = 'dataset/didi_chengdu/spatiotemporal_anchors.pkl'
# spatiotemporal_anchors = pkl.load(open(spatiotemporal_anchors_path, 'rb'))
# print(spatiotemporal_anchors)

# 读取spatiotemporal_similarity_array.pkl

# 将chengdu_1101_1115_data_sample10w.pkl中的数据的前10000条数据保存为pkl文件
# df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))
# df = df.iloc[:10000]
# pkl.dump(df, open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w_10000.pkl', 'wb'))

'''
def print_anchor_info(anchor_file_path):
    """
    读取锚点文件并打印锚点数量
    
    参数:
        anchor_file_path: 锚点文件的路径
    """
    try:
        # 读取锚点文件
        anchor_data = pkl.load(open(anchor_file_path, 'rb'))
        
        # 根据文件类型打印信息
        if isinstance(anchor_data, tuple):
            if len(anchor_data) == 2:
                # 处理 (anchor_indices, anchor_densities) 或 (selected_anchors, anchor_gains) 格式
                print(f"锚点数量: {len(anchor_data[0])}")
                print(f"第一个元素类型: {type(anchor_data[0])}")
                print(f"第二个元素类型: {type(anchor_data[1])}")
            elif len(anchor_data) == 3:
                # 处理 (similarity_array, similarity_rank) 格式
                print(f"相似度矩阵形状: {anchor_data[0].shape}")
                print(f"相似度排名矩阵形状: {anchor_data[1].shape}")
        else:
            print(f"锚点数据格式: {type(anchor_data)}")
            
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 读取密度锚点文件
    print("\n密度锚点信息:")
    # print_anchor_info('dataset/didi_chengdu/anchor_indices_and_densities.pkl')
    
    # 读取时空锚点文件
    print("\n时空锚点信息:")
    print_anchor_info('dataset/didi_chengdu/spatiotemporal_anchors.pkl')
    
    # 读取相似度矩阵文件
    print("\n相似度矩阵信息:")
    #print_anchor_info('dataset/didi_chengdu/spatiotemporal_similarity_array.pkl') 
'''

'''
# 读取轨迹数据文件
df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'rb'))
# # 将start_time从Unix timestamp转换为datetime格式
# df['start_time'] = pd.to_datetime(df['start_time'], unit='s')
# # 提取日期
# df['date'] = df['start_time'].dt.date
# # 统计每个日期的数量
# date_counts = df['date'].value_counts()
# print(date_counts)
# # 绘制日期分布图
# plt.figure(figsize=(10, 6))
# plt.bar(date_counts.index, date_counts.values)
# plt.xlabel('日期')
# plt.ylabel('数量')
# plt.title('日期分布')
# plt.savefig('temp/date_distribution.png')

# 将轨迹数据中start_time为1541001600到1541520000的轨迹数据保存为pkl文件
df_date_range = df[df['start_time'].isin(range(1541001600, 1541087999))]
pkl.dump(df_date_range, open('dataset/didi_chengdu/chengdu_1101_data.pkl', 'wb'))
'''


'''
# 统计chengdu_1101_1107_data中的轨迹数量
df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_data.pkl', 'rb'))
# 获取每个轨迹的route_length值
# route_length = df['route_length'].values
# sum = 0
# for i in range(len(route_length)):
#     sum += route_length[i] * 3
# print(sum)
# print(sum/len(df))
# 打印轨迹数量
print(len(df))
'''


# # 查看dataset/didi_chengdu/chengdu_1101_1115_data_seq_evaluation.pkl中的数据
# df = pkl.load(open('dataset/didi_chengdu/chengdu_1101_1115_data_seq_evaluation.pkl', 'rb'))
# print(df)

# '''
# 查看dataset/didi_chengdu/transition_prob_mat.npy
transition_prob_mat = np.load('dataset/didi_chengdu/transition_prob_mat.npy')
# 输出非0的值数量
print(np.count_nonzero(transition_prob_mat))
# print(transition_prob_mat)

# 查看dataset/didi_chengdu/line_graph_edge_idx.npy
line_graph_edge_idx = np.load('dataset/didi_chengdu/line_graph_edge_idx.npy')
# print(line_graph_edge_idx)

# 遍历路段连边，检查是否涵盖所有转移概率
num_un_covered = 0
for i in range(len(line_graph_edge_idx[0])):
    start_node = line_graph_edge_idx[0][i]
    end_node = line_graph_edge_idx[1][i]
    if transition_prob_mat[start_node, end_node] == 0:
        # print(f"路段连边{i}未涵盖转移概率: {start_node} -> {end_node}")
        num_un_covered += 1
print(f"未涵盖转移概率的路段连边数量: {num_un_covered}")

# '''