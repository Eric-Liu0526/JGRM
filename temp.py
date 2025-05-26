import pandas as pd
import pickle as pkl
import numpy as np

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
spatiotemporal_similarity_array_path = 'dataset/didi_chengdu/spatiotemporal_similarity_array.pkl'
spatiotemporal_similarity_array, similarity_rank = pkl.load(open(spatiotemporal_similarity_array_path, 'rb'))
sum_ = np.sum(np.logical_and(spatiotemporal_similarity_array[0] > 0.45,spatiotemporal_similarity_array[0] < 0.56))
print(sum_)
# 将numpy数组写入csv文件
# pd.DataFrame(spatiotemporal_similarity_array).to_csv('temp/spatiotemporal_similarity_array.csv', index=False)
# pd.DataFrame(similarity_rank).to_csv('temp/similarity_rank.csv', index=False)