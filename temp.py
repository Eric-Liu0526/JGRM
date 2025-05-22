import pandas as pd
import pickle as pkl

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
anchor_indices_path = 'dataset/didi_chengdu/anchor_indices_and_densities.pkl'
anchor_indices, anchor_densities = pkl.load(open(anchor_indices_path, 'rb'))
print(anchor_indices)
print(anchor_densities)


