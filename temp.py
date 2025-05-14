import pandas as pd
import pickle as pkl

# 读取.pkl文件的pd.DataFrame，将标题和第一行内容保存为csv文件
def pkl_to_csv(pkl_file_path, csv_file_path):
    df = pkl.load(open(pkl_file_path, 'rb'))
    first_row = df.iloc[[0]]
    first_row.to_csv(csv_file_path, header=True, index=False)

pkl_to_csv('dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl', 'temp/example.csv')