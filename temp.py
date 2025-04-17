import pickle
import pandas as pd

# 替换为你的 .pkl 文件路径
# file_path = 'dataset/didi_chengdu/chengdu_1101_1115_data_sample10w.pkl'
file_path = 'dataset/didi_chengdu/sub_g_traj_dict.pkl'
file_path = 'dataset/didi_chengdu/road_subgraph_node_ids.pkl'

# 打开并加载 .pkl 文件
with open(file_path, 'rb') as file:
    df = pickle.load(file)

# 获取标题行（列名）并将其作为一行数据
columns = df.columns.tolist()
header_row = pd.DataFrame([columns], columns=columns)

# 获取前 5 条数据
top_rows = df.head()

# 将标题行和前 5 条数据合并
output_df = pd.concat([header_row, top_rows], ignore_index=True)

# 将结果输出到 CSV 文件
output_file = 'output.csv'
output_df.to_csv(output_file, index=False, sep=';')

print(f"标题行和前 5 条数据已成功输出到文件：{output_file}")