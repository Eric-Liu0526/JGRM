import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import time
import pickle
from utils import Logger
import argparse
from JGRM import JGRMModel
from task import road_cls, speed_inf, time_est, sim_srh
from evluation_utils import get_road, fair_sampling, get_seq_emb_from_traj_withRouteOnly, get_seq_emb_from_traj_withALLModel, prepare_data
import torch
import os
torch.set_num_threads(5)

dev_id = 7
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
torch.cuda.set_device(dev_id)

def evaluation(city, exp_path, model_name, start_time):
    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256
    model_path = os.path.join(exp_path, 'model', model_name)
    embedding_name = model_name.split('.')[0]

    # load task 1 & task2 label
    feature_df = pd.read_csv("../dataset/didi_{}/edge_features.csv".format(city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    # load adj
    edge_index = np.load("../dataset/didi_{}/line_graph_edge_idx.npy".format(city))
    print("edge_index shape:", edge_index.shape)

    #
    test_node_data = pickle.load(
        open('../dataset/didi_{}/{}_1101_1115_data_sample10w.pkl'.format(city, city), 'rb')) # data_seq_evaluation.pkl

    road_list = get_road(test_node_data)
    print('number of road obervased in test data: {}'.format(len(road_list)))

    # prepare
    num_samples = 'all'
    if num_samples == 'all':
        pass
    elif isinstance(num_samples, int):
        test_node_data = fair_sampling(test_node_data, num_samples)

    road_list = get_road(test_node_data)
    print('number of road obervased after sampling: {}'.format(len(road_list)))

    seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model'] # model.to()包含inplace操作，不需要对象承接
    seq_model.eval()

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare sequence task
    test_seq_data = pickle.load(
        open('../dataset/didi_{}/{}_1101_1115_data_seq_evaluation.pkl'.format(city, city),
             'rb'))
    test_seq_data.reset_index(drop=True, inplace=True)
    test_seq_data = test_seq_data.sample(50000, random_state=0)
    test_index = test_seq_data.index.tolist()
    
    road_counts = pd.read_csv(f'../logs/didi_chengdu_road_count.csv')['traj_count']
    sorted_road = sorted(enumerate(road_counts), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, value in sorted_road]
    hot_roads = sorted_indices[:int(len(sorted_indices)*0.02)]
    hot_percent = []
    for idx, traj in enumerate(test_seq_data['cpath_list']):
        num = set(traj).intersection(set(hot_roads))
        hot_percent.append(len(num)/len(set(traj)))
    sorted_hot_per = sorted(enumerate(hot_percent), key=lambda x: x[1], reverse=True)
    hot_per_ranks = [index for index, value in sorted_hot_per]
    temp_list = []
    for index, value in sorted_hot_per:
        if value == 0:
            temp_list.append(index)
    print(f'num:{len(temp_list)}')
    sorted_per_once = sorted(list(set([value for _, value in sorted_hot_per])), reverse=True)
    # with open(f'../logs/didi_chengdu_traj_degree.pkl', 'rb') as f:
    #     traj_degree = pickle.load(f)
    # test_traj_degree = [traj_degree[i] for i in test_index]
    # degree_ranks = [i for _, i in sorted((v, i) for i, v in enumerate(test_traj_degree))]
    # temp_list = []
    # for i in range(int(len(degree_ranks)*0.01)):
    #     temp_list.append(degree_ranks[i])

    route_length = test_seq_data['route_length'].values
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    route_data[:, :, 1] = torch.zeros_like(route_data[:, :, 1]).long()
    route_data[:, :, 2] = torch.full_like(route_data[:, :, 2], fill_value=-1)
    test_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024)

    # task 3
    _, _, _, ranks = time_est.evaluation(seq_embedding, test_seq_data, num_nodes)
    rank_gaps = []
    # for hot_per_rank in range(len(hot_per_ranks)):
    #     hot_rank = sorted_per_once.index(sorted_hot_per[hot_per_rank][1])
    #     rank = ranks.index(hot_per_ranks[hot_per_rank])
    #     rank_gap = hot_rank - rank // len(sorted_per_once)
    #     rank_gaps.append(abs(rank_gap))
    # print('rank_gap mean: {:.4f}'.format(np.mean(rank_gaps)))
    for idx in temp_list:
        rank = ranks.index(idx)
        rank_gaps.append(rank)
    print('rank_gap mean: {:.4f}'.format(np.mean(rank_gaps)))
    # temp1_list = []
    # for i, idx in enumerate(ranks):
    #     road_set = set(test_seq_data.loc[idx, 'cpath_list'])
    #     inter_num = len(road_set.intersection(set(hot_roads)))
    #     temp1_list.append(inter_num)
    #     if i >len(ranks)*0.01:
    #         break
    

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    test_data = (
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024)


    # task 4
    # detour_base = pickle.load(
    #     open('/data/mazp/dataset/JMTR/didi_{}/detour_base_max5.pkl'.format(city), 'rb'))
    #
    # sim_srh.evaluation2(seq_embedding, None, seq_model, test_seq_data, num_nodes, detour_base, feature_df,
    #                     detour_rate=0.15, fold=10)  # 当road_embedding为None的时候过模型处理，时间特征为空

    geometry_df = pd.read_csv("../dataset/didi_{}/edge_geometry.csv".format(city))

    trans_mat = np.load('../dataset/didi_{}/transition_prob_mat.npy'.format(city))
    trans_mat = torch.tensor(trans_mat)

    sim_srh.evaluation3(seq_embedding, None, seq_model, test_seq_data, num_nodes, trans_mat, feature_df, geometry_df,
                        detour_rate=0.3, fold=10)  # 当road_embedding为None的时候过模型处理，时间特征为空

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))

if __name__ == '__main__':

    city = 'chengdu'
    exp_path = '../research/exp/JTMR_chengdu_250419223112'
    model_name = 'JTMR_chengdu_v1_30_100000_250419223112_29.pt'

    start_time = time.time()
    log_path = os.path.join(exp_path, 'evaluation')
    sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error

    evaluation(city, exp_path, model_name, start_time)

