import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
from utils import weight_init
from dataloader import get_train_loader, random_mask
from utils import setup_seed
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from JGRM import JGRMModel
from cl_loss import get_traj_match_loss
import os
import pickle as pkl
import networkx as nx
from scipy.sparse import csr_array

dev_id = 5
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
torch.cuda.set_device(dev_id)
torch.set_num_threads(10)
print("v1.0.3")
print("PS:batch通过随机采样")
print("PS:加上图增强")
print("PS:加上图损失")
print("PS:设置注意力头为1")
print("PS:loss = (route_mlm_loss + gps_mlm_loss + 2*match_loss + 10*graph_consistency) / 4")
'''
def graph_consistency_loss(original_rep, updated_rep, edge_index_csr):
    # 将 scipy.sparse.csr_array 转换为 PyTorch 张量
    edge_index = torch.tensor(edge_index_csr.nonzero(), dtype=torch.long)
    
    # 计算图结构一致性损失
    # 使用余弦相似度来度量原始和更新节点表示之间的相似度
    cosine_sim = F.cosine_similarity(original_rep, updated_rep, dim=-1)
    
    # 只考虑图中的边，即邻接节点之间的相似度
    consistency_loss = torch.mean(1 - cosine_sim[edge_index[0]])
    
    return consistency_loss
'''
def graph_consistency_loss(original_rep, updated_rep, edge_index_csr):
    # 提取边索引
    row, col = edge_index_csr.nonzero()  # 邻接矩阵中有值的位置，即 (i,j) 存在边
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)

    # 获取边两端的节点表示
    original_edge_sim = F.cosine_similarity(original_rep[row], original_rep[col], dim=-1)
    updated_edge_sim = F.cosine_similarity(updated_rep[row], updated_rep[col], dim=-1)

    # 相似度差异越小越好
    loss = F.mse_loss(updated_edge_sim, original_edge_sim)

    return loss


def train(config):

    city = config['city']

    vocab_size = config['vocab_size']
    num_samples = config['num_samples']
    data_path = config['data_path']
    adj_path = config['adj_path']
    retrain = config['retrain']
    save_path = config['save_path']

    num_worker = config['num_worker']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    warmup_step = config['warmup_step']
    weight_decay = config['weight_decay']

    route_min_len = config['route_min_len']
    route_max_len = config['route_max_len']
    gps_min_len = config['gps_min_len']
    gps_max_len = config['gps_max_len']

    road_feat_num = config['road_feat_num']
    road_embed_size = config['road_embed_size']
    gps_feat_num = config['gps_feat_num']
    gps_embed_size = config['gps_embed_size']
    route_embed_size = config['route_embed_size']

    hidden_size = config['hidden_size']
    drop_route_rate = config['drop_route_rate'] # route_encoder
    drop_edge_rate = config['drop_edge_rate']   # gat
    drop_road_rate = config['drop_road_rate']   # sharedtransformer

    verbose = config['verbose']
    version = config['version']
    seed = config['random_seed']

    mask_length = config['mask_length']
    mask_prob = config['mask_prob']

    # define seed
    setup_seed(seed)

    sub_g_dict = dict()
    # 读取子图轨迹字典：{sub_g_id: traj_set}
    with open(f'dataset/didi_chengdu/sub_g_traj_dict.pkl', 'rb') as f:
        sub_g_traj_dict = pkl.load(f)
    # 通过子图采样划分训练batch
    for sub_g_id in sub_g_traj_dict.keys():
        graph = pkl.load(open(f'dataset/didi_chengdu/traj_subg_{sub_g_id}.pkl', 'rb'))
        sub_g_dict[sub_g_id] = graph

    # define model, parmeters and optimizer
    edge_index = np.load(adj_path)
    model = JGRMModel(vocab_size, route_max_len, road_feat_num, road_embed_size, gps_feat_num,
                      gps_embed_size, route_embed_size, hidden_size, edge_index, drop_edge_rate, drop_route_rate, drop_road_rate, mode='x').cuda()
    # Modify it to your own directory
    init_road_emb = torch.load('dataset/didi_{}/init_w2v_road_emb.pt'.format(city), map_location='cuda:{}'.format(dev_id))
    model.node_embedding.weight = torch.nn.Parameter(init_road_emb['init_road_embd'])
    model.node_embedding.requires_grad_(True)
    print('load parameters in device {}'.format(model.node_embedding.weight.device)) # check process device

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # exp information
    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    model_name = 'JTMR_{}_{}_{}_{}_{}'.format(city, version, num_epochs, num_samples, nowtime)
    model_path = os.path.join(save_path, 'JTMR_{}_{}'.format(city, nowtime), 'model')
    log_path = os.path.join(save_path, 'JTMR_{}_{}'.format(city, nowtime), 'log')

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    checkpoints = [f for f in os.listdir(model_path) if f.startswith(model_name)]
    writer = SummaryWriter(log_path)
    if not retrain and checkpoints:
        checkpoint_path = os.path.join(model_path, sorted(checkpoints)[-1])
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.apply(weight_init)

    batch2g_dict, train_loader = get_train_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed)
    print('dataset is ready.')

    epoch_step = train_loader.dataset.route_data.shape[0] // batch_size
    total_steps = epoch_step * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            batchg_id = batch2g_dict[idx]
            gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length, traj_idx = batch

            # batch_graph正例的邻接矩阵（负例为其他子图对象，无连接）
            traj_idx = [tensor.item() for tensor in traj_idx]
            filtered_traj_idx = set(traj_idx).intersection(sub_g_dict[batchg_id].nodes)
            batchg = sub_g_dict[batchg_id].subgraph((filtered_traj_idx)).copy()
            batchg_adj = nx.adjacency_matrix(batchg, nodelist=filtered_traj_idx)

            masked_route_assign_mat, masked_gps_assign_mat = random_mask(gps_assign_mat, route_assign_mat, gps_length,
                                                                         vocab_size, mask_length, mask_prob)

            route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length =\
                route_data.cuda(), masked_route_assign_mat.cuda(), gps_data.cuda(), masked_gps_assign_mat.cuda(), route_assign_mat.cuda(), gps_length.cuda()

            # tag: 图增强
            
            gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, \
            gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep \
                = model(route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, batchg_adj)
            '''
            gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, \
            gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep \
                = model(route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length)
            '''
            # flatten road_rep
            mat2flatten = {}
            y_label = []
            route_length = (route_assign_mat != model.vocab_size).int().sum(1)
            gps_road_list, route_road_list, gps_road_joint_list, route_road_joint_list = [], [], [], []
            now_flatten_idx = 0
            for i, length in enumerate(route_length):
                y_label.append(route_assign_mat[i, :length]) # the mask location in route and gps traj is same
                gps_road_list.append(gps_road_rep[i, :length])
                route_road_list.append(route_road_rep[i, :length])
                gps_road_joint_list.append(gps_road_joint_rep[i, :length])
                route_road_joint_list.append(route_road_joint_rep[i, :length])
                for l in range(length):
                    mat2flatten[(i, l)] = now_flatten_idx
                    now_flatten_idx += 1

            y_label = torch.cat(y_label, dim=0)
            gps_road_rep = torch.cat(gps_road_list, dim=0)
            route_road_rep = torch.cat(route_road_list, dim=0)
            gps_road_joint_rep = torch.cat(gps_road_joint_list, dim=0)
            route_road_joint_rep = torch.cat(route_road_joint_list, dim=0)

            # project rep into the same space
            gps_traj_rep = model.gps_proj_head(gps_traj_rep)
            route_traj_rep = model.route_proj_head(route_traj_rep)

            # 获取图结构一致性损失
            graph_consistency = graph_consistency_loss(gps_road_joint_rep, route_road_joint_rep, batchg_adj)
    
            # (GRM LOSS) get gps & route rep matching loss
            tau = 0.07
            match_loss = get_traj_match_loss(gps_traj_rep, route_traj_rep, model, batch_size, tau)

            # (TC LOSS) route Contrast learning
            # norm_route_traj_rep = F.normalize(route_traj_rep, dim=1)
            # loss_fn = DCL(temperature=0.07)
            # cl_loss = loss_fn(norm_route_traj_rep, norm_route_traj_rep)

            # prepare label and mask_pos
            masked_pos = torch.nonzero(route_assign_mat != masked_route_assign_mat)
            masked_pos = [mat2flatten[tuple(pos.tolist())] for pos in masked_pos]
            y_label = y_label[masked_pos].long()

            # (MLM 1 LOSS) get gps rep road loss
            gps_mlm_pred = model.gps_mlm_head(gps_road_joint_rep) # project head update
            masked_gps_mlm_pred = gps_mlm_pred[masked_pos]
            gps_mlm_loss = nn.CrossEntropyLoss()(masked_gps_mlm_pred, y_label)

            # (MLM 2 LOSS) get route rep road loss
            route_mlm_pred = model.route_mlm_head(route_road_joint_rep) # project head update
            masked_route_mlm_pred = route_mlm_pred[masked_pos]
            route_mlm_loss = nn.CrossEntropyLoss()(masked_route_mlm_pred, y_label)

            # MLM 1 LOSS + MLM 2 LOSS + GRM LOSS
            loss = (route_mlm_loss + gps_mlm_loss + 2*match_loss + 10*graph_consistency) / 4

            step = epoch_step*epoch + idx
            writer.add_scalar('match_loss/match_loss', match_loss, step)
            writer.add_scalar('mlm_loss/gps_mlm_loss', gps_mlm_loss, step)
            writer.add_scalar('mlm_loss/route_mlm_loss', route_mlm_loss, step)
            writer.add_scalar('graph_loss/graph_loss', graph_consistency, step)
            writer.add_scalar('loss', loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (idx + 1) % verbose:
                t = datetime.now().strftime('%m-%d %H:%M:%S')
                print(f'{t} | (Train) | Epoch={epoch}\tbatch_id={idx + 1}\tloss={loss.item():.4f}')

        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_path, "_".join([model_name, f'{epoch}.pt'])))

    return model

if __name__ == '__main__':
    config = json.load(open('config/chengdu.json', 'r'))
    train(config)


