import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
from utils import weight_init
from dataloader import get_train_loader, random_mask, DynamicBatchDataset
from utils import setup_seed
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from JGRM import JGRMModel
from cl_loss import get_traj_match_loss
from dcl import DCL
import os
import pickle as pkl

# terminal: python JGRM_train.py >> logs/train-$(date "+%Y%m%d%H%M").txt

dev_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
torch.cuda.set_device(dev_id)
torch.set_num_threads(10)
print(f'Note: 加入对比学习线性衰减')
print(f'Note: 相似度计算加入jaccard相似度')

def contrastive_loss_batch(anchor_embedding, pos_embeddings_list, neg_embeddings_list, temperature):
    """
    anchor_embedding: [D]
    pos_embeddings_list: [P, D]
    neg_embeddings_list: [N, D]
    temperature: scalar

    Returns:
        Scalar loss (单个anchor)
    """
    # 归一化
    anchor_embedding = F.normalize(anchor_embedding, dim=-1)
    pos_embeddings_list = F.normalize(pos_embeddings_list, dim=-1)
    neg_embeddings_list = F.normalize(neg_embeddings_list, dim=-1)

    # 计算相似度
    sim_pos = torch.matmul(pos_embeddings_list, anchor_embedding) / temperature  # [P]
    sim_neg = torch.matmul(neg_embeddings_list, anchor_embedding) / temperature  # [N]

    exp_pos = torch.exp(sim_pos)  # [P]
    exp_neg = torch.exp(sim_neg)  # [N]

    denom = torch.sum(exp_pos) + torch.sum(exp_neg)  # scalar

    loss = -torch.log(exp_pos / denom)  # [P]
    loss = loss.mean()  # 平均

    return loss

def get_cl_weight(epoch, num_epochs, initial_weight=1.0, final_weight=0.1):
    """
    计算对比学习loss的权重，随着epoch线性衰减
    Args:
        epoch: 当前epoch
        num_epochs: 总epoch数
        initial_weight: 初始权重
        final_weight: 最终权重
    Returns:
        当前epoch的权重
    """
    return initial_weight - (initial_weight - final_weight) * (epoch / num_epochs)

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
    num_positive = config['num_positive']

    # define seed
    setup_seed(seed)

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

    # 首先获取原始数据加载器
    original_loader = get_train_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed)
    
    # 加载SpatiotemporalAnchorSelector生成的文件
    similarity_array, similarity_rank = pkl.load(open(config['similarity_matrix_path'], 'rb'))
    anchor_tids, _ = pkl.load(open(config['anchor_indices_path'], 'rb'))
    
    # 创建动态批次数据集
    dynamic_dataset = DynamicBatchDataset(
        original_dataset=original_loader.dataset,
        similarity_matrix=similarity_array,
        anchor_tids=anchor_tids,
        batch_size=batch_size,
        num_positive=num_positive,
        num_negative=batch_size-num_positive-1,
        neg_threshold=config.get('neg_threshold', 0.5),
        top_k=config.get('top_k', 100)
    )
    
    # 创建新的数据加载器
    train_loader = torch.utils.data.DataLoader(
        dynamic_dataset,
        batch_size=1,  # 因为批次已经在数据集中组织好了
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
        drop_last=True
    )
    
    print('dataset is ready.')

    epoch_step = len(train_loader.dataset.batches) // batch_size
    total_steps = epoch_step * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        # 设置当前epoch
        train_loader.dataset.set_epoch(epoch)
        
        for idx, batch in enumerate(train_loader):
            gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length, tid = batch
            # 由于batch_size=1，需要去掉第一个维度
            gps_data = gps_data.squeeze(0)
            gps_assign_mat = gps_assign_mat.squeeze(0)
            route_data = route_data.squeeze(0)
            route_assign_mat = route_assign_mat.squeeze(0)
            gps_length = gps_length.squeeze(0)
            tid = tid[0]  # 因为batch_size=1，所以tid是一个列表，取第一个元素

            masked_route_assign_mat, masked_gps_assign_mat = random_mask(gps_assign_mat, route_assign_mat, gps_length,
                                                                         vocab_size, mask_length, mask_prob)

            route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length =\
                route_data.cuda(), masked_route_assign_mat.cuda(), gps_data.cuda(), masked_gps_assign_mat.cuda(), route_assign_mat.cuda(), gps_length.cuda()

            gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, \
            gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep \
                = model(route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length)

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

            # (GRM LOSS) get gps & route rep matching loss
            tau = 0.07
            match_loss = get_traj_match_loss(gps_traj_rep, route_traj_rep, model, batch_size, tau)

            # (TC LOSS) route Contrast learning
            # norm_route_traj_rep = F.normalize(route_traj_rep, dim=1)
            # loss_fn = DCL(temperature=0.07)
            # cl_loss = loss_fn(norm_route_traj_rep, norm_route_traj_rep)

            # 计算新的对比学习loss
            gps_cl_loss = contrastive_loss_batch(
                anchor_embedding=gps_traj_rep[0],
                pos_embeddings_list=gps_traj_rep[1:num_positive+1],
                neg_embeddings_list=gps_traj_rep[num_positive+1:],
                temperature=0.07
            )

            route_cl_loss = contrastive_loss_batch(
                anchor_embedding=route_traj_rep[0],
                pos_embeddings_list=route_traj_rep[1:num_positive+1],
                neg_embeddings_list=route_traj_rep[num_positive+1:],
                temperature=0.07
            )

            # 计算当前epoch的对比学习权重
            cl_weight = get_cl_weight(epoch, num_epochs)

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

            # MLM 1 LOSS + MLM 2 LOSS + GRM LOSS + CL LOSS
            loss = (route_mlm_loss + gps_mlm_loss + 2*match_loss + cl_weight*(gps_cl_loss + route_cl_loss)) / (3 + 2*cl_weight)

            step = epoch_step*epoch + idx
            writer.add_scalar('match_loss/match_loss', match_loss, step)
            writer.add_scalar('mlm_loss/gps_mlm_loss', gps_mlm_loss, step)
            writer.add_scalar('mlm_loss/route_mlm_loss', route_mlm_loss, step)
            writer.add_scalar('cl_loss/gps_cl_loss', gps_cl_loss, step)
            writer.add_scalar('cl_loss/route_cl_loss', route_cl_loss, step)
            writer.add_scalar('cl_weight', cl_weight, step)
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


