import os.path as osp
import os

# from graph_conv import GraphConv
from model import GCN, GAT, MLP

import torch
import torch.nn.functional as F

import numpy as np
import random
import time
from args import parse_args
# from pytorchtools import EarlyStopping
from loss import PULoss
from utils import embedding_mixup
from metrics import get_metrics, get_confusion_matrix
from tqdm import tqdm
from plot_graphs import plt_test_tsne

import os.path as osp

import torch
import numpy as np

from dataset import Dataset



# argument parsing & setting
args = parse_args()
p, new_nodes_count, mode = args.p, args.n, args.m
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# set random seed
SEED = 0
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
np.random.seed(SEED)  # Numpy module.
random.seed(SEED)  # Python random module.

# load data
dataset_name = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
data_ = Dataset(dataset_name, path, p)
idx_train, idx_valid, idx_test = data_.idx_train, data_.idx_valid, data_.idx_test


# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device, args.gpu_id)


# Initialize Model
if args.model == "GCN":
    model = GCN(hidden_channels=256, in_channel=data_.num_node_features, out_channel=2).to(device)
elif args.model == "GAT":
    model = GAT(hidden_channels=256, in_channel=data_.num_node_features, out_channel=2).to(device)
elif args.model == "MLP":
    model = MLP(hidden_channels=256, in_channel=data_.num_node_features, out_channel=2).to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    

# func train one epoch
def train(data, class_prior, p_set, u_set, PinU, original_unl_train_id):
    model.train()

    if args.mixup:
        lam = np.random.beta(0.2, 0.2)
    else:
        lam = 1.0

    data = data.to(device)

    optimizer.zero_grad()
    loss_fct = PULoss(prior=torch.tensor(class_prior))

    if args.mixup:
        out = model(data.x, data.edge_index)

        # calculate cosine similarity
        candidate_ids = []

        for i in tqdm(p_set):
            pos_idx_for_cos_sim = i
            mx, mx_id = -1, 0

            for j in (u_set):
              if j in candidate_ids:
                continue
                
              unl_idx_for_cos_sim = j
              cos_sim = (F.cosine_similarity(out[pos_idx_for_cos_sim], out[unl_idx_for_cos_sim], dim=0))

              if cos_sim > mx:
                mx = cos_sim
                mx_id = unl_idx_for_cos_sim
                
                if cos_sim == 1.0:
                  break
                  
            candidate_ids.append(mx_id)

        np.random.shuffle(candidate_ids)
        candidate_ids = candidate_ids[:10]
        
        mixed_out, new_y, new_y_b, idxs, new_lam = embedding_mixup(out, data, p_set, candidate_ids, 10, lam, 0)

        # Mix(P, U)
        loss = loss_fct(mixed_out, new_y) * new_lam
        
        # Mix(P, P)
        mixed_p = embedding_mixup(out, data, p_set, PinU, new_nodes_count, lam, 1)
        loss += loss_fct(mixed_p, data.y[p_set])

        # Mix(U, U)
        # idxs = []
        rest_u_set = np.setdiff1d(u_set, np.array(idxs))
        mixed_u = embedding_mixup(out, data, p_set, rest_u_set, new_nodes_count, lam, 2)

        loss += loss_fct(mixed_u, data.y[rest_u_set])

    else:
        out = model(data.x, data.edge_index)
        loss = loss_fct(out[idx_train], data.y[idx_train])

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate(data, class_prior, original_data_y, ids):
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device))
    loss_fct = PULoss(prior=torch.tensor(class_prior))
    loss = loss_fct(out[ids], original_data_y[ids])

    return loss.item()


@torch.no_grad()
def test(data, original_data_y):
    model.eval()

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

    print("  predicted positive / positive nodes: ", torch.sum(pred[data_.final_pos_train_id]).item(), "/",
          pred[data_.final_pos_train_id].size())
    print("  predicted positive / transformed unlabeled nodes: ", torch.sum(pred[data_.training_pos_to_unl_ids]).item(), "/",
          pred[data_.training_pos_to_unl_ids].size())

    correct = pred.eq(original_data_y.to(device))
    accs = []

    for id_ in [idx_train, idx_test, idx_valid]:
      accs.append(correct[id_].sum().item() / id_.shape[0])

    return accs, pred


best_acc = 0
accord_epoch = 0
accord_train_acc = 0
accord_train_loss = 0

tm = time.localtime()
time_prefix = str(tm.tm_year) + '{0:02d}{1:02d}_{2:02d}{3:02d}{4:02d}_'.format(tm.tm_mon, tm.tm_mday, tm.tm_hour,
                                                                               tm.tm_min, tm.tm_sec)


for epoch in tqdm(range(1, 300)):
    loss = train(data_.data, data_.class_prior, data_.final_pos_train_id, data_.final_unl_train_id, data_.training_pos_to_unl_ids,
                 data_.original_unl_train_id)
    val_loss = validate(data_.data, data_.class_prior, data_.binarized_label, idx_valid)
    test_loss = validate(data_.data, data_.class_prior, data_.binarized_label, idx_test)

    accs, pred = test(data_.data, data_.binarized_label)

    Precision_train, Recall_train, F1_train = get_metrics(pred, data_.binarized_label.to(device), idx_train)
    Precision, Recall, F1 = get_metrics(pred, data_.binarized_label.to(device), idx_test)  # test

    print("---------")
    print(
        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {accs[0]:.4f}, Test Acc: {accs[2]:.4f}, Val Acc: {accs[1]:.4f}')
    print(f'\tPrecision_train: {Precision_train.item():.4f}\tRecall_train: {Recall_train.item():.4f}\tF1_train: {F1_train:.4f}')
    print(f'\tPrecision: {Precision.item():.4f}\tRecall: {Recall.item():.4f}\tF1: {F1:.4f}')


if args.save:
    torch.save(model.state_dict(),
               time_prefix + 'mode' + str(mode) + '_' + dataset_name + '_' + str(p) + '_withMixup_' + str(
                   args.mixup) + '_model.pt')
    print("model saved")


# load model & TSNE visualization
if args.plt:
    model.load_state_dict(
        torch.load(time_prefix + 'mode' + str(mode) + '_' + dataset_name + '_' + str(p) + '_withMixup_' + str(
            args.mixup) + '_model.pt'))

    z = model(data_.data.x.to(device), data_.data.edge_index.to(device))

    plt_test_tsne(z, data_.data, data_.binarized_label, args.desc, args.mixup, time_prefix, dataset_name)
    print("tsne graph is saved.")