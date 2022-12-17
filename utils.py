import torch
import torch.nn.functional as F
import numpy as np
import random
import copy


def idNode(data, id_new_value_old):
    data = copy.deepcopy(data)
    data.x = None
    data.y[data.val_id] = -1
    data.y[data.test_id] = -1
    data.y = data.y[id_new_value_old]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype=torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype=torch.long)
    row = data.edge_index[0]
    col = data.edge_index[1]
    row = id_old_value_new[row]
    col = id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data


def shuffleData(data):
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.num_nodes)
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id] = train_id_shuffle
    data = idNode(data, id_new_value_old)

    return data, id_new_value_old


def embedding_mixup(out, data, p_set, u_set, n, lam, mode):
    """
      random.choice: pick 1
      random.sample: pick many, no overlapping
      random.choices: pick many, overlapping
    """
    
    # PU
    if mode == 0:
        lam = lam if lam > 0.5 else (1 - lam)  # so that lam always > 0.5, and weight higher on positive term.
        
        random_pos_idx = random.choices(list(p_set), k=n)
        random_unl_idx = random.sample(list(u_set), k=n)

        new_y = torch.ones(n).type(torch.LongTensor)  # int 64
        new_y_b = torch.zeros(n).type(torch.LongTensor)
        
        return out[random_pos_idx, :] * lam + out[random_unl_idx, :] * (1 - lam), new_y, new_y_b, random_unl_idx, lam

    # PP
    elif mode == 1:
        random_pos_idx = random.sample(list(p_set), k=len(p_set))
        random_pos_idx_ = random.sample(list(p_set), k=len(p_set))
        
        return out[random_pos_idx, :] * lam + out[random_pos_idx_, :] * (1 - lam)

    # UU
    elif mode == 2:
        random_unl_idx = random.sample(list(u_set), k=len(u_set))
        random_unl_idx_ = random.sample(list(u_set), k=len(u_set))
        
        return out[random_unl_idx, :] * lam + out[random_unl_idx_, :] * (1 - lam)
        
