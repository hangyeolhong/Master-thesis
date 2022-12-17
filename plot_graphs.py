from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def plt_graph(train_list, test_list, val_list):
    """
    plt.plot([i for i in range(1, 300)], train_acc_list, label='train')
    plt.plot([i for i in range(1, 300)], test_acc_list, label='test')
    plt.plot([i for i in range(1, 300)], val_acc_list, label='val')
    """

    
    plt.plot([i for i in range(1, 300)], train_list, label='train loss', marker="1")
    plt.plot([i for i in range(1, 300)], test_list, label='test loss', marker="2")
    plt.xlabel('epoch')
    plt.ylabel('loss@epoch')
    plt.legend()
    plt.savefig('graph_loss.png')



def plt_tsne(z, data, original_data_y, desc, use_mixup, time_prefix, mixed_p_emb, dataset_name):
    emb = TSNE(n_components=2).fit_transform(z[data.train_id].detach().cpu().numpy())

    labels = original_data_y[data.train_id].detach().cpu().numpy()  # ground truth pn labels
    pu_labels = data.y[data.train_id].detach().cpu().numpy()  # transformed pu labels

    fig, ax = plt.subplots()

    color = ['#CD18FC', '#9FB649']  # '#FC1818' : red


    for idx, label in enumerate(np.unique(labels)):

        if label == 1:
            pu_label_idx_zero = np.where(pu_labels == 0)
            pu_label_idx_one = np.where(pu_labels == 1)
            ground_truth_idx = np.where(labels == 1)

            transformed_idx = np.intersect1d(pu_label_idx_zero, ground_truth_idx)  # ids of 1 -> 0.
            non_transformed_idx = np.intersect1d(pu_label_idx_one, ground_truth_idx)  # ids of originally 1.

            transformed_emb = emb[transformed_idx].squeeze()
            non_transformed_emb = emb[non_transformed_idx].squeeze()

            ax.scatter(x=transformed_emb[:, 0], y=transformed_emb[:, 1], c='#039DFC', label='P in U', alpha=0.1,
                       marker='^')

            ax.scatter(x=non_transformed_emb[:, 0], y=non_transformed_emb[:, 1], c=color[idx], label='P', alpha=0.1)

        else:
            emb_ = emb[np.where(labels == label), :].squeeze()
            ax.scatter(x=emb_[:, 0], y=emb_[:, 1], c=color[idx], label='N in U', alpha=0.1)

    
      
    else:
        ax.legend()
        plt.savefig(desc + time_prefix + '_' + dataset_name + '_withMixup_' + str(use_mixup) + '.png', dpi=300)