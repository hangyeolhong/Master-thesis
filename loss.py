import torch
import torch.nn.functional as F
import numpy as np
import random
import copy


class PULoss(torch.nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x)), gamma=1, beta=0, nnPU=False):
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")

        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss  # lambda x: (torch.tensor(1., device=x.device) - torch.sign(x))/torch.tensor(2, device=x.device)
        self.nnPU = nnPU
        self.positive = 1
        self.unlabeled = 0  # or -1
        self.min_count = torch.tensor(1.)


    def forward(self, inp, target, test=False):
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        if inp.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = self.prior.cuda()

        n_positive, n_unlabeled = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count,
                                                                                            torch.sum(unlabeled))

        # mapping input into 0~1 using sigmoid
        inp = torch.sigmoid(inp)

        # divide positive, unlabeled x data
        pos_idx = torch.nonzero(torch.where(target == self.positive, 1, 0), as_tuple=True)
        unl_idx = torch.nonzero(torch.where(target == self.unlabeled, 1, 0), as_tuple=True)
        pos_x_data = inp[pos_idx]
        unl_x_data = inp[unl_idx]

        one_hot_target = F.one_hot(target, num_classes=2)
        one_hot_inv_target = F.one_hot(1 - target, num_classes=2)

        # ===== BCELoss ===== #
        # for BCEloss, change Long -> Float tensor + cuda
        pos_y = one_hot_target[pos_idx].type(torch.FloatTensor).to('cuda')
        pos_y_inv = one_hot_inv_target[pos_idx].type(torch.FloatTensor).to('cuda')
        unl_y = one_hot_target[unl_idx].type(torch.FloatTensor).to('cuda')
        

        criterion = torch.nn.BCELoss(reduction='sum')
        y_positive = criterion(pos_x_data, pos_y) / n_positive  # mean
        y_positive_inv = criterion(pos_x_data, pos_y_inv) / n_positive
        y_unlabeled = criterion(unl_x_data, unl_y) / n_unlabeled

        if not y_positive.nelement():
            y_positive = 0
        if not y_positive_inv.nelement():
            y_positive_inv = 0
        if not y_unlabeled.nelement():
            y_unlabeled = 0
        
        """
        # ===== binary_cross_entropy ===== #
        y_positive = F.binary_cross_entropy(pos_x_data, pos_y)  # mean!
        y_positive_inv = F.binary_cross_entropy(pos_x_data, pos_y_inv)
        y_unlabeled = F.binary_cross_entropy(unl_x_data, unl_y)

        # ===== nll_loss ===== #
        pos_y_label = target[pos_idx]
        unl_y_label = target[unl_idx]
        y_positive = F.nll_loss(pos_x_data, torch.LongTensor(pos_y_label).cuda())
        y_positive_inv = F.nll_loss(pos_x_data, torch.LongTensor(pos_y_label - 1).cuda())
        y_unlabeled = F.nll_loss(unl_x_data, torch.LongTensor(unl_y_label).cuda())
        """

        positive_risk = self.prior * y_positive
        negative_risk = -self.prior * y_positive_inv + y_unlabeled

        if negative_risk < -self.beta and self.nnPU:
            return -self.gamma * negative_risk
        else:
            return positive_risk + negative_risk
