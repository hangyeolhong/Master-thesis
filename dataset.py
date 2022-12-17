import os.path as osp
import os

import torch
from torch_geometric.datasets import Planetoid, Coauthor, WebKB
import torch_geometric.transforms as T

import pdb
import numpy as np
import random
import copy
import os.path as osp


from load_wikipedia import WikipediaNetwork


class Dataset():

    def __init__(self, dataset_name, path, p):
    
      if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
        self.dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
      elif dataset_name in ['Chameleon', 'Squirrel']:
        self.dataset = WikipediaNetwork(path, dataset_name, transform=T.NormalizeFeatures())
      # self.dataset = WebKB(path, dataset_name, transform=T.NormalizeFeatures())
    
      self.data = self.dataset[0]

      self.num_node_features = self.data.num_node_features

      self.largest_class = self.find_largest()

      self.binarized_label = self.binarize()  # original_data_y

      self.data.y = copy.deepcopy(self.binarized_label)  # only y is different
    
      self.idx_train, self.idx_valid, self.idx_test = self.get_train_val_test(self.data.num_nodes, train_size=0.6, val_size=0.8)
      
      self.pos_train_id = self.idx_train[np.where(self.data.y[self.idx_train] == 1)]

      self.num_pos_train_id, self.num_neg_train_id = self.pos_train_id.shape[0], self.data.y[self.idx_train].shape[0] - self.pos_train_id.shape[0]

      self.class_prior = round(self.num_pos_train_id / (self.num_pos_train_id + self.num_neg_train_id), 3)  # maybe this is right & training set , (fixed)

      self.data.y, self.final_pos_train_id, self.final_unl_train_id, self.training_pos_to_unl_ids, self.original_unl_train_id = self.get_positive_unlabeled(int(p * self.num_pos_train_id))
      
      
    def find_largest(self):
        res = []
        
        for c in range(self.dataset.num_classes):
          idx = (self.data.y == c).nonzero(as_tuple=False).view(-1)
          res.append(len(idx))
          
        return res.index(max(res))
    
    
    def binarize(self):
        """
        Treat the label with the largest # of nodes as P, and the rest as N for binary classification. 
        """
        
        original_data_y = copy.deepcopy(self.data.y)
        
        for c in range(self.dataset.num_classes):
          idx = (self.data.y == c).nonzero(as_tuple=False).view(-1)
          if c == self.largest_class:
              original_data_y[idx] = 1  # P
          else:
              original_data_y[idx] = 0  # N
        
        return original_data_y
        
        
    def get_train_val_test(self, nnodes, train_size=0.6, val_size=0.8):
        node_id = np.arange(nnodes)
        np.random.shuffle(node_id)
        idx_train = node_id[:int(nnodes * train_size)]
        idx_valid = node_id[int(nnodes * train_size):int(nnodes * val_size)]
        idx_test = node_id[int(nnodes * val_size):]
        
        return idx_train, idx_valid, idx_test
        
        
    def get_positive_unlabeled(self, training_pos_set_len):
        y = copy.deepcopy(self.binarized_label)
        
        # transform positive data -> unlabeled data
        np.random.shuffle(self.pos_train_id)
        training_pos_ids = self.pos_train_id[:training_pos_set_len]
        training_pos_to_unl_ids = self.pos_train_id[training_pos_set_len:]
        original_unl_train_id = self.idx_train[np.where(self.data.y[self.idx_train] == 0)]
        y[training_pos_to_unl_ids] = 0  # 1 -> 0
        
        
        final_pos_train_id = self.idx_train[np.where(y[self.idx_train] == 1)]  # all of y is 1
        final_unl_train_id = self.idx_train[np.where(y[self.idx_train] == 0)]  # all of y is 0
          
        return y, final_pos_train_id, final_unl_train_id, training_pos_to_unl_ids, original_unl_train_id
