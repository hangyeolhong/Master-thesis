import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing, GATConv
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
import pdb


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='mean', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, x_cen):
        h = torch.matmul(x, self.weight)
        aggr_out = self.propagate(edge_index, size=None, h=h, edge_weight=None)
        return aggr_out + self.lin(x_cen)

    def message(self, h_j):
        return h_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_channel, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(1 * hidden_channels, out_channel)

    def forward(self, x0, edge_index):
        x1 = self.conv1(x0, edge_index, x0)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        
        x2 = self.conv2(x1, edge_index, x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)
        
        x = self.lin(x2)

        return x



class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(GAT, self).__init__()
        self.hid = 8  # hidden_channels
        self.in_head = 8
        self.out_head = 1
        
        self.conv1 = GATConv(in_channel, hidden_channels, heads=self.in_head, dropout=0.4)
        self.conv2 = GATConv(hidden_channels * self.in_head, hidden_channels, concat=False, heads=self.out_head, dropout=0.4)
        self.lin = torch.nn.Linear(1 * hidden_channels, out_channel)


    def forward(self, x0, edge_index):
        x1 = F.dropout(x0, p=0.4, training=self.training)
        x1 = self.conv1(x1, edge_index)
        x2 = F.relu(x1)
        
        x2 = F.dropout(x2, p=0.4, training=self.training)
        x2 = self.conv2(x2, edge_index)
        
        x = self.lin(x2)
        return x
        

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, in_channel, out_channel):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(1 * in_channel, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channel)


    def forward(self, x0, edge_index):
        x1 = self.lin1(x0)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.4, training=self.training)

        x2 = self.lin2(x1)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.4, training=self.training)

        x = self.lin3(x2)
        return x