# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, ChebConv
import parameters  # 确保你有一个名为parameters.py的文件或相应的配置方式

# 图卷积神经网络 + 低维特征嵌入
class CombinedGCN(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels):
        super(CombinedGCN, self).__init__()
        self.embedding = nn.Linear(low_dim_input_size, embedding_dim)
        self.conv1 = GCNConv(high_dim_input_size + embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, output_dim)

    def forward(self, high_dim_features, low_dim_features, edge_index):
        # 对低维特征应用嵌入层
        low_dim_embedded = F.relu(self.embedding(low_dim_features))
        # 将高维特征和嵌入后的低维特征合并
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=1)
        x = self.conv1(combined_features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# 图注意力神经网络 + 低维特征嵌入
class CombinedGAT(nn.Module):
    def __init__(self, high_dim_input_size, low_dim_input_size, embedding_dim, output_dim, hidden_channels, num_heads=8):
        super(CombinedGAT, self).__init__()
        self.embedding = nn.Linear(low_dim_input_size, embedding_dim)
        self.gat1 = GATConv(in_channels=high_dim_input_size + embedding_dim, out_channels=hidden_channels, heads=num_heads, dropout=0.6, concat=True)
        self.gat2 = GATConv(in_channels=hidden_channels * num_heads, out_channels=output_dim, heads=1, concat=False, dropout=0.6)
        
    def forward(self, high_dim_features, low_dim_features, edge_index):
        low_dim_embedded = F.elu(self.embedding(low_dim_features.float()))
        combined_features = torch.cat([high_dim_features, low_dim_embedded], dim=1)
        x = self.gat1(combined_features, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

# 图卷积神经网络
class GCN(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels=128):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dim_nodes, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 2)

    def forward(self, x, adjacency):
        x = self.conv1(x, adjacency)
        x = F.relu(x)
        x = self.conv2(x, adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x

# 切比雪夫-图卷积神经网络
class Cheb(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels):
        super(Cheb, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = ChebConv(dim_nodes, hidden_channels, K=parameters.Kco)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=parameters.Kco)
        self.lin1 = Linear(hidden_channels, 2)

    def forward(self, x, *adjacency):
        x = self.conv1(x, *adjacency)
        x = F.relu(x)
        x = self.conv2(x, *adjacency)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x

# 图注意力网络
class GAT(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels=8, heads=8, output_channels=2):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dim_nodes, hidden_channels, heads=heads, dropout=0.6)
        # On the last layer we reduce the output heads to 1, meaning we concatenate
        # the output of the previous heads from 8*hidden_channels to hidden_channels again.
        self.conv2 = GATConv(hidden_channels*heads, output_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, adjacency):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, adjacency))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, adjacency)
        return F.log_softmax(x, dim=1)
    

# 手动写卷积操作，保留在这但是可能用不上    
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

class GcnNet(nn.Module):
    def __init__(self, input_dim, hidden_channels=64):
        super(GcnNet, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_channels)
        self.gcn2 = GraphConvolution(hidden_channels, 2)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits


