# data_loader.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.utils import dense_to_sparse

class SubjectDataset(Dataset):
    def __init__(self, high_dim_features, low_dim_features, labels):#,edge_index):
        # 由于我们将在模型中处理低维特征的嵌入，所以我们在这里不需要转换为Tensor
        self.high_dim_features = high_dim_features
        self.low_dim_features = low_dim_features
        self.labels = labels
        # self.edge_index = edge_index

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        high_dim_sample = torch.tensor(self.high_dim_features.iloc[idx].values, dtype=torch.float)
        low_dim_sample = self.low_dim_features.iloc[idx].values  # 作为NumPy数组保持
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return high_dim_sample, low_dim_sample, label #,self.edge_index

def create_full_connected_edge_index(num_nodes):
    # 生成一个全连接网络的邻接矩阵
    adj_matrix = torch.ones((num_nodes, num_nodes)) - torch.eye(num_nodes)
    # 将邻接矩阵转换为边索引
    edge_index, _ = dense_to_sparse(adj_matrix)
    return edge_index

def load_and_align_data(high_dim_path, low_dim_path, labels_path, test_size=0.2, val_size = 0.1,random_state=42):
    # 加载数据
    high_dim_df = pd.read_csv(high_dim_path)
    low_dim_df = pd.read_csv(low_dim_path)
    labels_df = pd.read_csv(labels_path)

    # 假设第一列是subject_id，对齐数据
    merged_df = high_dim_df.merge(low_dim_df, on='PTID',how = 'inner').merge(labels_df, on='PTID',how = 'inner')

    # 假设'high_dim_data'和'low_dim_data'分别是包含在您的CSV文件列名中的高维和低维数据标识
    high_dim_cols = [col for col in merged_df.columns if 'high_' in col]
    low_dim_cols = [col for col in merged_df.columns if 'low_' in col]
    labels = merged_df['label_DX'].values

    # 分离出高维和低维特征
    high_dim_features = merged_df[high_dim_cols]
    low_dim_features = merged_df[low_dim_cols]
    
    # # 归一化特征
    # scaler_high = MinMaxScaler()
    # scaler_low = MinMaxScaler()
    
    # high_dim_features = scaler_high.fit_transform(high_dim_features)
    # low_dim_features = scaler_low.fit_transform(low_dim_features)

    # 划分训练集和测试集
    initial_train_high, test_high, initial_train_low, test_low, initial_train_labels, test_labels = train_test_split(
        high_dim_features, low_dim_features, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
     # 进一步划分出验证集
    train_high, val_high, train_low, val_low, train_labels, val_labels = train_test_split(
        initial_train_high, initial_train_low, initial_train_labels, test_size=val_size, random_state=random_state, stratify=initial_train_labels)

    # 在这里调用 create_full_connected_edge_index 函数来创建 edge_index
    # num_nodes = len(merged_df)
    # edge_index = create_full_connected_edge_index(num_nodes)

    # 创建数据集实例
    train_dataset = SubjectDataset(train_high, train_low, train_labels) #, edge_index)
    val_dataset = SubjectDataset(val_high, val_low, val_labels) #, edge_index)
    test_dataset = SubjectDataset(test_high, test_low, test_labels) #, edge_index)

    return train_dataset, val_dataset, test_dataset

def create_data_loader(dataset, batch_size=32, shuffle=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader