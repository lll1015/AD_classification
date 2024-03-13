# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import json
from data.data_loader import load_and_align_data, create_data_loader
from models.model import CombinedGAT,CombinedGCN
from utils import EarlyStopping  # 假设你有评估和早停的辅助函数
from torch_geometric.data import Data
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.utils import dense_to_sparse
from config import config


# # 加载配置文件
# with open('config.json', 'r') as config_file:
#    config = json.load(config_file)

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 添加日志记录功能
# logging.basicConfig(filename='training.log', level=logging.INFO, 
#                     format='%(asctime)s:%(levelname)s:%(message)s')

# 加载数据并创建数据集
train_dataset, val_dataset, _ = load_and_align_data(high_dim_path = config['data']['high_dim_path'], 
                                                  low_dim_path = config['data']['low_dim_path'],
                                                  labels_path = config['data']['labels_path'],
                                                  test_size = config['data']['test_size'],
                                                  val_size = config['data']['val_size'],
                                                  random_state = config['data']['random_state'])

# 创建数据加载器
train_loader = create_data_loader(train_dataset, batch_size=config['data']['batch_size'], shuffle=config['data']['shuffle'])
val_loader = create_data_loader(val_dataset, batch_size=config['data']['batch_size'], shuffle=config['data']['shuffle'])

model = CombinedGAT(high_dim_input_size=config["model"]["high_dim_input_size"],  # 适当调整这些参数，这里可以写成config
                 low_dim_input_size=config["model"]["low_dim_input_size"],
                 embedding_dim=config["model"]["embedding_dim"],
                 output_dim=config["model"]["output_dim"],  # 根据您的任务调整
                 hidden_channels=config["model"]["hidden_channels"],
                 num_heads=config["model"]["num_heads"]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

# 初始化早停对象
early_stopping = EarlyStopping(patience=config["earlystopping"]["patience"], delta=config["earlystopping"]["delta"])

best_val_loss = float('inf')

for epoch in range(config['train']['epochs']):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for high_dim_features, low_dim_features, labels in train_loader: # , edge_index in train_loader:
        
        batch_size = high_dim_features.size(0)  # 获取当前批次的大小
        
        # 为当前批次生成全连接的邻接矩阵
        adj_matrix = torch.ones((batch_size, batch_size)) - torch.eye(batch_size)
        edge_index, _ = dense_to_sparse(adj_matrix)
        
        
        # 准备数据
        high_dim_features = high_dim_features.to(device)
        low_dim_features = low_dim_features.float().to(device)
        labels = labels.to(device)
        edge_index = edge_index.to(device)

        # 前向传播
        outputs = model(high_dim_features, low_dim_features, edge_index)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total

    # 计算验证集上的损失
    model.eval()
    val_total_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for high_dim_features, low_dim_features, labels in val_loader: # ,edge_index in val_loader:  # 假设你有一个验证集加载器val_loader
            
            batch_size = high_dim_features.size(0)  # 获取当前批次的大小
        
            # 为当前批次生成全连接的邻接矩阵
            adj_matrix = torch.ones((batch_size, batch_size)) - torch.eye(batch_size)
            edge_index, _ = dense_to_sparse(adj_matrix)
                
            high_dim_features = high_dim_features.to(device)
            low_dim_features = low_dim_features.float().to(device)
            labels = labels.to(device)
            edge_index = edge_index.to(device)

            outputs = model(high_dim_features, low_dim_features, edge_index)
            loss = criterion(outputs, labels)
            val_total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_loss =  val_total_loss/len(val_loader)
    val_acc = 100 * val_correct / val_total
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}')
    
    # 早停检查和保存最佳模型
#    early_stopping(val_loss,model)
#    if early_stopping.early_stop:
#        print("Early stopping")
#        break
