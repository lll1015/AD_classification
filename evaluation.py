import torch
from models.model import GATModel  # 确保这个import语句与你的项目结构相匹配
from data.data_loader import create_data_loader, load_and_align_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import numpy as np

# 加载配置文件
with open('models/config.json', 'r') as config_file:
    config = json.load(config_file)

# 加载数据集
_, _, test_dataset = load_and_align_data(high_dim_path = config['data']['high_dim_path'], 
                                                  low_dim_path = config['data']['low_dim_path'],
                                                  labels_pathconfig = config['data']['labels_path'],
                                                  test_size = config['data']['test_size'],
                                                  val_size = config['data']['val_size'],
                                                  random_state = config['data']['random_state'])

# 创建数据加载器
test_loader = create_data_loader(test_dataset, batch_size=config['data']['batch_size'], shuffle=config['data']['shuffle'])

# 初始化模型并加载权重
model = GATModel(high_dim_input_size=config["model"]["high_dim_input_size"],  # 适当调整这些参数，这里可以写成config
                 low_dim_input_size=config["model"]["low_dim_input_size"],
                 embedding_dim=config["model"]["embedding_dim"],
                 output_dim=config["model"]["output_dim"],  # 根据您的任务调整
                 hidden_channels=config["model"]["hidden_channels"],
                 num_heads=config["model"]["num_heads"])

model.load_state_dict(torch.load('path/to/model.pth'))
model.eval()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 测试
correct = 0
total = 0
with torch.no_grad():
    for high_dim_features, low_dim_features, labels, edge_index in test_loader:
        high_dim_features = high_dim_features.to(device)
        low_dim_features = low_dim_features.to(device)
        labels = labels.to(device)
        edge_index = edge_index.to(device)

        outputs = model(high_dim_features, low_dim_features, edge_index)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test dataset: {100 * correct / total}%')
