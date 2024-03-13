# config.py

import json

config = {
  "data": {
    "high_dim_path": 'data/FINAL_MRI_DATA1.csv',
    "low_dim_path": 'data/FINAL_TABLEDATA_MRI.csv',
    "labels_path": 'data/Y_label_NEW_01.csv',
    "batch_size": 128,
    "shuffle": True,
    "test_size":0.1,
    "val_size":0.1,
    "random_state":12345
  },
  "model": {
    "type": "GCN",
    "high_dim_input_size": 498,  
    "low_dim_input_size":17,
    "embedding_dim":56,
    "output_dim":2,
    "hidden_channels":128,
    "num_heads":4 
  },
  "train": {
    "epochs": 250,
    "learning_rate": 0.01,
    "device": "cuda:1"
  },
  "earlystopping":{
    "patience":5,
    "delta":0.01
  }
}

# 将配置字典转换为JSON字符串
config_json = json.dumps(config, indent=4)