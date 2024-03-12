# config.py

config = {
  "data": {
    "high_dim_path": "data/FINAL_MRI_DATA1.csv",
    "low_dim_path": "data/FINAL_TABLEDATA_MRI.csv",
    "labels_path": "data/Y_label.csv",
    "batch_size": 32,
    "shuffle": True,
    "test_size":0.2,
    "val_size":0.1,
    "random_state":42
  },
  "model": {
    "type": "GCN",
    "high_dim_input_size": 100,  
    "low_dim_input_size":10,
    "embedding_dim":64,
    "output_dim":2,
    "hidden_channels":128,
    "num_heads":8 
  },
  "train": {
    "epochs": 100,
    "learning_rate": 0.01,
    "device": "cuda:1"
  },
  "earlystopping":{
    "patience":5,
    "delta":0.001
  }
}