# src/models.py - 模型定义

import torch
import torch.nn as nn

class PVFaultDetectionModel(nn.Module):
    """光伏板故障检测模型 - 用于分类任务"""
    
    def __init__(self, input_size=10, hidden_size=64, num_classes=2):
        super(PVFaultDetectionModel, self).__init__()
        # 定义网络层
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size//2, num_classes)
        self.dropout = nn.Dropout(0.2)  # 添加dropout防止过拟合
    
    def forward(self, x):
        # 定义前向传播过程
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class PVAnomalyDetectionModel(nn.Module):
    """光伏板异常检测自编码器模型 - 用于异常检测"""
    
    def __init__(self, input_size=10, hidden_size=32):
        super(PVAnomalyDetectionModel, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU()
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # 假设输入特征已归一化到[0,1]
        )
    
    def forward(self, x):
        # 前向传播过程：编码然后解码
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 你可以在这里添加更多模型，如LSTM模型处理时序数据
class PVTemporalModel(nn.Module):
    """光伏板时序数据分析模型"""
    
    def __init__(self, input_size=10, hidden_size=64, num_layers=1, num_classes=2):
        super(PVTemporalModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层用于分类
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 假设输入x的形状是(batch_size, sequence_length, input_size)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 我们只需要最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out