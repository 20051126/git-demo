# main.py - 主程序入口

import os
import torch
import pandas as pd
import joblib
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 导入自定义模块
from src.models import PVFaultDetectionModel
from src.training import train_model
from src.inference import predict_fault
from src.data_processing import load_and_preprocess_data

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

def main():
    # 数据路径
    data_path = "data/sensor_data.csv"
    
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在!")
        print("请确保已将传感器数据放置在正确的路径下。")
        return
    
    # 加载并预处理数据
    print("正在加载和预处理数据...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
    
    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    input_size = X_train.shape[1]
    model = PVFaultDetectionModel(input_size=input_size, hidden_size=64, num_classes=2)
    
    # 训练模型
    print("开始训练模型...")
    # 修正：接收train_model返回的两个值
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=10,
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 保存模型
    model_path = "models/pv_fault_model.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")
    scaler_path = "models/scaler.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存至: {scaler_path}")
    
    # 示例预测
    sample_idx = 0
    sample_input = X_test[sample_idx:sample_idx+1]  # 使用numpy数组
    predicted_class, confidence = predict_fault(trained_model, sample_input)
    
    print(f"\n示例预测:")
    print(f"输入特征: {sample_input.flatten()}")
    print(f"实际标签: {'故障' if y_test[sample_idx] == 1 else '正常'}")
    print(f"预测结果: {'故障' if predicted_class == 1 else '正常'}, 置信度: {confidence:.4f}")
    
    # 可视化训练结果
    # (如果train_model函数中已经包含可视化，可以省略)

if __name__ == "__main__":
    main()