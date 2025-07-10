# src/inference.py
import torch
import numpy as np
from src.models import PVFaultDetectionModel

def predict_fault(model, input_data, device='cpu'):
    """
    使用训练好的模型进行故障预测
    
    参数:
    model (nn.Module): 训练好的模型
    input_data (torch.Tensor or np.ndarray): 输入数据
    device (str): 设备
    
    返回:
    int: 预测类别 (0=正常, 1=故障)
    float: 预测概率
    """
    model.to(device)
    model.eval()
    
    # 转换输入数据为Tensor
    if isinstance(input_data, np.ndarray):
        input_data = torch.FloatTensor(input_data)
    
    with torch.no_grad():
        input_data = input_data.to(device)
        output = model(input_data)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class, confidence

def load_model(model_path, input_size=10, device='cpu'):
    """
    从文件加载训练好的模型
    
    参数:
    model_path (str): 模型文件路径
    input_size (int): 输入特征大小
    device (str): 设备
    
    返回:
    nn.Module: 加载好的模型
    """
    # 初始化模型
    model = PVFaultDetectionModel(input_size=input_size)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    
    return model