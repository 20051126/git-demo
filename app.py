# app.py - 用于接收传感器数据并进行故障预测的Flask API

import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import joblib

# 导入模型类
from src.models import PVFaultDetectionModel

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 模型和标准化器初始化
model = None
scaler = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model_and_scaler():
    global model, scaler
    
    # 加载标准化器
    scaler_path = "models/scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("标准化器加载成功")
    else:
        print(f"标准化器文件 {scaler_path} 不存在，将使用默认标准化")
        scaler = StandardScaler()
    
    # 加载模型
    model_path = "models/pv_fault_model.pth"
    if os.path.exists(model_path):
        # 获取特征数量（这里假设为6，对应6个传感器）
        input_size = 6
        model = PVFaultDetectionModel(input_size=input_size, hidden_size=64, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("模型加载成功")
    else:
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练模型")

def preprocess_input(data):
    """对输入数据进行预处理，与训练时保持一致"""
    # 确保数据格式正确
    input_data = np.array([
        data['temperature'],
        data['humidity'],
        data['light_intensity'],
        data['voltage'],
        data['current'],
        data['power']
    ]).reshape(1, -1)
    
    # 应用标准化
    if scaler:
        input_data = scaler.transform(input_data)
    
    return input_data

def predict_fault(input_data):
    """使用模型进行故障预测"""
    if model is None:
        load_model_and_scaler()
    
    # 转换为张量
    input_tensor = torch.FloatTensor(input_data).to(device)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # 转换为Python类型
    predicted_class = predicted.item()
    confidence = confidence.item()
    
    return predicted_class, confidence

@app.route('/predict', methods=['POST'])
def receive_sensor_data():
    """接收传感器数据并返回故障预测结果"""
    try:
        # 获取JSON数据
        data = request.get_json()
        if not data:
            return jsonify({"error": "未接收到数据"}), 400
        
        # 预处理数据
        processed_data = preprocess_input(data)
        
        # 进行故障预测
        predicted_class, confidence = predict_fault(processed_data)
        
        # 构建响应
        response = {
            "prediction": "故障" if predicted_class == 1 else "正常",
            "confidence": confidence,
            "details": {
                "temperature": data['temperature'],
                "humidity": data['humidity'],
                "light_intensity": data['light_intensity'],
                "voltage": data['voltage'],
                "current": data['current'],
                "power": data['power']
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 加载模型和标准化器
    load_model_and_scaler()
    
    # 启动服务器
    app.run(host='0.0.0.0', port=5000, debug=True)