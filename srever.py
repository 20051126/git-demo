from flask import Flask, request, jsonify
import torch
import numpy as np
from src.models import PVFaultDetectionModel
from src.inference import predict_fault
import joblib

app = Flask(__name__)

# 加载模型和标准化器
model_path = "models/pv_fault_model.pth"
scaler_path = "models/scaler.pkl"  # 需要保存标准化器

# 初始化模型
input_size = 10  # 假设输入特征数量：温度、湿度、光照、电压、电流、功率
model = PVFaultDetectionModel(input_size=input_size, hidden_size=64, num_classes=2)
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载标准化器
scaler = joblib.load(scaler_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取传感器数据
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # 提取特征并转换为模型期望的格式
        features = np.array([
            data['temperature'],
            data['humidity'],
            data['light_intensity'],
            data['voltage'],
            data['current'],
            data['power']
        ]).reshape(1, -1)
        
        # 应用标准化
        scaled_features = scaler.transform(features)
        
        # 进行预测
        predicted_class, confidence = predict_fault(model, scaled_features)
        
        # 构建响应
        result = {
            "fault_status": int(predicted_class),
            "fault_description": "故障" if predicted_class == 1 else "正常",
            "confidence": float(confidence),
            "timestamp": data.get('timestamp', None)
        }
        
        # 打印预测结果到终端
        print(f"预测结果: {result['fault_description']}, 置信度: {result['confidence']:.4f}")
        print(f"传感器数据: {features.flatten()}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)