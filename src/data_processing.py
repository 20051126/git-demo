# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    加载并预处理光伏传感器数据
    
    参数:
    file_path (str): 数据文件路径
    test_size (float): 测试集比例
    random_state (int): 随机种子
    
    返回:
    tuple: 包含训练集和测试集的元组 (X_train, X_test, y_train, y_test, scaler)
    """
    # 加载数据
    try:
        df = pd.read_csv(file_path)
        print(f"数据加载成功，共 {len(df)} 条记录")
    except Exception as e:
        print(f"数据加载失败: {e}")
        # 生成示例数据用于测试
        print("生成示例数据用于测试...")
        df = generate_sample_data(1000)
    
    # 数据探索
    print(f"数据基本信息:")
    df.info()
    
    # 假设最后一列为目标变量(故障/正常)，其余为特征
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler

def generate_sample_data(num_samples=1000, num_features=10):
    """
    生成示例光伏传感器数据用于测试
    
    参数:
    num_samples (int): 样本数量
    num_features (int): 特征数量
    
    返回:
    pd.DataFrame: 包含示例数据的DataFrame
    """
    # 生成正常数据
    normal_data = np.random.randn(num_samples, num_features) * 0.5 + 5
    
    # 生成故障数据 (某些特征值异常)
    fault_data = np.random.randn(num_samples, num_features) * 1.0 + 7
    fault_data[:, 2:5] = fault_data[:, 2:5] * 2  # 增强某些特征的异常
    
    # 创建标签
    normal_labels = np.zeros(num_samples)
    fault_labels = np.ones(num_samples)
    
    # 合并数据
    all_data = np.vstack([normal_data, fault_data])
    all_labels = np.concatenate([normal_labels, fault_labels])
    
    # 创建DataFrame
    columns = [f"feature_{i}" for i in range(num_features)] + ["fault"]
    df = pd.DataFrame(np.hstack([all_data, all_labels.reshape(-1, 1)]), columns=columns)
    
    return df