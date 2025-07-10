import pandas as pd
import numpy as np

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 生成正常数据（1000条记录，6个传感器特征）
normal_data = {
    'voltage': np.random.normal(220, 10, 1000),           # 电压 (V)
    'current': np.random.normal(5, 0.5, 1000),            # 电流 (A)
    'temperature': np.random.normal(25, 3, 1000),         # 温度 (°C)
    'humidity': np.random.normal(40, 10, 1000),           # 湿度 (%)
    'light_intensity': np.random.normal(800, 100, 1000),  # 光照强度 (W/m²)
    'power': np.random.normal(1100, 100, 1000),          # 功率 (W)
    'fault': np.zeros(1000)                                # 故障标签 (0=正常)
}

# 生成故障数据（500条记录，6个传感器特征，部分特征有异常值）
fault_data = {
    'voltage': np.random.normal(180, 20, 500),            # 电压异常降低
    'current': np.random.normal(3, 0.8, 500),             # 电流异常降低
    'temperature': np.random.normal(60, 10, 500),         # 温度异常升高
    'humidity': np.random.normal(40, 10, 500),            # 湿度正常
    'light_intensity': np.random.normal(800, 100, 500),  # 光照强度正常
    'power': np.random.normal(800, 150, 500),            # 功率异常降低
    'fault': np.ones(500)                                  # 故障标签 (1=故障)
}

# 合并正常和故障数据
df = pd.concat([pd.DataFrame(normal_data), pd.DataFrame(fault_data)], ignore_index=True)

# 打乱数据顺序
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存为CSV文件
df.to_csv('sensor_data.csv', index=False)

print(f"数据已生成并保存至 'sensor_data.csv'")
print(f"数据集包含 {len(df)} 条记录，其中 {int(df['fault'].sum())} 条故障记录")