# src/training.py - 模型训练模块

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    训练光伏故障检测模型
    
    参数:
    model (nn.Module): 待训练的模型
    train_loader (DataLoader): 训练数据加载器
    test_loader (DataLoader): 测试数据加载器
    num_epochs (int): 训练轮数
    learning_rate (float): 学习率
    device (str): 训练设备
    
    返回:
    nn.Module: 训练好的模型
    dict: 训练历史记录
    """
    # 将模型移至指定设备
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'test_f1': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 计算训练损失和准确率
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 在测试集上评估
        test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion, device)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test F1: {test_f1:.4f}')
    
    # 可视化训练结果
    plot_training_history(history)
    
    return model, history

def evaluate_model(model, data_loader, criterion, device='cpu'):
    """
    在给定数据集上评估模型性能
    
    参数:
    model (nn.Module): 待评估的模型
    data_loader (DataLoader): 数据加载器
    criterion (nn.Module): 损失函数
    device (str): 设备
    
    返回:
    tuple: (平均损失, 准确率, F1分数)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 收集所有标签和预测结果用于计算F1分数
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # 计算F1分数
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 打印混淆矩阵和分类报告
    if total > 0:  # 确保有数据
        print("\n混淆矩阵:")
        print(confusion_matrix(all_labels, all_preds))
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, target_names=['正常', '故障']))
        
        # 绘制ROC曲线
        if len(np.unique(all_labels)) == 2:  # 仅在二分类时绘制
            plot_roc_curve(all_labels, get_probabilities(model, data_loader, device))
    
    return running_loss / len(data_loader), 100.0 * correct / total, f1

def get_probabilities(model, data_loader, device='cpu'):
    """获取模型预测的概率值"""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类的概率
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)

def plot_training_history(history):
    """
    可视化训练历史
    
    参数:
    history (dict): 包含训练历史的字典
    """
    plt.figure(figsize=(14, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率和F1分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.plot(history['test_f1'], label='Test F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value (%)')
    plt.title('Training and Test Metrics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_roc_curve(labels, probs):
    """
    绘制ROC曲线
    
    参数:
    labels (array): 真实标签
    probs (array): 预测概率
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()