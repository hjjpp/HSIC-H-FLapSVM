#!/usr/bin/env python3
"""
统一数据加载模块
加载WM特征.xlsx中的各类特征和标签数据
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_wm_data(excel_file='WM特征.xlsx', test_size=0.2, random_state=42):
    """
    加载WM特征数据和标签

    Returns:
        features_dict: 字典包含各类特征数据
        X_combined: 合并的特征矩阵
        y: 标签数组
        X_train, X_test, y_train, y_test: 训练测试分割
    """

    print("Loading WM features data...")

    # 读取Excel文件的所有sheet
    excel_data = pd.read_excel(excel_file, sheet_name=None)
    print(f"Available sheets: {list(excel_data.keys())}")

    # 加载各类特征
    features_dict = {}
    feature_names_mapping = {
        'AATP_Features': 'AATP',
        'GE_Features': 'GE',
        'NMBAC_Features': 'NMBAC',
        'PSSM_AB_Features': 'PSSM_AB',
        'PSSM_DWT_Features': 'PSSM_DWT',
        'PSSM_PSE_Features': 'PSSM_PSE'
    }

    for sheet_name, feature_name in feature_names_mapping.items():
        if sheet_name in excel_data:
            df = excel_data[sheet_name]
            # 去除可能的索引列
            if 'Row' in df.columns:
                feature_data = df.drop('Row', axis=1).values
            else:
                feature_data = df.values

            features_dict[feature_name] = feature_data
            print(f"Loaded {feature_name}: {feature_data.shape}")
        else:
            print(f"Warning: {sheet_name} not found in Excel file")

    # 加载标签
    if 'Labels' in excel_data:
        labels_df = excel_data['Labels']
        # 标签可能在第二列或第一列
        if labels_df.shape[1] >= 2:
            y = labels_df.iloc[:, 1].values
        else:
            y = labels_df.iloc[:, 0].values

        print(f"Loaded labels: {y.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Positive ratio: {np.mean(y)*100:.1f}%")
    else:
        raise ValueError("Labels sheet not found in Excel file")

    # 合并所有特征（按文档顺序）
    feature_order = ['GE', 'NMBAC', 'PSSM_PSE', 'PSSM_AB', 'PSSM_DWT', 'AATP']
    combined_features = []

    for feature_name in feature_order:
        if feature_name in features_dict:
            combined_features.append(features_dict[feature_name])

    X_combined = np.hstack(combined_features)
    print(f"Combined features shape: {X_combined.shape}")

    # 训练测试分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Train labels: {np.bincount(y_train)}, Test labels: {np.bincount(y_test)}")

    return features_dict, X_combined, y, X_train, X_test, y_train, y_test

def get_feature_subset(features_dict, feature_names):
    """
    获取指定特征的子集

    Args:
        features_dict: 特征字典
        feature_names: 特征名称列表，如['PSSM_PSE', 'PSSM_DWT']

    Returns:
        X_subset: 合并的特征子集
    """

    feature_subsets = []
    for feature_name in feature_names:
        if feature_name in features_dict:
            feature_subsets.append(features_dict[feature_name])
        else:
            print(f"Warning: Feature {feature_name} not found")

    if feature_subsets:
        X_subset = np.hstack(feature_subsets)
        print(f"Feature subset {'+'.join(feature_names)}: {X_subset.shape}")
        return X_subset
    else:
        raise ValueError(f"No valid features found in {feature_names}")

if __name__ == "__main__":
    # 测试数据加载
    features_dict, X_combined, y, X_train, X_test, y_train, y_test = load_wm_data()

    # 测试特征子集
    X_subset = get_feature_subset(features_dict, ['PSSM_PSE', 'PSSM_DWT'])
    print(f"Test subset shape: {X_subset.shape}")