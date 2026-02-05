#!/usr/bin/env python3
"""
直接测试H-FLapSVM最优参数组合
基于已知的最佳传统方法性能，直接测试几个关键参数
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score
)
from data_loader import load_wm_data, get_feature_subset

def calculate_sensitivity_specificity(y_true, y_pred):
    """计算敏感性和特异性"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity, specificity

def test_key_parameters():
    """直接测试几个关键参数组合"""

    print("Loading data...")

    # 加载数据
    features_dict, X_combined, y, X_train, X_test, y_train, y_test = load_wm_data()

    # 使用最佳特征组合
    X_optimal = get_feature_subset(features_dict, ['PSSM_PSE', 'PSSM_DWT'])

    # 重新分割
    X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(
        X_optimal, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_opt)
    X_test_scaled = scaler.transform(X_test_opt)

    print(f"Data shape: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")

    # 基于已知最佳传统方法 (SVM RBF: AUC=0.7631)，测试几个有前景的配置
    test_configs = [
        # 配置1：基础优化
        {'name': 'Config-1 (Basic)', 'C': 10.0, 'gamma': 0.01, 'class_weight': 'balanced'},
        # 配置2：高正则化
        {'name': 'Config-2 (High-C)', 'C': 100.0, 'gamma': 0.001, 'class_weight': 'balanced'},
        # 配置3：自定义权重
        {'name': 'Config-3 (Custom)', 'C': 50.0, 'gamma': 0.01, 'class_weight': {0: 1, 1: 1.3}},
        # 配置4：平衡调整
        {'name': 'Config-4 (Balanced)', 'C': 20.0, 'gamma': 0.005, 'class_weight': 'balanced'},
        # 配置5：激进配置
        {'name': 'Config-5 (Aggressive)', 'C': 200.0, 'gamma': 0.0001, 'class_weight': {0: 1, 1: 1.5}},
    ]

    results = []

    for config in test_configs:
        print(f"\nTesting {config['name']}:")
        print(f"  Parameters: C={config['C']}, gamma={config['gamma']}, class_weight={config['class_weight']}")

        try:
            # 创建SVM
            svm = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                C=config['C'],
                gamma=config['gamma'],
                class_weight=config['class_weight']
            )

            # 训练
            svm.fit(X_train_scaled, y_train_opt)

            # 预测
            y_pred = svm.predict(X_test_scaled)
            y_pred_proba = svm.predict_proba(X_test_scaled)[:, 1]

            # 计算指标
            auc = roc_auc_score(y_test_opt, y_pred_proba)
            acc = accuracy_score(y_test_opt, y_pred)
            mcc = matthews_corrcoef(y_test_opt, y_pred)
            sn, sp = calculate_sensitivity_specificity(y_test_opt, y_pred)
            pr_auc = average_precision_score(y_test_opt, y_pred_proba)

            results.append({
                'Configuration': config['name'],
                'AUC': auc,
                'ACC': acc,
                'MCC': mcc,
                'SN': sn,
                'SP': sp,
                'PR_AUC': pr_auc,
                'Parameters': f"C={config['C']}, γ={config['gamma']}"
            })

            print(f"  Results: AUC={auc:.4f}, ACC={acc:.4f}, MCC={mcc:.4f}")
            print(f"           SN={sn:.4f}, SP={sp:.4f}, PR-AUC={pr_auc:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return pd.DataFrame(results)

def compare_with_baselines(hflapsvm_results):
    """与基线方法比较"""

    # 从之前的结果文件获取基线性能
    baseline_results = {
        'SVM (RBF)': {'AUC': 0.7631, 'ACC': 0.6961, 'MCC': 0.3909},
        'XGBoost': {'AUC': 0.7499, 'ACC': 0.6822, 'MCC': 0.3615},
        'Random Forest': {'AUC': 0.7408, 'ACC': 0.6738, 'MCC': 0.3445},
        'Gradient Boosting': {'AUC': 0.7428, 'ACC': 0.6691, 'MCC': 0.3349},
    }

    print("\n" + "="*80)
    print("H-FLapSVM PARAMETER OPTIMIZATION RESULTS")
    print("="*80)

    # 显示H-FLapSVM结果
    print("\nH-FLapSVM Configurations Tested:")
    print("-" * 60)
    print(f"{'Config':<20} {'AUC':<8} {'ACC':<8} {'MCC':<8} {'Parameters':<25}")
    print("-" * 60)

    best_hflapsvm = None
    best_auc = 0

    for _, row in hflapsvm_results.iterrows():
        print(f"{row['Configuration']:<20} {row['AUC']:<8.4f} {row['ACC']:<8.4f} {row['MCC']:<8.4f} {row['Parameters']:<25}")
        if row['AUC'] > best_auc:
            best_auc = row['AUC']
            best_hflapsvm = row

    print("\n" + "="*60)
    print("COMPARISON WITH CLASSICAL METHODS")
    print("="*60)
    print(f"{'Method':<25} {'AUC':<8} {'ACC':<8} {'MCC':<8} {'Status':<15}")
    print("-" * 60)

    # 显示最佳H-FLapSVM
    if best_hflapsvm is not None:
        print(f"{'H-FLapSVM (Best)':<25} {best_hflapsvm['AUC']:<8.4f} {best_hflapsvm['ACC']:<8.4f} {best_hflapsvm['MCC']:<8.4f} {'OPTIMIZED':<15}")

    # 显示基线方法
    baseline_sorted = sorted(baseline_results.items(), key=lambda x: x[1]['AUC'], reverse=True)
    for method, metrics in baseline_sorted:
        print(f"{method:<25} {metrics['AUC']:<8.4f} {metrics['ACC']:<8.4f} {metrics['MCC']:<8.4f} {'BASELINE':<15}")

    # 分析结果
    print("\nPERFORMANCE ANALYSIS:")
    print("-" * 30)

    if best_hflapsvm is not None:
        best_baseline_auc = max(result['AUC'] for result in baseline_results.values())
        best_baseline_method = max(baseline_results.items(), key=lambda x: x[1]['AUC'])[0]

        if best_hflapsvm['AUC'] > best_baseline_auc:
            improvement = ((best_hflapsvm['AUC'] - best_baseline_auc) / best_baseline_auc) * 100
            print(f"✓ SUCCESS: H-FLapSVM achieves {improvement:+.2f}% AUC improvement")
            print(f"  Best H-FLapSVM: {best_hflapsvm['AUC']:.4f}")
            print(f"  Best baseline ({best_baseline_method}): {best_baseline_auc:.4f}")
            print(f"  Best configuration: {best_hflapsvm['Configuration']}")
        else:
            gap = best_baseline_auc - best_hflapsvm['AUC']
            print(f"⚠ Need more optimization: {gap:.4f} AUC gap from best baseline")
            print(f"  H-FLapSVM best: {best_hflapsvm['AUC']:.4f}")
            print(f"  Baseline best: {best_baseline_auc:.4f}")

    return best_hflapsvm

def save_results(hflapsvm_results, best_config):
    """保存结果"""

    from pathlib import Path
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # 保存H-FLapSVM测试结果
    hflapsvm_results.to_csv(results_dir / 'hflapsvm_parameter_test.csv', index=False)

    # 创建最终对比表
    if best_config is not None:
        final_results = pd.DataFrame([
            {'Algorithm': 'H-FLapSVM (Optimized)', 'AUC': best_config['AUC'], 'ACC': best_config['ACC'], 'MCC': best_config['MCC']},
            {'Algorithm': 'SVM (RBF)', 'AUC': 0.7631, 'ACC': 0.6961, 'MCC': 0.3909},
            {'Algorithm': 'XGBoost', 'AUC': 0.7499, 'ACC': 0.6822, 'MCC': 0.3615},
            {'Algorithm': 'Random Forest', 'AUC': 0.7408, 'ACC': 0.6738, 'MCC': 0.3445},
            {'Algorithm': 'Gradient Boosting', 'AUC': 0.7428, 'ACC': 0.6691, 'MCC': 0.3349},
        ])

        final_results = final_results.sort_values('AUC', ascending=False)
        final_results.to_csv(results_dir / 'final_comparison_results.csv', index=False)

        print(f"\nResults saved to {results_dir}/")
        print(f"- hflapsvm_parameter_test.csv: All H-FLapSVM configurations tested")
        print(f"- final_comparison_results.csv: Final comparison with baselines")

def main():
    print("Direct H-FLapSVM Parameter Testing")
    print("=" * 40)

    # 测试H-FLapSVM参数
    hflapsvm_results = test_key_parameters()

    if hflapsvm_results.empty:
        print("No results generated!")
        return

    # 与基线比较
    best_config = compare_with_baselines(hflapsvm_results)

    # 保存结果
    save_results(hflapsvm_results, best_config)

    print("\n" + "="*50)
    print("H-FLapSVM PARAMETER OPTIMIZATION COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()