#!/usr/bin/env python3
"""
Multiple Kernel Learning (MKL) Methods Module
Implements various MKL algorithms: MKL-HSIC, MKL-CKA, MKL-MW, MKL-HKA
For Stage 3: Multi-kernel Learning Fusion Experiments
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from scipy.optimize import minimize
from scipy.linalg import eigh, inv, norm
import warnings
warnings.filterwarnings('ignore')

class MultipleKernelLearning:
    """Base class for Multiple Kernel Learning methods"""

    def __init__(self, kernel_types=['rbf', 'poly', 'linear'], kernel_params=None):
        """
        Initialize MKL base class

        Args:
            kernel_types: List of kernel types to use
            kernel_params: Dictionary of kernel parameters
        """
        self.kernel_types = kernel_types
        self.kernel_params = kernel_params if kernel_params else {
            'rbf': {'gamma': 'scale'},
            'poly': {'degree': 3, 'gamma': 'scale', 'coef0': 1},
            'linear': {}
        }
        self.kernel_weights = None
        self.kernels = []

    def compute_kernel_matrix(self, X, Y=None, kernel_type='rbf', **params):
        """Compute kernel matrix for given kernel type"""
        if kernel_type == 'rbf':
            gamma = params.get('gamma', 1.0)
            if gamma == 'scale':
                gamma = 1.0 / X.shape[1]
            return rbf_kernel(X, Y, gamma=gamma)
        elif kernel_type == 'poly':
            degree = params.get('degree', 3)
            gamma = params.get('gamma', 1.0)
            coef0 = params.get('coef0', 1)
            if gamma == 'scale':
                gamma = 1.0 / X.shape[1]
            return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
        elif kernel_type == 'linear':
            return linear_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def compute_all_kernels(self, X, Y=None):
        """Compute all kernel matrices"""
        kernels = []
        for kernel_type in self.kernel_types:
            params = self.kernel_params.get(kernel_type, {})
            K = self.compute_kernel_matrix(X, Y, kernel_type, **params)
            kernels.append(K)
        return kernels

    def combine_kernels(self, kernels, weights=None):
        """Combine multiple kernels with given weights"""
        if weights is None:
            weights = np.ones(len(kernels)) / len(kernels)

        combined_kernel = np.zeros_like(kernels[0])
        for i, kernel in enumerate(kernels):
            combined_kernel += weights[i] * kernel

        return combined_kernel

class MKL_HSIC(MultipleKernelLearning):
    """MKL with Hilbert-Schmidt Independence Criterion"""

    def __init__(self, kernel_types=['rbf', 'poly', 'linear'], kernel_params=None,
                 reg_param=1e-6):
        """
        Initialize MKL-HSIC

        Args:
            kernel_types: List of kernel types
            kernel_params: Kernel parameters
            reg_param: Regularization parameter
        """
        super().__init__(kernel_types, kernel_params)
        self.reg_param = reg_param

    def center_kernel(self, K):
        """Center kernel matrix"""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def compute_hsic(self, K_x, K_y):
        """Compute HSIC between two kernel matrices"""
        n = K_x.shape[0]

        # Center kernels
        K_x_c = self.center_kernel(K_x)
        K_y_c = self.center_kernel(K_y)

        # Compute HSIC
        hsic = np.trace(K_x_c @ K_y_c) / ((n - 1) ** 2)

        return hsic

    def compute_target_kernel(self, y):
        """Compute target kernel matrix from labels"""
        n = len(y)
        K_y = np.outer(y, y)  # Linear kernel on labels
        return K_y

    def learn_kernel_weights(self, X, y):
        """Learn kernel weights using HSIC criterion"""
        # Compute all base kernels
        base_kernels = self.compute_all_kernels(X)

        # Compute target kernel
        K_target = self.compute_target_kernel(y)

        # Compute HSIC values for each kernel
        hsic_values = []
        for K_base in base_kernels:
            hsic = self.compute_hsic(K_base, K_target)
            hsic_values.append(hsic)

        hsic_values = np.array(hsic_values)

        # Normalize to get weights
        if np.sum(hsic_values) > 0:
            weights = hsic_values / np.sum(hsic_values)
        else:
            weights = np.ones(len(base_kernels)) / len(base_kernels)

        self.kernel_weights = weights
        self.kernels = base_kernels

        return weights

    def fit_predict(self, X_train, y_train, X_test):
        """Fit MKL-HSIC and predict"""
        # Learn kernel weights
        weights = self.learn_kernel_weights(X_train, y_train)

        # Compute combined kernel for training
        K_train = self.combine_kernels(self.kernels, weights)

        # Compute kernels for test set
        test_kernels = []
        for i, kernel_type in enumerate(self.kernel_types):
            params = self.kernel_params.get(kernel_type, {})
            K_test = self.compute_kernel_matrix(X_test, X_train, kernel_type, **params)
            test_kernels.append(K_test)

        # Combine test kernels
        K_test = self.combine_kernels(test_kernels, weights)

        return K_train, K_test, weights

class MKL_CKA(MultipleKernelLearning):
    """MKL with Centered Kernel Alignment"""

    def __init__(self, kernel_types=['rbf', 'poly', 'linear'], kernel_params=None):
        super().__init__(kernel_types, kernel_params)

    def center_kernel(self, K):
        """Center kernel matrix"""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    def compute_cka(self, K_x, K_y):
        """Compute Centered Kernel Alignment"""
        # Center kernels
        K_x_c = self.center_kernel(K_x)
        K_y_c = self.center_kernel(K_y)

        # Compute CKA
        numerator = np.trace(K_x_c @ K_y_c)
        denominator = np.sqrt(np.trace(K_x_c @ K_x_c) * np.trace(K_y_c @ K_y_c))

        if denominator > 0:
            cka = numerator / denominator
        else:
            cka = 0

        return cka

    def learn_kernel_weights(self, X, y):
        """Learn kernel weights using CKA"""
        # Compute all base kernels
        base_kernels = self.compute_all_kernels(X)

        # Compute target kernel
        K_target = np.outer(y, y)

        # Compute CKA values
        cka_values = []
        for K_base in base_kernels:
            cka = self.compute_cka(K_base, K_target)
            cka_values.append(cka)

        cka_values = np.array(cka_values)

        # Normalize to get weights (use absolute values)
        abs_cka = np.abs(cka_values)
        if np.sum(abs_cka) > 0:
            weights = abs_cka / np.sum(abs_cka)
        else:
            weights = np.ones(len(base_kernels)) / len(base_kernels)

        self.kernel_weights = weights
        self.kernels = base_kernels

        return weights

    def fit_predict(self, X_train, y_train, X_test):
        """Fit MKL-CKA and predict"""
        weights = self.learn_kernel_weights(X_train, y_train)

        # Compute combined training kernel
        K_train = self.combine_kernels(self.kernels, weights)

        # Compute test kernels
        test_kernels = []
        for i, kernel_type in enumerate(self.kernel_types):
            params = self.kernel_params.get(kernel_type, {})
            K_test = self.compute_kernel_matrix(X_test, X_train, kernel_type, **params)
            test_kernels.append(K_test)

        K_test = self.combine_kernels(test_kernels, weights)

        return K_train, K_test, weights

class MKL_MW(MultipleKernelLearning):
    """MKL with Multiple Weights (Simple averaging with performance weighting)"""

    def __init__(self, kernel_types=['rbf', 'poly', 'linear'], kernel_params=None):
        super().__init__(kernel_types, kernel_params)

    def evaluate_single_kernel(self, K, y, cv_folds=3):
        """Evaluate single kernel performance using cross-validation"""
        n = len(y)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv.split(np.arange(n), y):
            K_train = K[np.ix_(train_idx, train_idx)]
            K_val = K[np.ix_(val_idx, train_idx)]

            y_train, y_val = y[train_idx], y[val_idx]

            try:
                # Use kernel SVM
                svm = SVC(kernel='precomputed', C=1.0, probability=True, random_state=42)
                svm.fit(K_train, y_train)
                y_pred_prob = svm.predict_proba(K_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred_prob)
                scores.append(auc)
            except:
                scores.append(0.5)  # Random performance

        return np.mean(scores)

    def learn_kernel_weights(self, X, y):
        """Learn weights based on individual kernel performance"""
        # Compute all base kernels
        base_kernels = self.compute_all_kernels(X)

        # Evaluate each kernel
        performances = []
        for K in base_kernels:
            perf = self.evaluate_single_kernel(K, y)
            performances.append(perf)

        performances = np.array(performances)

        # Convert to weights (softmax-like)
        if np.max(performances) > np.min(performances):
            exp_perf = np.exp(performances - np.max(performances))
            weights = exp_perf / np.sum(exp_perf)
        else:
            weights = np.ones(len(base_kernels)) / len(base_kernels)

        self.kernel_weights = weights
        self.kernels = base_kernels

        return weights

    def fit_predict(self, X_train, y_train, X_test):
        """Fit MKL-MW and predict"""
        weights = self.learn_kernel_weights(X_train, y_train)

        K_train = self.combine_kernels(self.kernels, weights)

        # Compute test kernels
        test_kernels = []
        for i, kernel_type in enumerate(self.kernel_types):
            params = self.kernel_params.get(kernel_type, {})
            K_test = self.compute_kernel_matrix(X_test, X_train, kernel_type, **params)
            test_kernels.append(K_test)

        K_test = self.combine_kernels(test_kernels, weights)

        return K_train, K_test, weights

class MKL_HKA(MultipleKernelLearning):
    """MKL with Hilbert-Schmidt Kernel Alignment (variant of CKA)"""

    def __init__(self, kernel_types=['rbf', 'poly', 'linear'], kernel_params=None):
        super().__init__(kernel_types, kernel_params)

    def compute_hka(self, K_x, K_y):
        """Compute Hilbert-Schmidt Kernel Alignment"""
        # Frobenius inner product
        numerator = np.trace(K_x.T @ K_y)
        denominator = np.sqrt(np.trace(K_x.T @ K_x) * np.trace(K_y.T @ K_y))

        if denominator > 0:
            hka = numerator / denominator
        else:
            hka = 0

        return hka

    def learn_kernel_weights(self, X, y):
        """Learn kernel weights using HKA"""
        base_kernels = self.compute_all_kernels(X)

        # Create ideal kernel from labels
        K_ideal = np.outer(y, y)

        # Compute HKA values
        hka_values = []
        for K_base in base_kernels:
            hka = self.compute_hka(K_base, K_ideal)
            hka_values.append(hka)

        hka_values = np.array(hka_values)

        # Normalize to weights
        abs_hka = np.abs(hka_values)
        if np.sum(abs_hka) > 0:
            weights = abs_hka / np.sum(abs_hka)
        else:
            weights = np.ones(len(base_kernels)) / len(base_kernels)

        self.kernel_weights = weights
        self.kernels = base_kernels

        return weights

    def fit_predict(self, X_train, y_train, X_test):
        """Fit MKL-HKA and predict"""
        weights = self.learn_kernel_weights(X_train, y_train)

        K_train = self.combine_kernels(self.kernels, weights)

        test_kernels = []
        for i, kernel_type in enumerate(self.kernel_types):
            params = self.kernel_params.get(kernel_type, {})
            K_test = self.compute_kernel_matrix(X_test, X_train, kernel_type, **params)
            test_kernels.append(K_test)

        K_test = self.combine_kernels(test_kernels, weights)

        return K_train, K_test, weights

class KernelSVM:
    """Kernel SVM wrapper for precomputed kernels"""

    def __init__(self, C=1.0, random_state=42):
        self.C = C
        self.random_state = random_state
        self.svm = None

    def fit(self, K_train, y_train):
        """Fit SVM with precomputed kernel"""
        self.svm = SVC(kernel='precomputed', C=self.C, probability=True,
                      random_state=self.random_state)
        self.svm.fit(K_train, y_train)
        return self

    def predict(self, K_test):
        """Predict with precomputed kernel"""
        return self.svm.predict(K_test)

    def predict_proba(self, K_test):
        """Predict probabilities with precomputed kernel"""
        return self.svm.predict_proba(K_test)

def evaluate_mkl_method(mkl_method, X_train, y_train, X_test, y_test, method_name):
    """Evaluate a single MKL method"""
    print(f"\n--- Evaluating {method_name} ---")

    try:
        # Get kernels and weights
        K_train, K_test, weights = mkl_method.fit_predict(X_train, y_train, X_test)

        print(f"  Kernel weights: {weights}")

        # Train SVM on combined kernel
        ksvm = KernelSVM(C=1.0, random_state=42)
        ksvm.fit(K_train, y_train)

        # Predict on test set
        y_pred = ksvm.predict(K_test)
        y_prob = ksvm.predict_proba(K_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate SN and SP
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

        results = {
            'Method': method_name,
            'Kernel_Weights': weights.tolist(),
            'AUC': f"{auc:.4f}",
            'ACC (%)': f"{acc*100:.2f}",
            'MCC': f"{mcc:.4f}",
            'SN': f"{sn:.4f}",
            'SP': f"{sp:.4f}",
            'AUC_raw': auc,
            'ACC_raw': acc,
            'MCC_raw': mcc,
            'SN_raw': sn,
            'SP_raw': sp,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        print(f"  AUC: {results['AUC']}")
        print(f"  ACC: {results['ACC (%)']}%")
        print(f"  MCC: {results['MCC']}")

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def test_mkl_methods():
    """Test all MKL methods with sample data"""
    print("Testing MKL Methods Implementation")
    print("="*50)

    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples)

    # Split data
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    # Test each method
    methods = {
        'MKL-HSIC': MKL_HSIC(),
        'MKL-CKA': MKL_CKA(),
        'MKL-MW': MKL_MW(),
        'MKL-HKA': MKL_HKA()
    }

    results = []
    for name, method in methods.items():
        result = evaluate_mkl_method(method, X_train, y_train, X_test, y_test, name)
        if result:
            results.append(result)

    if results:
        print(f"\n{'='*50}")
        print("MKL Methods Test Results")
        print(f"{'='*50}")

        df_results = pd.DataFrame(results)
        display_cols = ['Method', 'AUC', 'ACC (%)', 'MCC', 'SN', 'SP']
        print(df_results[display_cols].to_string(index=False))

    print("\nMKL methods implementation test completed!")

if __name__ == "__main__":
    test_mkl_methods()