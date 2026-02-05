#!/usr/bin/env python3
"""
Advanced Classifiers Module
Implements H-FLapSVM, FLapSVM, LapSVM classifiers
For Stage 5: Advanced Classifier Performance Comparison
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import inv, eigh, pinv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

class LaplacianSVM(BaseEstimator, ClassifierMixin):
    """Laplacian Support Vector Machine (LapSVM)"""

    def __init__(self, C=1.0, gamma_A=1.0, gamma_I=1.0, kernel='rbf',
                 kernel_params=None, n_neighbors=5, random_state=42):
        """
        Initialize LapSVM

        Args:
            C: Regularization parameter for labeled data
            gamma_A: Regularization parameter for Laplacian term
            gamma_I: Regularization parameter for intrinsic regularization
            kernel: Kernel type
            kernel_params: Kernel parameters
            n_neighbors: Number of neighbors for Laplacian construction
            random_state: Random state
        """
        self.C = C
        self.gamma_A = gamma_A
        self.gamma_I = gamma_I
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params else {'gamma': 'scale'}
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.alpha_ = None
        self.support_vectors_ = None
        self.support_labels_ = None
        self.b_ = 0

    def _compute_kernel_matrix(self, X, Y=None):
        """Compute kernel matrix"""
        if self.kernel == 'rbf':
            gamma = self.kernel_params.get('gamma', 1.0)
            if gamma == 'scale':
                gamma = 1.0 / X.shape[1]
            return rbf_kernel(X, Y, gamma=gamma)
        else:
            raise ValueError(f"Kernel {self.kernel} not supported")

    def _compute_laplacian_matrix(self, X):
        """Compute Laplacian matrix from data"""
        # Build k-NN graph
        A = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                           mode='connectivity', include_self=False)
        A = A.toarray()

        # Make symmetric
        A = (A + A.T) / 2

        # Compute degree matrix
        D = np.diag(np.sum(A, axis=1))

        # Laplacian matrix
        L = D - A

        return L

    def fit(self, X, y):
        """Fit LapSVM"""
        n_samples = X.shape[0]

        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)

        # Compute Laplacian matrix
        L = self._compute_laplacian_matrix(X)

        # Convert labels to {-1, 1}
        y_transformed = 2 * y - 1

        # Solve the optimization problem
        # (K + gamma_A/gamma_I * L + I/gamma_I) * alpha = y

        I = np.eye(n_samples)
        A_matrix = K + (self.gamma_A / self.gamma_I) * L + I / self.gamma_I

        try:
            # Solve linear system
            self.alpha_ = np.linalg.solve(A_matrix, y_transformed)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self.alpha_ = pinv(A_matrix) @ y_transformed

        # Store support vectors (all training samples in LapSVM)
        self.support_vectors_ = X
        self.support_labels_ = y

        return self

    def decision_function(self, X):
        """Compute decision function"""
        K_test = self._compute_kernel_matrix(X, self.support_vectors_)
        return K_test @ self.alpha_

    def predict(self, X):
        """Predict class labels"""
        decision = self.decision_function(X)
        return (decision > 0).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities using sigmoid"""
        decision = self.decision_function(X)
        prob_positive = 1 / (1 + np.exp(-decision))
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])

class FuzzyLaplacianSVM(LaplacianSVM):
    """Fuzzy Laplacian Support Vector Machine (FLapSVM)"""

    def __init__(self, C=1.0, gamma_A=1.0, gamma_I=1.0, kernel='rbf',
                 kernel_params=None, n_neighbors=5, fuzzy_m=2.0, random_state=42):
        """
        Initialize FLapSVM

        Args:
            C: Regularization parameter
            gamma_A: Laplacian regularization
            gamma_I: Intrinsic regularization
            kernel: Kernel type
            kernel_params: Kernel parameters
            n_neighbors: Neighbors for Laplacian
            fuzzy_m: Fuzzy membership parameter
            random_state: Random state
        """
        super().__init__(C, gamma_A, gamma_I, kernel, kernel_params, n_neighbors, random_state)
        self.fuzzy_m = fuzzy_m
        self.membership_weights_ = None

    def _compute_fuzzy_membership(self, X, y):
        """Compute fuzzy membership weights"""
        n_samples = X.shape[0]

        # Simple fuzzy membership based on local class distribution
        weights = np.ones(n_samples)

        # For each sample, compute local class purity
        for i in range(n_samples):
            # Find k nearest neighbors
            distances = np.linalg.norm(X - X[i], axis=1)
            neighbor_indices = np.argsort(distances)[1:self.n_neighbors+1]  # Exclude self

            # Compute class purity in neighborhood
            neighbor_labels = y[neighbor_indices]
            same_class_count = np.sum(neighbor_labels == y[i])
            purity = same_class_count / len(neighbor_labels)

            # Higher purity means higher membership
            weights[i] = purity ** (1 / self.fuzzy_m)

        # Normalize weights
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-10)
        weights = np.maximum(weights, 0.1)  # Minimum weight

        return weights

    def fit(self, X, y, sample_weights=None):
        """Fit FLapSVM with fuzzy membership"""
        n_samples = X.shape[0]

        # Compute or use provided membership weights
        if sample_weights is not None:
            self.membership_weights_ = sample_weights
        else:
            self.membership_weights_ = self._compute_fuzzy_membership(X, y)

        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)

        # Compute Laplacian matrix
        L = self._compute_laplacian_matrix(X)

        # Convert labels to {-1, 1}
        y_transformed = 2 * y - 1

        # Weight the system by fuzzy membership
        W = np.diag(self.membership_weights_)

        # Solve weighted optimization problem
        # (K + gamma_A/gamma_I * L + I/gamma_I) * alpha = W * y

        I = np.eye(n_samples)
        A_matrix = K + (self.gamma_A / self.gamma_I) * L + I / self.gamma_I
        b_vector = W @ y_transformed

        try:
            self.alpha_ = np.linalg.solve(A_matrix, b_vector)
        except np.linalg.LinAlgError:
            self.alpha_ = pinv(A_matrix) @ b_vector

        self.support_vectors_ = X
        self.support_labels_ = y

        return self

class HyperLaplacianFuzzySVM(FuzzyLaplacianSVM):
    """Hyper-Laplacian Fuzzy Support Vector Machine (H-FLapSVM)"""

    def __init__(self, C=1.0, gamma_A=1.0, gamma_I=1.0, gamma_H=1.0,
                 kernel='rbf', kernel_params=None, n_neighbors=5,
                 fuzzy_m=2.0, hyper_p=0.5, random_state=42):
        """
        Initialize H-FLapSVM

        Args:
            C: Regularization parameter
            gamma_A: Laplacian regularization
            gamma_I: Intrinsic regularization
            gamma_H: Hyper-Laplacian regularization
            kernel: Kernel type
            kernel_params: Kernel parameters
            n_neighbors: Neighbors for Laplacian
            fuzzy_m: Fuzzy membership parameter
            hyper_p: Hyper-Laplacian power parameter (0 < p < 1)
            random_state: Random state
        """
        super().__init__(C, gamma_A, gamma_I, kernel, kernel_params,
                        n_neighbors, fuzzy_m, random_state)
        self.gamma_H = gamma_H
        self.hyper_p = hyper_p

    def _compute_hyper_laplacian_matrix(self, X):
        """Compute Hyper-Laplacian matrix"""
        # Build similarity matrix with weights
        n_samples = X.shape[0]
        W = np.zeros((n_samples, n_samples))

        # Compute pairwise distances
        for i in range(n_samples):
            distances = np.linalg.norm(X - X[i], axis=1)
            # Find k nearest neighbors
            neighbor_indices = np.argsort(distances)[1:self.n_neighbors+1]

            # Compute similarity weights using RBF kernel
            gamma = self.kernel_params.get('gamma', 1.0)
            if gamma == 'scale':
                gamma = 1.0 / X.shape[1]

            for j in neighbor_indices:
                weight = np.exp(-gamma * distances[j]**2)
                W[i, j] = weight
                W[j, i] = weight  # Symmetric

        # Compute degree matrix
        D = np.diag(np.sum(W, axis=1))

        # Standard Laplacian
        L = D - W

        # Hyper-Laplacian: L^p where 0 < p < 1
        # Use eigendecomposition for fractional power
        eigenvals, eigenvecs = eigh(L)

        # Compute L^p
        eigenvals_p = np.maximum(eigenvals, 1e-10) ** self.hyper_p
        L_hyper = eigenvecs @ np.diag(eigenvals_p) @ eigenvecs.T

        return L_hyper

    def fit(self, X, y, sample_weights=None):
        """Fit H-FLapSVM with hyper-Laplacian regularization"""
        n_samples = X.shape[0]

        # Compute fuzzy membership weights
        if sample_weights is not None:
            self.membership_weights_ = sample_weights
        else:
            self.membership_weights_ = self._compute_fuzzy_membership(X, y)

        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)

        # Compute standard Laplacian matrix
        L = self._compute_laplacian_matrix(X)

        # Compute Hyper-Laplacian matrix
        L_hyper = self._compute_hyper_laplacian_matrix(X)

        # Convert labels to {-1, 1}
        y_transformed = 2 * y - 1

        # Weight by fuzzy membership
        W = np.diag(self.membership_weights_)

        # Solve optimization with both Laplacian and Hyper-Laplacian terms
        # (K + gamma_A/gamma_I * L + gamma_H/gamma_I * L_hyper + I/gamma_I) * alpha = W * y

        I = np.eye(n_samples)
        A_matrix = (K +
                   (self.gamma_A / self.gamma_I) * L +
                   (self.gamma_H / self.gamma_I) * L_hyper +
                   I / self.gamma_I)
        b_vector = W @ y_transformed

        try:
            self.alpha_ = np.linalg.solve(A_matrix, b_vector)
        except np.linalg.LinAlgError:
            self.alpha_ = pinv(A_matrix) @ b_vector

        self.support_vectors_ = X
        self.support_labels_ = y

        return self

def evaluate_advanced_classifier(classifier, X_train, y_train, X_test, y_test,
                                classifier_name, sample_weights=None):
    """Evaluate an advanced classifier"""
    print(f"\n--- Evaluating {classifier_name} ---")

    try:
        # Data preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit classifier
        if hasattr(classifier, 'fit') and 'sample_weights' in classifier.fit.__code__.co_varnames:
            classifier.fit(X_train_scaled, y_train, sample_weights=sample_weights)
        else:
            classifier.fit(X_train_scaled, y_train)

        # Predict
        y_pred = classifier.predict(X_test_scaled)

        if hasattr(classifier, 'predict_proba'):
            y_prob = classifier.predict_proba(X_test_scaled)[:, 1]
        else:
            # Use decision function if available
            if hasattr(classifier, 'decision_function'):
                decision = classifier.decision_function(X_test_scaled)
                y_prob = 1 / (1 + np.exp(-decision))  # Sigmoid
            else:
                y_prob = y_pred.astype(float)  # Fallback

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate SN and SP
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0

        results = {
            'Method': classifier_name,
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

def test_advanced_classifiers():
    """Test all advanced classifiers"""
    print("Testing Advanced Classifiers Implementation")
    print("="*60)

    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples)

    # Split data
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    # Generate sample fuzzy weights
    sample_weights = np.random.uniform(0.1, 1.0, len(X_train))

    # Test each classifier
    classifiers = {
        'SVM (Baseline)': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
        'LapSVM': LaplacianSVM(C=1.0, gamma_A=1.0, gamma_I=1.0, random_state=42),
        'FLapSVM': FuzzyLaplacianSVM(C=1.0, gamma_A=1.0, gamma_I=1.0, fuzzy_m=2.0, random_state=42),
        'H-FLapSVM': HyperLaplacianFuzzySVM(C=1.0, gamma_A=1.0, gamma_I=1.0,
                                           gamma_H=1.0, fuzzy_m=2.0, hyper_p=0.5, random_state=42)
    }

    results = []
    for name, classifier in classifiers.items():
        # Use sample weights for fuzzy methods
        weights = sample_weights if 'Fuzzy' in name or 'H-FLap' in name else None
        result = evaluate_advanced_classifier(classifier, X_train, y_train, X_test, y_test, name, weights)
        if result:
            results.append(result)

    if results:
        print(f"\n{'='*60}")
        print("Advanced Classifiers Test Results")
        print(f"{'='*60}")

        df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_true', 'y_pred', 'y_prob']} for r in results])
        display_cols = ['Method', 'AUC', 'ACC (%)', 'MCC', 'SN', 'SP']
        print(df_results[display_cols].to_string(index=False))

    print("\nAdvanced classifiers implementation test completed!")

if __name__ == "__main__":
    test_advanced_classifiers()