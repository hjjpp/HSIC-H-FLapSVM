#!/usr/bin/env python3
"""
Fuzzy Membership Methods Module
Implements various fuzzy membership methods: KECA, KNR, SR, SVDD, Optimal Kernels
For Stage 4: Fuzzy Membership Optimization Experiments
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import KernelPCA
from scipy.linalg import eigh, inv
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class FuzzyMembershipBase:
    """Base class for fuzzy membership methods"""

    def __init__(self, kernel_type='rbf', kernel_params=None):
        """
        Initialize fuzzy membership base class

        Args:
            kernel_type: Type of kernel to use
            kernel_params: Kernel parameters
        """
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params if kernel_params else {'gamma': 'scale'}
        self.membership_weights = None

    def compute_kernel_matrix(self, X, Y=None):
        """Compute kernel matrix"""
        if self.kernel_type == 'rbf':
            gamma = self.kernel_params.get('gamma', 1.0)
            if gamma == 'scale':
                gamma = 1.0 / X.shape[1]
            return rbf_kernel(X, Y, gamma=gamma)
        elif self.kernel_type == 'poly':
            degree = self.kernel_params.get('degree', 3)
            gamma = self.kernel_params.get('gamma', 1.0)
            coef0 = self.kernel_params.get('coef0', 1)
            if gamma == 'scale':
                gamma = 1.0 / X.shape[1]
            return polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def compute_membership_weights(self, X, y):
        """Compute fuzzy membership weights - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement compute_membership_weights")

class KECA_FuzzyMembership(FuzzyMembershipBase):
    """Kernel Entropy Component Analysis for Fuzzy Membership"""

    def __init__(self, kernel_type='rbf', kernel_params=None, n_components=None, sigma=1.0):
        """
        Initialize KECA fuzzy membership

        Args:
            kernel_type: Type of kernel
            kernel_params: Kernel parameters
            n_components: Number of components for KECA
            sigma: Bandwidth parameter for entropy estimation
        """
        super().__init__(kernel_type, kernel_params)
        self.n_components = n_components
        self.sigma = sigma

    def compute_kernel_entropy(self, K):
        """Compute kernel entropy using eigenvalues"""
        # Normalize kernel matrix
        n = K.shape[0]
        K_normalized = K / np.trace(K)

        # Compute eigenvalues
        eigenvals, _ = eigh(K_normalized)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues

        # Normalize eigenvalues to form probability distribution
        eigenvals = eigenvals / np.sum(eigenvals)

        # Compute entropy
        entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-10))

        return entropy

    def compute_membership_weights(self, X, y):
        """Compute KECA-based fuzzy membership weights"""
        print(f"Computing KECA fuzzy membership weights...")

        # Compute kernel matrix
        K = self.compute_kernel_matrix(X)
        n_samples = K.shape[0]

        # Center the kernel matrix
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        K_centered = H @ K @ H

        # Eigendecomposition for KECA
        eigenvals, eigenvecs = eigh(K_centered)
        eigenvals = eigenvals[::-1]  # Sort in descending order
        eigenvecs = eigenvecs[:, ::-1]

        # Select number of components
        if self.n_components is None:
            # Use explained variance ratio
            cumsum_eigenvals = np.cumsum(eigenvals[eigenvals > 0])
            total_variance = cumsum_eigenvals[-1]
            self.n_components = np.argmax(cumsum_eigenvals / total_variance >= 0.95) + 1
            self.n_components = min(self.n_components, n_samples // 2)

        print(f"  Using {self.n_components} components for KECA")

        # Project data to KECA space
        keca_features = eigenvecs[:, :self.n_components] @ np.diag(np.sqrt(np.maximum(eigenvals[:self.n_components], 0)))

        # Compute membership weights based on local density in KECA space
        weights = np.zeros(n_samples)

        for i in range(n_samples):
            # Compute distances to all other points in KECA space
            distances = np.linalg.norm(keca_features - keca_features[i], axis=1)

            # Compute local density using Gaussian kernel
            local_density = np.mean(np.exp(-distances**2 / (2 * self.sigma**2)))

            # Higher density means higher membership weight
            weights[i] = local_density

        # Normalize weights to [0, 1]
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-10)

        # Ensure minimum weight to avoid zero weights
        weights = np.maximum(weights, 0.1)

        self.membership_weights = weights

        print(f"  KECA weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        print(f"  KECA weights mean: {np.mean(weights):.4f}")

        return weights

class KNR_FuzzyMembership(FuzzyMembershipBase):
    """Kernel Nearest Regression for Fuzzy Membership"""

    def __init__(self, kernel_type='rbf', kernel_params=None, alpha=1.0):
        """Initialize KNR fuzzy membership"""
        super().__init__(kernel_type, kernel_params)
        self.alpha = alpha  # Regularization parameter

    def compute_membership_weights(self, X, y):
        """Compute KNR-based fuzzy membership weights"""
        print(f"Computing KNR fuzzy membership weights...")

        # Use kernel ridge regression - fix gamma parameter
        kernel_params_fixed = self.kernel_params.copy()
        if 'gamma' in kernel_params_fixed and kernel_params_fixed['gamma'] == 'scale':
            kernel_params_fixed['gamma'] = 1.0 / X.shape[1]

        krr = KernelRidge(kernel=self.kernel_type, alpha=self.alpha, **kernel_params_fixed)
        krr.fit(X, y.astype(float))

        # Predict on training data
        y_pred = krr.predict(X)

        # Convert predictions to membership weights
        # Higher confidence (closer to 0 or 1) means higher membership
        confidence = np.abs(y_pred - 0.5) * 2  # Scale to [0, 1]
        weights = confidence

        # Normalize and ensure minimum weight
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-10)
        weights = np.maximum(weights, 0.1)

        self.membership_weights = weights

        print(f"  KNR weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        print(f"  KNR weights mean: {np.mean(weights):.4f}")

        return weights

class SR_FuzzyMembership(FuzzyMembershipBase):
    """Sparse Representation for Fuzzy Membership"""

    def __init__(self, kernel_type='rbf', kernel_params=None, sparsity_param=0.1):
        """Initialize SR fuzzy membership"""
        super().__init__(kernel_type, kernel_params)
        self.sparsity_param = sparsity_param

    def compute_membership_weights(self, X, y):
        """Compute SR-based fuzzy membership weights"""
        print(f"Computing SR fuzzy membership weights...")

        n_samples = X.shape[0]

        # Compute kernel matrix
        K = self.compute_kernel_matrix(X)

        # Compute sparse representation weights using L1 regularization
        from sklearn.linear_model import Lasso

        weights = np.zeros(n_samples)

        for i in range(n_samples):
            # Use other samples to represent sample i
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False

            K_dict = K[mask, :][:, mask]  # Dictionary
            k_target = K[i, mask]  # Target

            # Solve sparse representation
            lasso = Lasso(alpha=self.sparsity_param, fit_intercept=False, max_iter=1000)
            try:
                lasso.fit(K_dict, k_target)
                # Reconstruction error as membership weight
                k_reconstructed = K_dict @ lasso.coef_
                reconstruction_error = np.linalg.norm(k_target - k_reconstructed)
                # Lower error means higher membership
                weights[i] = np.exp(-reconstruction_error)
            except:
                weights[i] = 0.5  # Default weight

        # Normalize weights
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-10)
        weights = np.maximum(weights, 0.1)

        self.membership_weights = weights

        print(f"  SR weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        print(f"  SR weights mean: {np.mean(weights):.4f}")

        return weights

class SVDD_FuzzyMembership(FuzzyMembershipBase):
    """Support Vector Data Description for Fuzzy Membership"""

    def __init__(self, kernel_type='rbf', kernel_params=None, nu=0.1):
        """Initialize SVDD fuzzy membership"""
        super().__init__(kernel_type, kernel_params)
        self.nu = nu  # Outlier fraction

    def compute_membership_weights(self, X, y):
        """Compute SVDD-based fuzzy membership weights"""
        print(f"Computing SVDD fuzzy membership weights...")

        # Use One-Class SVM for each class separately
        weights = np.zeros(len(X))

        for class_label in [0, 1]:
            class_mask = y == class_label
            if not np.any(class_mask):
                continue

            X_class = X[class_mask]

            # Train SVDD (One-Class SVM) on class data
            svdd = OneClassSVM(kernel=self.kernel_type, nu=self.nu, **self.kernel_params)
            svdd.fit(X_class)

            # Compute decision scores (distance to boundary)
            decision_scores = svdd.decision_function(X_class)

            # Convert to membership weights (higher score = higher membership)
            class_weights = decision_scores - np.min(decision_scores)
            class_weights = class_weights / (np.max(class_weights) + 1e-10)

            # Assign weights
            weights[class_mask] = class_weights

        # Ensure minimum weight
        weights = np.maximum(weights, 0.1)

        self.membership_weights = weights

        print(f"  SVDD weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        print(f"  SVDD weights mean: {np.mean(weights):.4f}")

        return weights

class OptimalKernels_FuzzyMembership(FuzzyMembershipBase):
    """Optimal Kernels for Fuzzy Membership"""

    def __init__(self, kernel_type='rbf', kernel_params=None):
        """Initialize Optimal Kernels fuzzy membership"""
        super().__init__(kernel_type, kernel_params)

    def compute_kernel_alignment(self, K, y):
        """Compute kernel alignment with ideal kernel"""
        # Create ideal kernel from labels
        y_matrix = np.outer(y, y)

        # Compute centered alignment
        n = len(y)
        H = np.eye(n) - np.ones((n, n)) / n

        K_c = H @ K @ H
        y_c = H @ y_matrix @ H

        alignment = np.trace(K_c @ y_c) / (np.sqrt(np.trace(K_c @ K_c) * np.trace(y_c @ y_c)) + 1e-10)

        return alignment

    def compute_membership_weights(self, X, y):
        """Compute optimal kernel-based fuzzy membership weights"""
        print(f"Computing Optimal Kernels fuzzy membership weights...")

        # Compute base kernel
        K = self.compute_kernel_matrix(X)
        n_samples = K.shape[0]

        # Compute local kernel alignments
        weights = np.zeros(n_samples)

        # Use sliding window approach for local alignment
        window_size = min(50, n_samples // 4)

        for i in range(n_samples):
            # Define local neighborhood
            distances = np.linalg.norm(X - X[i], axis=1)
            neighbor_indices = np.argsort(distances)[:window_size]

            # Extract local kernel and labels
            K_local = K[np.ix_(neighbor_indices, neighbor_indices)]
            y_local = y[neighbor_indices]

            # Compute local alignment
            local_alignment = self.compute_kernel_alignment(K_local, y_local)

            # Find position of current sample in neighborhood
            local_pos = np.where(neighbor_indices == i)[0][0]

            # Weight based on local alignment and centrality
            centrality = 1.0 / (1.0 + local_pos / window_size)  # Higher weight for more central samples
            weights[i] = abs(local_alignment) * centrality

        # Normalize weights
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-10)
        weights = np.maximum(weights, 0.1)

        self.membership_weights = weights

        print(f"  Optimal Kernels weights range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        print(f"  Optimal Kernels weights mean: {np.mean(weights):.4f}")

        return weights

class FuzzySVM:
    """Fuzzy Support Vector Machine"""

    def __init__(self, C=1.0, kernel='rbf', kernel_params=None, random_state=42):
        """
        Initialize Fuzzy SVM

        Args:
            C: Regularization parameter
            kernel: Kernel type
            kernel_params: Kernel parameters
            random_state: Random state
        """
        self.C = C
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params else {}
        self.random_state = random_state
        self.svm = None

    def fit(self, X, y, sample_weights=None):
        """Fit Fuzzy SVM with sample weights"""
        # Use sample_weight parameter in SVM
        self.svm = SVC(
            C=self.C,
            kernel=self.kernel,
            probability=True,
            random_state=self.random_state,
            **self.kernel_params
        )

        self.svm.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X):
        """Predict using fitted SVM"""
        return self.svm.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using fitted SVM"""
        return self.svm.predict_proba(X)

def evaluate_fuzzy_membership_method(membership_method, X_train, y_train, X_test, y_test, method_name):
    """Evaluate a fuzzy membership method with Fuzzy SVM"""
    print(f"\n--- Evaluating {method_name} ---")

    try:
        # Compute membership weights
        weights = membership_method.compute_membership_weights(X_train, y_train)

        # Data preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Fuzzy SVM with membership weights
        fsvm = FuzzySVM(C=1.0, kernel='rbf', kernel_params={'gamma': 'scale'}, random_state=42)
        fsvm.fit(X_train_scaled, y_train, sample_weights=weights)

        # Predict on test set
        y_pred = fsvm.predict(X_test_scaled)
        y_prob = fsvm.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Calculate SN and SP
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0

        results = {
            'Method': method_name,
            'Membership_Weights_Stats': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights)
            },
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
            'weights': weights,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        print(f"  AUC: {results['AUC']}")
        print(f"  ACC: {results['ACC (%)']}%")
        print(f"  MCC: {results['MCC']}")
        print(f"  Weight stats: mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")

        return results

    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def test_fuzzy_membership_methods():
    """Test all fuzzy membership methods"""
    print("Testing Fuzzy Membership Methods Implementation")
    print("="*60)

    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    n_features = 30
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
        'KECA': KECA_FuzzyMembership(kernel_type='rbf', n_components=10),
        'KNR': KNR_FuzzyMembership(kernel_type='rbf', alpha=1.0),
        'SR': SR_FuzzyMembership(kernel_type='rbf', sparsity_param=0.1),
        'SVDD': SVDD_FuzzyMembership(kernel_type='rbf', nu=0.1),
        'Optimal Kernels': OptimalKernels_FuzzyMembership(kernel_type='rbf')
    }

    results = []
    for name, method in methods.items():
        result = evaluate_fuzzy_membership_method(method, X_train, y_train, X_test, y_test, name)
        if result:
            results.append(result)

    if results:
        print(f"\n{'='*60}")
        print("Fuzzy Membership Methods Test Results")
        print(f"{'='*60}")

        df_results = pd.DataFrame([{k: v for k, v in r.items() if k not in ['weights', 'y_true', 'y_pred', 'y_prob', 'Membership_Weights_Stats']} for r in results])
        display_cols = ['Method', 'AUC', 'ACC (%)', 'MCC', 'SN', 'SP']
        print(df_results[display_cols].to_string(index=False))

    print("\nFuzzy membership methods implementation test completed!")

if __name__ == "__main__":
    test_fuzzy_membership_methods()