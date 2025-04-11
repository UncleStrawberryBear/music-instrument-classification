# src_classifier.py

import torch
import numpy as np

class SRCClassifier:
    """
    使用 OMP 做稀疏表示分类 (residual-based)
    """
    def __init__(self, sparsity=20, device= torch.device('cpu')):
        self.sparsity = sparsity
        self.device = device
        self.dictionary = None   # (feat_dim, num_train_samples)
        self.labels = None       # list/array, 每个列对应的标签

    def fit(self, D, y):
        """
        D: [feat_dim, num_train_samples], 每一列是一个训练样本的特征
        y: (num_train_samples,) 或 list
        """
        self.dictionary = D
        self.labels = np.array(y)

    def predict(self, X):
        """
        X: [feat_dim, num_test_samples]
        返回: 长度 = num_test_samples 的预测标签数组
        """
        preds = []
        # 先归一化字典的每一列
        norms = torch.norm(self.dictionary, dim=0, keepdim=True) + 1e-8
        D_normed = self.dictionary / norms

        unique_labels = sorted(set(self.labels))

        for i in range(X.shape[1]):
            y = X[:, i]
            # 归一化 y
            y_norm = torch.norm(y)
            if y_norm > 1e-8:
                y = y / y_norm
            # OMP 求解 x
            x_hat = self._omp(y, D_normed, self.sparsity)

            # 计算对每个类别的重构残差
            residuals = []
            for cls in unique_labels:
                idx_cls = np.where(self.labels == cls)[0]
                x_c = np.zeros_like(x_hat)
                x_c[idx_cls] = x_hat[idx_cls].cpu().numpy()

                # 用 D_normed 的相应列重构
                D_c = D_normed[:, idx_cls].cpu().numpy()
                r = np.linalg.norm(y.cpu().numpy() - D_c @ x_c[idx_cls])
                residuals.append(r)

            pred_cls = unique_labels[np.argmin(residuals)]
            preds.append(pred_cls)

        return np.array(preds)

    def _omp(self, y, D, k):
        """
        y: [feat_dim]
        D: [feat_dim, num_samples]
        k: sparsity
        返回: x_hat: [num_samples]
        """
        # 转 numpy 计算
        y_np = y.cpu().numpy()
        D_np = D.cpu().numpy()

        residual = y_np.copy()
        idx_support = []
        x_hat = np.zeros(D_np.shape[1], dtype=np.float32)

        for _ in range(k):
            # 相关匹配
            corrs = np.abs(D_np.T @ residual)  # (num_samples,)
            idx_new = np.argmax(corrs)
            if idx_new not in idx_support:
                idx_support.append(idx_new)

            # 最小二乘求解
            D_sub = D_np[:, idx_support]
            x_sub, _, _, _ = np.linalg.lstsq(D_sub, y_np, rcond=None)
            residual = y_np - D_sub @ x_sub

        x_hat[idx_support] = x_sub
        return torch.from_numpy(x_hat).to(self.device)
