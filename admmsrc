import torch
import numpy as np


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Soft thresholding operator: S_tau(v) = sign(v) * max(|v| - tau, 0)
    """
    return torch.sign(x) * torch.relu(torch.abs(x) - tau)


class SRCClassifierADMM:
    """
    Sparse Representation-based Classifier using ADMM to solve a LASSO subproblem.
    """

    def __init__(
        self,
        lambda_reg: float = 0.1,
        rho: float = 1.0,
        max_iters: int = 50,
        device: torch.device = torch.device('cpu')
    ):
        self.lambda_reg = lambda_reg
        self.rho = rho
        self.max_iters = max_iters
        self.device = device
        self.dictionary = None   # torch.Tensor of shape (m, N)
        self.labels = None       # numpy.ndarray of shape (N,)

    def fit(self, D: torch.Tensor, y: np.ndarray):
        """
        Store dictionary and labels.
        D: feature_dim x num_train_samples
        y: array of length num_train_samples (labels)
        """
        self.dictionary = D.to(self.device)
        self.labels = np.array(y)

    def _admm(self, y: torch.Tensor) -> torch.Tensor:
        """
        Solve for x in: min_x 0.5||D x - y||^2 + lambda_reg * ||x||_1 via ADMM.
        y: (m,) torch tensor
        Returns x_hat: (N,) torch tensor
        """
        D = self.dictionary  # (m, N)
        m, N = D.shape

        # Precompute inverse of (D^T D + rho I)
        DtD = D.T @ D
        inv_mat = torch.inverse(DtD + self.rho * torch.eye(N, device=self.device))

        # Initialize variables
        x = torch.zeros(N, device=self.device)
        z = torch.zeros(N, device=self.device)
        u = torch.zeros(N, device=self.device)

        # ADMM iterations
        for _ in range(self.max_iters):
            # x-update: solve (D^T D + rho I) x = D^T y + rho (z - u)
            x = inv_mat @ (D.T @ y + self.rho * (z - u))
            # z-update: soft threshold
            z = soft_threshold(x + u, self.lambda_reg / self.rho)
            # u-update: dual variable
            u = u + x - z

        return x

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Classify each column in X by computing ADMM-based sparse code
        and choosing the class whose sub-dictionary reconstruction gives
        the smallest residual.

        X: feature_dim x num_test_samples
        Returns: numpy array of predicted labels
        """
        # Normalize dictionary columns
        D = self.dictionary
        norms = D.norm(dim=0, keepdim=True) + 1e-8
        D_normed = D / norms

        unique_labels = sorted(set(self.labels))
        preds = []

        # For each test vector
        for i in range(X.shape[1]):
            y = X[:, i].to(self.device)
            # normalize y
            y = y / (y.norm() + 1e-8)

            # solve sparse code via ADMM
            x_hat = self._admm(y)

            # compute residuals per class
            residuals = []
            for cls in unique_labels:
                # mask x_hat for current class
                mask = (self.labels == cls)
                x_c = torch.zeros_like(x_hat)
                x_c[mask] = x_hat[mask]

                # reconstruct
                y_c = D_normed @ x_c
                r = (y - y_c).norm().item()
                residuals.append(r)

            # pick class with minimal residual
            preds.append(unique_labels[int(np.argmin(residuals))])

        return np.array(preds)


if __name__ == '__main__':
    # 简单演示用法
    m, N, T = 100, 200, 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 随机构造字典和标签
    D = torch.randn(m, N, device=device)
    labels = np.repeat(np.arange(5), N//5)

    clf = SRCClassifierADMM(lambda_reg=0.1, rho=1.0, max_iters=30, device=device)
    clf.fit(D, labels)
    # 随机测试数据
    X_test = torch.randn(m, T, device=device)
    preds = clf.predict(X_test)
    print('Predicted labels:', preds)
