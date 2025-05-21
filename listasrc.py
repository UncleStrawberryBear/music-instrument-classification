import torch
import numpy as np


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(x) * torch.relu(torch.abs(x) - tau)


class LISTA(torch.nn.Module):
    """
    Trainable LISTA network as an unfolded sparse coder
    """
    def __init__(
        self,
        D: torch.Tensor,
        depth: int = 10,
        lambda_init: float = 0.1,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.device = device
        self.depth = depth
        self.D = D.to(device)                        # (m, N)
        self.Dt = D.t().to(device)                   # (N, m)
        self.N = D.shape[1]

        # 初始化参数 W, S, threshold
        self.W = torch.nn.Parameter(torch.eye(self.N, D.shape[0], device=device))  # (N, m)
        self.S = torch.nn.Parameter(torch.eye(self.N, device=device))              # (N, N)
        self.threshold = torch.nn.Parameter(lambda_init * torch.ones(depth, device=device))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (batch, m)
        return: (batch, N)
        """
        x = torch.zeros((y.shape[0], self.N), device=self.device)
        for t in range(self.depth):
            x = soft_threshold(x @ self.S.T + y @ self.W.T, self.threshold[t])
        return x


class SRCClassifierLISTA:
    """
    Classification wrapper around a trained LISTA sparse coder
    """
    def __init__(self, lista_model: torch.nn.Module, labels: np.ndarray):
        self.model = lista_model.eval()
        self.labels = np.array(labels)
        assert len(self.labels) == self.model.D.shape[1], "标签长度与字典样本数不一致！"

    def predict(self, Y: torch.Tensor) -> np.ndarray:
        """
        Y: (m, num_test) torch tensor
        return: predicted label list
        """
        with torch.no_grad():
            Y = Y.T  # shape (num_test, m)
            x_hats = self.model(Y)  # shape (num_test, N)

            D = self.model.D        # (m, N)
            preds = []
            for i in range(x_hats.shape[0]):
                x = x_hats[i]
                y = Y[i]
                residuals = []
                for cls in sorted(set(self.labels)):
                    mask = (self.labels == cls)
                    x_c = torch.zeros_like(x)
                    x_c[mask] = x[mask]
                    y_c = D @ x_c
                    r = (y - y_c).norm().item()
                    residuals.append(r)
                preds.append(int(np.argmin(residuals)))
        return np.array(preds)
