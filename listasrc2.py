# listasrc2.py
# -----------------------------------------------------------
# Coupled‑LISTA + SRC 分类器（可选学习 θ_k 与步长 B=W1）
# -----------------------------------------------------------
import torch
import numpy as np
from typing import Sequence, List

# ---------- 工具 ---------- #

def soft_threshold(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """逐元素 soft‑threshold"""
    return torch.sign(x) * torch.relu(x.abs() - tau)


def spectral_norm_sq(mat: torch.Tensor) -> float:
    """‖M‖₂² (最大奇异值平方)"""
    return torch.linalg.svdvals(mat)[0].pow(2).item()

# ---------- Coupled‑LISTA 网络 ---------- #
class LISTA(torch.nn.Module):
    r"""Coupled‑LISTA (LISTA‑CP)
    x^{k+1} = shrink( B y + (I − B A) x^{k}, θ_k )
    默认 B 共用，θ_k 向量共用或可学习。可通过 learn_B / learn_theta 打开训练。
    """

    def __init__(
        self,
        A: torch.Tensor,                # (m, N) 字典 / 测量矩阵
        depth: int = 10,
        lam: float = 0.1,
        learn_B: bool = False,
        learn_theta: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.register_buffer("A", A.to(device))
        self.register_buffer("At", A.t().to(device))
        self.device = device
        self.depth = depth
        self.N = A.shape[1]

        # —— 初始化 B ——
        L = spectral_norm_sq(A)
        B_init = self.At / L                      # (N, m)
        self.B = torch.nn.Parameter(B_init, requires_grad=learn_B)

        # —— 初始化 θ_k ——
        theta0 = lam / L
        theta_vec = theta0 * torch.ones(depth, device=device)
        self.theta = torch.nn.Parameter(theta_vec, requires_grad=learn_theta)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """y: (batch, m) → x_hat: (batch, N)"""
        I = torch.eye(self.N, device=self.device)
        S = I - self.B @ self.A                    # W2 = I − BA
        x = torch.zeros(y.size(0), self.N, device=self.device)
        for k in range(self.depth):
            x = soft_threshold(x @ S.T + y @ self.B.T, self.theta[k])
        return x


# ---------- SRC 分类封装 ---------- #
class SRCClassifierLISTA:
    """LISTA‑SRC 分类器，支持联合学习 θ_k 与 B"""

    def __init__(
        self,
        depth: int = 10,
        lam: float = 0.1,
        learn_theta: bool = False,
        learn_B: bool = False,
        finetune_epochs: int = 30,
        lr_theta: float = 1e-3,
        lr_B: float = 3e-4,
        device: torch.device = torch.device("cpu"),
    ):
        self.depth = depth
        self.lam = lam
        self.learn_theta = learn_theta
        self.learn_B = learn_B
        self.finetune_epochs = finetune_epochs
        self.lr_theta = lr_theta
        self.lr_B = lr_B
        self.device = device
        self.model: LISTA | None = None
        self.labels: np.ndarray | None = None

    # ---------- 训练 ---------- #
    def fit(self, D: torch.Tensor, y: Sequence[int]):
        """D: (feat_dim, N) torch.Tensor, y: labels (len=N)"""
        D = D.to(self.device)
        self.labels = np.asarray(y)

        # ⚠ A = D.T → (m=N_feat, N_samples)
        self.model = LISTA(
            A=D.t(),
            depth=self.depth,
            lam=self.lam,
            learn_B=self.learn_B,
            learn_theta=self.learn_theta or self.learn_B,  # 开 B 必须让 θ grad 通过
            device=self.device,
        ).to(self.device)

        # ----- 可选微调 θ 与/或 B ----- #
        if self.learn_theta or self.learn_B:
            Y_train = D.t()                              # (N, m)
            I_targets = torch.eye(D.shape[1], device=self.device)
            params: List[dict] = []
            if self.learn_theta:
                params.append({"params": self.model.theta, "lr": self.lr_theta})
            if self.learn_B:
                params.append({"params": self.model.B, "lr": self.lr_B})
            opt = torch.optim.Adam(params)
            mse = torch.nn.MSELoss()
            for ep in range(self.finetune_epochs):
                opt.zero_grad()
                x_hat = self.model(Y_train)
                loss = mse(x_hat, I_targets)
                loss.backward()
                opt.step()
                if (ep + 1) % 5 == 0:
                    print(
                        f"[Finetune] Epoch {ep+1}/{self.finetune_epochs}  MSE={loss.item():.4e}"
                    )

    # ---------- 预测 ---------- #
    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call .fit() first.")
        A = self.model.A             # (m,N)
        Dn = A / (A.norm(dim=0, keepdim=True) + 1e-8)
        Xn = X / (X.norm(dim=0, keepdim=True) + 1e-8)
        uniq = np.unique(self.labels)
        Xhat = self.model(Xn.t())    # (T, N)
        preds = []
        for i, x in enumerate(Xhat):
            y = Xn.t()[i]
            res = []
            for c in uniq:
                mask = torch.tensor(self.labels == c, device=self.device)
                x_c = torch.zeros_like(x)
                x_c[mask] = x[mask]
                res.append((y - Dn @ x_c).norm().item())
            preds.append(int(uniq[int(np.argmin(res))]))
        return np.asarray(preds)
