# listasrc.py
# -----------------------------------------------------------
# Train-free Coupled-LISTA (Gregor & LeCun ′10, Chen & Chen ′18)
# 用于 Sparse Representation Classification (SRC)
# -----------------------------------------------------------
import torch
import numpy as np
from typing import Sequence

# ---------- 工具 ---------- #
def soft_threshold(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """逐元素 soft-threshold"""
    return torch.sign(x) * torch.relu(x.abs() - tau)

def spectral_norm_sq(M: torch.Tensor) -> float:
    """‖M‖₂² —— 最大奇异值平方"""
    return torch.linalg.svdvals(M)[0].pow(2).item()

# ---------- Coupled-LISTA 网络 ---------- #
class LISTA(torch.nn.Module):
    r"""
    Coupled-LISTA (LISTA-CP). 公式：
        x^{k+1} = shrink( B y + (I − B A) x^{k}, θ_k )
    默认所有层共用 B, 且 θ_k 可学习（初始化 λ/L；可置 requires_grad=False
    则退化为无训练版本）。
    """
    def __init__(
        self,
        A: torch.Tensor,         # (m, N) — 字典/测量矩阵
        depth: int      = 10,    # 层数
        lam:   float    = 0.1,   # LASSO λ
        learn_theta: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.register_buffer("A",  A.to(device))   # (m, N)
        self.register_buffer("At", A.t().to(device))
        self.N     = A.shape[1]
        self.depth = depth
        self.device = device

        # —— 步长 1/L —— #
        L = spectral_norm_sq(A)
        B_init = self.At / L                    # (N, m)
        self.B = torch.nn.Parameter(B_init, requires_grad=False)

        # —— 阈值 θ_k —— #
        theta0 = lam / L
        theta_vec = theta0 * torch.ones(depth, device=device)
        self.theta = (torch.nn.Parameter(theta_vec)
                      if learn_theta else
                      torch.nn.Parameter(theta_vec, requires_grad=False))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y : (batch, m) —— 多列测试向量
        return : (batch, N) —— 稀疏系数估计
        """
        I = torch.eye(self.N, device=self.device)  # (N, N)
        S = I - self.B @ self.A                    # 预先计算 W₂
        x = torch.zeros(y.size(0), self.N, device=self.device)

        for k in range(self.depth):
            x = soft_threshold(x @ S.T + y @ self.B.T, self.theta[k])
        return x


# ---------- SRC 分类封装 ---------- #
class SRCClassifierLISTA:
    """
    用 Coupled-LISTA 近似求解稀疏系数，然后按残差做 SRC。
    若 `learn_theta=True` 可在 fit() 内对 θ_k 进行少量微调。
    """
    def __init__(
        self,
        depth: int      = 10,
        lam:   float    = 0.1,
        learn_theta: bool = False,
        finetune_epochs: int = 20,
        lr: float = 1e-3,
        device: torch.device = torch.device('cpu')
    ):
        self.depth   = depth
        self.lam     = lam
        self.learn_theta    = learn_theta
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.device = device

        self.model   = None          # LISTA 网络
        self.labels  = None          # numpy array, shape (N,)

    # ---------- 训练 / 保存字典 ---------- #
    def fit(self, D: torch.Tensor, y: Sequence[int]):
        """
        D : (feat_dim, num_train)  torch.Tensor
        y : 列标签 (len = num_train)
        """
        D = D.to(self.device)
        self.labels = np.asarray(y)
        # 构造 LISTA 网络
        self.model = LISTA(D, depth=self.depth,
                           lam=self.lam,
                           learn_theta=self.learn_theta,
                           device=self.device).to(self.device)

        # —— 如需微调 θ_k —— #
        if self.learn_theta:
            # 自监督：用字典列本身作为 y，target 稀疏系数 e_i
            I_targets = torch.eye(D.shape[1], device=self.device)  # (N, N)
            optimizer = torch.optim.Adam([self.model.theta], lr=self.lr)
            loss_fn   = torch.nn.MSELoss()
            Y_train   = D.t()    # (N, m) → (batch=N, m)

            for ep in range(self.finetune_epochs):
                optimizer.zero_grad()
                X_hat = self.model(Y_train)             # (N, N)
                loss = loss_fn(X_hat, I_targets)
                loss.backward()
                optimizer.step()
                if (ep+1) % 5 == 0:
                    print(f"[LISTA θ-finetune] epoch {ep+1}/{self.finetune_epochs}  "
                          f"MSE={loss.item():.4e}")

    # ---------- 推理 ---------- #
    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        X : (feat_dim, num_test)  torch.Tensor
        return : numpy array of predicted labels
        """
        if self.model is None:
            raise RuntimeError("You must call .fit() first.")

        D = self.model.A                       # (m, N)
        # L2 归一化字典列 & 测试向量
        Dn = D / (D.norm(dim=0, keepdim=True) + 1e-8)
        X  = X.to(self.device)
        Xn = X / (X.norm(dim=0, keepdim=True) + 1e-8)

        uniq_lbls = np.unique(self.labels)
        preds = []

        # 批处理
        Xn_T = Xn.t()                          # (num_test, m)
        Xhat = self.model(Xn_T)                # (num_test, N)

        for i, x_hat in enumerate(Xhat):
            y = Xn_T[i]                        # (m,)
            residuals = []
            for c in uniq_lbls:
                mask = torch.tensor(self.labels == c, device=self.device)
                x_c  = torch.zeros_like(x_hat)
                x_c[mask] = x_hat[mask]
                r = (y - Dn @ x_c).norm().item()
                residuals.append(r)
            preds.append(int(uniq_lbls[int(np.argmin(residuals))]))
        return np.array(preds)
