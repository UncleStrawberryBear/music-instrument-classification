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
                x = x_hats[i]       # (N,)
                y = Y[i]            # (m,)
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


if __name__ == '__main__':
    # 模拟数据
    m, N, num_train = 249, 3804, 249
    D = torch.randn(m, N)
    labels = np.tile(np.arange(7), int(np.ceil(N / 7)))[:N]  # 修复标签长度不一致的问题
    assert len(labels) == N

    # 初始化 LISTA 模型
    lista_model = LISTA(D, depth=10, device=D.device)

    # 定义目标稀疏表示 (假设给出)
    X_target = torch.randn(num_train, N)       # shape: (batch, N)
    Y_input = D @ X_target.T                   # shape: (m, batch)

    # 定义 loss 和优化器
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lista_model.parameters(), lr=1e-3)

    # 训练 loop
    for epoch in range(30):
        lista_model.train()
        optimizer.zero_grad()
        X_pred = lista_model(Y_input.T)        # input shape: (batch, m)
        loss = loss_fn(X_pred, X_target)       # shape match: (batch, N)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # 分类器测试（optional）
    #classifier = SRCClassifierLISTA(lista_model, labels)
    #Y_test = torch.randn(m, 10)
    #preds = classifier.predict(Y_test)
    #print('Predicted labels:', preds)
    # 分类器测试（optional）
    classifier = SRCClassifierLISTA(lista_model, labels)

    # 从字典中采样 10 个测试样本（使用真实的字典列和标签）
    indices = np.random.choice(N, 10, replace=False)
    Y_test = D[:, indices]                   # shape: (m, 10)
    true_labels = labels[indices]           # shape: (10,)

    # 预测
    preds = classifier.predict(Y_test)

    # 打印对比
    print('True labels     :', true_labels)
    print('Predicted labels:', preds)
