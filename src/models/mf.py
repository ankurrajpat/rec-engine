"""Matrix factorization with biases, trained with BCE on sampled negatives."""
import numpy as np
import pandas as pd
import torch


class MFModel(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.user_emb = torch.nn.Embedding(n_users, dim)
        self.item_emb = torch.nn.Embedding(n_items, dim)
        self.user_bias = torch.nn.Embedding(n_users, 1)
        self.item_bias = torch.nn.Embedding(n_items, 1)
        for emb in (self.user_emb, self.item_emb):
            torch.nn.init.normal_(emb.weight, std=0.05)
        for b in (self.user_bias, self.item_bias):
            torch.nn.init.zeros_(b.weight)

    def forward(self, users, items):
        dot = (self.user_emb(users) * self.item_emb(items)).sum(dim=1)
        return dot + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1)


def train_mf(train: pd.DataFrame, n_users: int, n_items: int, dim: int = 64,
             epochs: int = 4, negatives: int = 4, batch_size: int = 8192,
             lr: float = 0.02, seed: int = 42, max_per_epoch: int | None = None,
             verbose: bool = True):
    """Implicit-feedback MF: observed events are positives, unobserved pairs
    are sampled as negatives (uniform), optimized with BCE + Adam."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cpu"
    model = MFModel(n_users, n_items, dim, seed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    users = train.user.values
    items = train.item.values
    n = len(train)
    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        if max_per_epoch is not None:
            perm = perm[:max_per_epoch]
        total = 0.0
        steps = 0
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            u = torch.as_tensor(users[idx])
            pos_i = torch.as_tensor(items[idx])
            neg_i = torch.as_tensor(rng.integers(0, n_items, size=(len(idx), negatives)))
            u_neg = u.repeat_interleave(negatives)
            logits = torch.cat([model(u, pos_i), model(u_neg, neg_i.reshape(-1))])
            labels = torch.cat([torch.ones(len(idx)), torch.zeros(len(idx) * negatives)])
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        history.append(total / steps)
        if verbose:
            print(f"[mf] epoch {epoch + 1}/{epochs} bce={total / steps:.4f}", flush=True)
    model.eval()
    return model, {"final_bce": history[-1], "loss_history": history}


def mf_score_fn(model: MFModel):
    @torch.no_grad()
    def score(user_idx: np.ndarray) -> np.ndarray:
        u = torch.as_tensor(user_idx)
        ue = model.user_emb(u)  # (B, d)
        bias = model.user_bias(u)  # (B, 1)
        scores = ue @ model.item_emb.weight.T  # (B, n_items)
        scores = scores + bias + model.item_bias.weight.T
        return scores.numpy().astype(np.float32)
    return score
