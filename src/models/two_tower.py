"""Two-tower retrieval model with in-batch sampled softmax.

User tower: user-id embedding (+ optional demographics, unused by default).
Item tower: item-id embedding + a linear projection of the multi-hot genre
vector, so brand-new items with known genres get a non-random vector
(cold-start fallback).
"""
import numpy as np
import pandas as pd
import torch

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
GENRE_TO_IDX = {g: i for i, g in enumerate(ALL_GENRES)}


def genre_multihot(movies: pd.DataFrame, n_items: int) -> torch.Tensor:
    """(n_items, n_genres) multi-hot tensor aligned to reindexed item ids."""
    mat = np.zeros((n_items, len(ALL_GENRES)), dtype=np.float32)
    for item, genres in zip(movies.item.values, movies.genres.values):
        for g in str(genres).split("|"):
            if g in GENRE_TO_IDX:
                mat[item, GENRE_TO_IDX[g]] = 1.0
    return torch.as_tensor(mat)


class TwoTowerModel(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64,
                 n_genres: int = len(ALL_GENRES), seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.user_emb = torch.nn.Embedding(n_users, dim)
        self.item_emb = torch.nn.Embedding(n_items, dim)
        self.genre_proj = torch.nn.Linear(n_genres, dim, bias=False)
        torch.nn.init.normal_(self.user_emb.weight, std=0.05)
        torch.nn.init.normal_(self.item_emb.weight, std=0.05)
        # Learnable logit scale (sampled softmax temperature).
        self.logit_scale = torch.nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def user_vector(self, users):
        return torch.nn.functional.normalize(self.user_emb(users), dim=1)

    def item_vector(self, items, genre_feats):
        raw = self.item_emb(items) + self.genre_proj(genre_feats)
        return torch.nn.functional.normalize(raw, dim=1)

    def forward(self, users, items, genre_feats):
        """In-batch logits: (B, B) similarity of each user to every item in
        the batch; the diagonal holds the positive pairs."""
        u = self.user_vector(users)
        v = self.item_vector(items, genre_feats)
        scale = torch.clamp(self.logit_scale, max=np.log(100.0)).exp()
        return scale * (u @ v.T)


def train_two_tower(train: pd.DataFrame, genre_feats: torch.Tensor, n_users: int,
                    n_items: int, dim: int = 64, epochs: int = 8,
                    batch_size: int = 2048, lr: float = 3e-3, seed: int = 42,
                    max_per_epoch: int | None = None, verbose: bool = True):
    torch.manual_seed(seed)
    model = TwoTowerModel(n_users, n_items, dim, n_genres=genre_feats.shape[1], seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    users = train.user.values
    items = train.item.values
    n = len(train)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        perm = rng.permutation(n)
        if max_per_epoch is not None:
            perm = perm[:max_per_epoch]
        total, steps = 0.0, 0
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            u = torch.as_tensor(users[idx])
            i = torch.as_tensor(items[idx])
            gf = genre_feats[i]
            logits = model(u, i, gf)
            labels = torch.arange(len(idx))
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        history.append(total / steps)
        if verbose:
            print(f"[two-tower] epoch {epoch + 1}/{epochs} ce={total / steps:.4f}", flush=True)
    model.eval()
    return model, {"final_ce": history[-1], "loss_history": history}


@torch.no_grad()
def export_item_embeddings(model: TwoTowerModel, genre_feats: torch.Tensor,
                           n_items: int, batch: int = 4096) -> np.ndarray:
    out = []
    for s in range(0, n_items, batch):
        idx = torch.arange(s, min(s + batch, n_items))
        out.append(model.item_vector(idx, genre_feats[idx]).numpy())
    return np.vstack(out).astype(np.float32)


@torch.no_grad()
def export_user_embeddings(model: TwoTowerModel, n_users: int, batch: int = 4096) -> np.ndarray:
    out = []
    for s in range(0, n_users, batch):
        idx = torch.arange(s, min(s + batch, n_users))
        out.append(model.user_vector(idx).numpy())
    return np.vstack(out).astype(np.float32)


def two_tower_score_fn(user_emb: np.ndarray, item_emb: np.ndarray):
    def score(user_idx: np.ndarray) -> np.ndarray:
        return (user_emb[user_idx] @ item_emb.T).astype(np.float32)
    return score
