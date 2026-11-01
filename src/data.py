"""MovieLens 1M download, loading, reindexing and chronological split."""
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download(dest: Path = config.DATA_DIR.parent) -> Path:
    """Download and extract MovieLens 1M next to data/raw. Idempotent."""
    target = dest / "ml-1m"
    if target.exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ml-1m.zip"
    urllib.request.urlretrieve(ML1M_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()
    return target


def load_ratings(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Load ratings.dat with users/items reindexed to contiguous 0-based ids.

    Returns a DataFrame [user, item, rating, timestamp] plus writes id maps
    next to the data so every stage uses the same indexing.
    """
    df = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_raw", "item_raw", "rating", "timestamp"],
    )
    user_ids = {int(u): i for i, u in enumerate(sorted(df.user_raw.unique()))}
    item_ids = {int(m): i for i, m in enumerate(sorted(df.item_raw.unique()))}
    df["user"] = df.user_raw.map(user_ids)
    df["item"] = df.item_raw.map(item_ids)
    maps = {"user_raw_to_idx": user_ids, "item_raw_to_idx": item_ids}
    (data_dir / "id_maps.json").write_text(json.dumps(maps))
    return df[["user", "item", "rating", "timestamp"]].sort_values("timestamp").reset_index(drop=True)


def load_movies(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Load movies.dat joined to the reindexed item ids (with genres)."""
    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        engine="python",
        encoding="latin-1",
        names=["item_raw", "title", "genres"],
    )
    maps = json.loads((data_dir / "id_maps.json").read_text())
    item_map = {int(k): v for k, v in maps["item_raw_to_idx"].items()}
    movies["item"] = movies.item_raw.map(item_map)
    return movies.dropna(subset=["item"]).assign(item=lambda d: d.item.astype(int))


def load_users(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Load users.dat joined to the reindexed user ids (gender/age/occupation)."""
    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_raw", "gender", "age", "occupation", "zip"],
    )
    maps = json.loads((data_dir / "id_maps.json").read_text())
    user_map = {int(k): v for k, v in maps["user_raw_to_idx"].items()}
    users["user"] = users.user_raw.map(user_map)
    return users.dropna(subset=["user"]).assign(user=lambda d: d.user.astype(int))


def chronological_split(ratings: pd.DataFrame, frac: float = config.TRAIN_FRAC):
    """Global timestamp split: first `frac` of the timeline is train.

    Returns (train, test, cutoff). Every train event is strictly older than
    every test event, which mirrors how a recommender is used in production:
    train on the past, predict the future.
    """
    cutoff = ratings.timestamp.quantile(frac)
    train = ratings[ratings.timestamp < cutoff].reset_index(drop=True)
    test = ratings[ratings.timestamp >= cutoff].reset_index(drop=True)
    return train, test, int(cutoff)


def filter_seen(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Drop test rows whose user or item never appears in train.

    Those rows are unscoreable for id-based models; they are the subject of
    the cold-start analysis instead of silently inflating/deflating metrics.
    """
    seen_users = set(train.user.unique())
    seen_items = set(train.item.unique())
    keep = test.user.isin(seen_users) & test.item.isin(seen_items)
    return test[keep].reset_index(drop=True)
