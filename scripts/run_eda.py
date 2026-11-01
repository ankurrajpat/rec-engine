"""EDA: dataset stats + figures saved to results/figures/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config, data


def main():
    ratings = data.load_ratings()
    movies = data.load_movies()
    train, test, cutoff = data.chronological_split(ratings)

    n_users, n_items = ratings.user.nunique(), ratings.item.nunique()
    density = len(ratings) / (n_users * n_items)
    stats = {
        "n_ratings": len(ratings),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "density": float(density),
        "sparsity": float(1 - density),
        "rating_mean": float(ratings.rating.mean()),
        "train_events": len(train),
        "test_events": len(test),
        "split_cutoff_unix": cutoff,
        "test_rows_dropped_cold": int(len(test) - len(data.filter_seen(train, test))),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ratings.rating.value_counts().sort_index().plot.bar(ax=axes[0], color="#4C72B0")
    axes[0].set_title("Rating distribution")
    axes[0].set_xlabel("stars")

    item_counts = ratings.item.value_counts().values
    axes[1].loglog(np.arange(1, len(item_counts) + 1), np.sort(item_counts)[::-1], color="#DD8452")
    axes[1].set_title("Item long tail (log-log)")
    axes[1].set_xlabel("item rank")
    axes[1].set_ylabel("# ratings")

    user_counts = ratings.user.value_counts()
    axes[2].hist(user_counts.values, bins=50, color="#55A868")
    axes[2].set_title("Ratings per user")
    axes[2].set_xlabel("# ratings")
    axes[2].set_ylabel("# users")

    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "eda.png", dpi=110)

    genre_counts = movies.genres.str.split("|").explode().value_counts().head(10)
    fig2, ax = plt.subplots(figsize=(7, 4))
    genre_counts.iloc[::-1].plot.barh(ax=ax, color="#8172B3")
    ax.set_title("Top genres")
    fig2.tight_layout()
    fig2.savefig(config.FIGURES_DIR / "genres.png", dpi=110)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
