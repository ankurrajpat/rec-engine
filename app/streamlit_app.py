"""Streamlit demo: pick a user, compare models, inspect recommendations."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="rec-engine", page_icon="🎬", layout="wide")


@st.cache_resource
def load_artifacts():
    art = config.ARTIFACTS_DIR
    movies = pd.read_csv(config.DATA_DIR / "movies.dat", sep="::", engine="python",
                         encoding="latin-1", names=["item_raw", "title", "genres"])
    id_maps = json.loads((config.DATA_DIR / "id_maps.json").read_text())
    train = pd.read_csv(art / "train.csv.gz")
    return {
        "user_emb": np.load(art / "user_embeddings.npy"),
        "item_emb": np.load(art / "item_embeddings.npy"),
        "pop": np.load(art / "popularity_scores.npy"),
        "movies": movies,
        "id_maps": id_maps,
        "train": train,
    }


def recommend(A, user_raw, k, model):
    idx_to_raw = {int(v): int(k_) for k_, v in A["id_maps"]["item_raw_to_idx"].items()}
    user_map = A["id_maps"]["user_raw_to_idx"]
    uidx = int(user_map[str(user_raw)]) if str(user_raw) in user_map else None
    fallback = None
    if model == "popularity" or uidx is None:
        scores = A["pop"]
        fallback = "popularity" if uidx is None and model != "popularity" else None
    else:
        scores = A["user_emb"][uidx] @ A["item_emb"].T
    order = np.argsort(-scores)
    if uidx is not None:
        seen = set(A["train"].loc[A["train"].user == uidx, "item"].values)
    else:
        seen = set()
    picks = [i for i in order if i not in seen][:k]
    rows = []
    for rank, i in enumerate(picks, 1):
        raw = idx_to_raw[int(i)]
        m = A["movies"][A["movies"].item_raw == raw].iloc[0]
        rows.append({"rank": rank, "title": m.title, "genres": m.genres.replace("|", ", "),
                     "score": round(float(scores[i]), 4)})
    return pd.DataFrame(rows), fallback, uidx


def main():
    st.title("🎬 rec-engine")
    st.caption("Two-tower movie recommendations trained on MovieLens 1M")
    A = load_artifacts()

    st.sidebar.header("Inputs")
    user_raw = st.sidebar.number_input("MovieLens user id (1-6040; any other id demos cold-start)",
                                       min_value=0, max_value=999999, value=1)
    k = st.sidebar.slider("Number of recommendations", 5, 30, 10)
    model = st.sidebar.selectbox("Model", ["two_tower", "popularity"])

    df, fallback, uidx = recommend(A, user_raw, k, model)
    if fallback:
        st.warning(f"User {user_raw} has no train history - showing the popularity fallback.")
    st.subheader(f"Top {k} for user {user_raw} ({model})")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if uidx is not None:
        hist = A["train"][A["train"].user == uidx].nlargest(10, "rating")
        idx_to_raw = {int(v): int(k_) for k_, v in A["id_maps"]["item_raw_to_idx"].items()}
        hist_titles = [A["movies"][A["movies"].item_raw == idx_to_raw[int(i)]].iloc[0].title
                       for i in hist.item.values]
        with st.expander("User's highest-rated train movies (model never sees these at serve time)"):
            st.write(pd.DataFrame({"title": hist_titles, "rating": hist.rating.values}))

    st.sidebar.markdown("---")
    st.sidebar.caption("Artifacts: artifacts/ (run scripts/train_all.py to regenerate)")


if __name__ == "__main__":
    main()
