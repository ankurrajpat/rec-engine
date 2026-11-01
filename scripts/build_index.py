"""Build the FAISS inner-product index over two-tower item embeddings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np

from src import config


def main():
    item_emb = np.load(config.ARTIFACTS_DIR / "item_embeddings.npy").astype(np.float32)
    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(item_emb)
    faiss.write_index(index, str(config.ARTIFACTS_DIR / "item_index.faiss"))
    print(f"index built: {index.ntotal} items, dim {item_emb.shape[1]}")


if __name__ == "__main__":
    main()
