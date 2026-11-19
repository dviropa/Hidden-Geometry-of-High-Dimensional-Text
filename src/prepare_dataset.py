
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

X = load_npz(RAW / "X.npz").tocsr()
y = np.load(RAW / "y.npy")

# Indices verified in duplicate analysis
same_label_duplicates = [10629, 8236, 13812, 18121]
conflicting_labels = [4025, 7738]
zero_vectors = [3248]

excluded = sorted(set(
    same_label_duplicates
    + conflicting_labels
    + zero_vectors
))

keep = np.ones(X.shape[0], dtype=bool)
keep[excluded] = False

X_clean = X[keep]
y_clean = y[keep]
original_indices = np.flatnonzero(keep)

save_npz(OUT / "X_clean.npz", X_clean)
np.save(OUT / "y_clean.npy", y_clean)
np.save(OUT / "original_indices.npy", original_indices)

print("Original samples:", X.shape[0])
print("Removed samples:", len(excluded))
print("Clean samples:", X_clean.shape[0])
print("Features:", X_clean.shape[1])
print("Classes:", len(np.unique(y_clean)))
