
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
X = load_npz(ROOT / "data/raw/X.npz").tocsr()
y = np.load(ROOT / "data/raw/y.npy")

X.eliminate_zeros()

# Zero vectors
nnz_per_row = np.diff(X.indptr)
zero_rows = np.flatnonzero(nnz_per_row == 0)

print("=== ZERO VECTORS ===")
print("Count:", len(zero_rows))
print("Percentage:", 100 * len(zero_rows) / X.shape[0])
print("Indices (first 20):", zero_rows[:20])

# Exact duplicate sparse vectors
# Use hashes to avoid dense conversion
import hashlib

hashes = []
for i in range(X.shape[0]):
    row = X.getrow(i)
    payload = (
        row.indices.tobytes()
        + row.data.tobytes()
    )
    hashes.append(hashlib.sha256(payload).hexdigest())

unique_hashes, counts = np.unique(
    hashes, return_counts=True
)

print("\n=== DUPLICATE VECTORS ===")
print("Unique vectors:", len(unique_hashes))
print("Duplicate extra rows:", int(np.sum(counts - 1)))
print("Largest duplicate group:", int(counts.max()))

# Feature frequency thresholds
freq = np.bincount(X.indices, minlength=X.shape[1])

print("\n=== FEATURE FILTERING ===")
for threshold in [1, 2, 5, 10, 20, 50, 100]:
    remaining = np.sum(freq >= threshold)
    print(f"min_df={threshold}: {remaining:,} features")
