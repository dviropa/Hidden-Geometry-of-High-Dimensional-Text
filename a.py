
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

data_dir = Path("data/raw")

X = load_npz(data_dir / "X.npz")
y = np.load(data_dir / "y.npy")

n, d = X.shape

print("=== DATASET VALIDATION ===")
print(f"Samples: {n:,}")
print(f"Features: {d:,}")
print(f"Classes: {len(np.unique(y))}")
print(f"Non-zero values: {X.nnz:,}")
print(f"Sparsity: {1 - X.nnz / (n * d):.6%}")
print(f"Missing values: {np.isnan(X.data).sum()}")
print(f"Dimension-to-sample ratio: {d / n:.2f}")

print("\n=== CLASS DISTRIBUTION ===")
classes, counts = np.unique(y, return_counts=True)

for cls, count in zip(classes, counts):
    print(f"Class {cls:2d}: {count:,} samples")
