
from pathlib import Path
from collections import defaultdict
import hashlib
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]

X = load_npz(ROOT / "data/raw/X.npz").tocsr()
y = np.load(ROOT / "data/raw/y.npy")

X.sum_duplicates()
X.sort_indices()
X.eliminate_zeros()

groups = defaultdict(list)

for i in range(X.shape[0]):
    start, end = X.indptr[i:i+2]

    payload = (
        X.indices[start:end].tobytes()
        + X.data[start:end].tobytes()
    )

    key = hashlib.sha256(payload).hexdigest()
    groups[key].append(i)

duplicate_groups = [
    indices
    for indices in groups.values()
    if len(indices) > 1
]

print("=== DUPLICATE GROUPS ===")
print("Groups:", len(duplicate_groups))

for group in duplicate_groups:
    labels = y[group]
    print(
        f"Indices: {group} | "
        f"Classes: {labels.tolist()} | "
        f"Same class: {len(set(labels)) == 1}"
    )
