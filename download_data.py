
from sklearn.datasets import fetch_20newsgroups_vectorized
from scipy.sparse import save_npz
import numpy as np
from pathlib import Path

# Create data directory
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

# Download full dataset
dataset = fetch_20newsgroups_vectorized(
    subset="all",
    data_home=str(data_dir),
    download_if_missing=True
)

X = dataset.data
y = dataset.target

# Save feature matrix and labels
save_npz(data_dir / "X.npz", X)
np.save(data_dir / "y.npy", y)

print("Download completed!")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", len(np.unique(y)))
