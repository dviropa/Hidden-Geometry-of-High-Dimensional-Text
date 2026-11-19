
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
# OUT = ROOT / "results" / "svd"

SEED = 42
N_SAMPLES = 2000
N_COMPONENTS = 300
OUT = ROOT / "results" / f"svd_{N_COMPONENTS}"
OUT.mkdir(parents=True, exist_ok=True)

# Load cleaned data
X = load_npz(DATA / "X_clean.npz").tocsr()
y = np.load(DATA / "y_clean.npy")

# Reproducible stratified subsample
indices, _ = train_test_split(
    np.arange(X.shape[0]),
    train_size=N_SAMPLES,
    stratify=y,
    random_state=SEED
)

X_sample = X[indices]
y_sample = y[indices]

print("=== INPUT ===", flush=True)
print("Samples:", X_sample.shape[0], flush=True)
print("Features:", X_sample.shape[1], flush=True)
print("Nonzero values:", X_sample.nnz, flush=True)

# Sparse-friendly dimensionality reduction
svd = TruncatedSVD(
    n_components=N_COMPONENTS,
    algorithm="randomized",
    n_iter=7,
    random_state=SEED
)

print("\nRunning TruncatedSVD...", flush=True)
Z = svd.fit_transform(X_sample)

explained = svd.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("\n=== SVD RESULTS ===")
for k in [10, 20, 50, 100, 200, 300]:
    print(
        f"Components {k}: "
        f"{cumulative[k - 1]:.4%} explained variance"
    )

print("\nTotal explained variance:",
      f"{cumulative[-1]:.4%}")

for threshold in [0.90, 0.95, 0.99]:
    reached = np.flatnonzero(cumulative >= threshold)

    if len(reached):
        print(
            f"{threshold:.0%} threshold: "
            f"{reached[0] + 1} components"
        )
    else:
        print(
            f"{threshold:.0%} threshold: "
            f"not reached within {N_COMPONENTS} components"
        )

# Save numerical results
results = pd.DataFrame({
    "component": np.arange(1, N_COMPONENTS + 1),
    "singular_value": svd.singular_values_,
    "explained_variance_ratio": explained,
    "cumulative_explained_variance": cumulative
})
results.to_csv(OUT / "svd_results.csv", index=False)

np.save(OUT / "embedding.npy", Z)
np.save(OUT / "labels.npy", y_sample)
np.save(OUT / "sample_indices.npy", indices)

# Scree plot
plt.figure(figsize=(9, 5))
plt.plot(
    results["component"],
    results["explained_variance_ratio"]
)
plt.xlabel("Component")
plt.ylabel("Explained variance ratio")
plt.title("TruncatedSVD Scree Plot")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "scree_plot.png", dpi=300)
plt.close()

# Cumulative explained variance
plt.figure(figsize=(9, 5))
plt.plot(
    results["component"],
    cumulative
)

for threshold in [0.90, 0.95]:
    plt.axhline(
        threshold,
        linestyle="--",
        label=f"{threshold:.0%} threshold"
    )

plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance ratio")
plt.ylim(0, 1)
plt.title("Cumulative Explained Variance")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUT / "cumulative_variance.png",
    dpi=300
)
plt.close()

summary = {
    "seed": SEED,
    "samples": N_SAMPLES,
    "features": int(X_sample.shape[1]),
    "components": N_COMPONENTS,
    "cumulative_variance_100": float(cumulative[-1])
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nResults saved to:", OUT)
