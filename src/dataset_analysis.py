
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results" / "dataset_analysis"
OUT.mkdir(parents=True, exist_ok=True)

X = load_npz(DATA / "X.npz").tocsr()
y = np.load(DATA / "y.npy")

# Remove explicit stored zeros, if any
X.eliminate_zeros()

# Number of nonzero features per document
features_per_document = np.diff(X.indptr)

# Number of documents containing each feature
feature_frequency = np.bincount(
    X.indices, minlength=X.shape[1]
)

print("=== DOCUMENT STATISTICS ===")
print("Mean:", features_per_document.mean())
print("Median:", np.median(features_per_document))
print("Min:", features_per_document.min())
print("Max:", features_per_document.max())

print("\n=== FEATURE STATISTICS ===")
print("Unused features:", np.sum(feature_frequency == 0))
print("Features appearing once:", np.sum(feature_frequency == 1))
print("Features appearing in <= 5 documents:",
      np.sum(feature_frequency <= 5))

# Figure 1: nonzero features per document
plt.figure(figsize=(9, 5))
plt.hist(features_per_document, bins=80)
plt.xlabel("Nonzero features per document")
plt.ylabel("Number of documents")
plt.title("Document Feature Density")
plt.tight_layout()
plt.savefig(OUT / "document_density.png", dpi=300)
plt.close()

# Figure 2: document frequency distribution
plt.figure(figsize=(9, 5))
plt.hist(feature_frequency, bins=80, log=True)
plt.xlabel("Number of documents containing a feature")
plt.ylabel("Number of features (log scale)")
plt.title("Feature Document Frequency")
plt.tight_layout()
plt.savefig(OUT / "feature_frequency.png", dpi=300)
plt.close()

# Figure 3: rank-frequency plot
sorted_frequency = np.sort(feature_frequency)[::-1]
positive = sorted_frequency[sorted_frequency > 0]

plt.figure(figsize=(9, 5))
plt.loglog(
    np.arange(1, len(positive) + 1),
    positive
)
plt.xlabel("Feature rank")
plt.ylabel("Document frequency")
plt.title("Feature Rank-Frequency Distribution")
plt.tight_layout()
plt.savefig(OUT / "rank_frequency.png", dpi=300)
plt.close()

print("\nFigures saved to:", OUT)
