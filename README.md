# High-Dimensional Geometry of 20 Newsgroups

Independent course project based on the *methodological outline* of the supplied Pan-Cancer dimensionality study. No measurements are copied from that study.

## Run

```bash
python -m pip install -r requirements.txt
python src/pipeline.py --samples 800 --features 5000 --svd 300 --seed 42
```

The first run downloads the public 20 Newsgroups corpus using scikit-learn. Internet access is required. Outputs are saved to `results/` (JSON, CSV, PNG). `--samples` is the sample count before dropping empty documents.

## Important methodological differences

- TF-IDF is sparse, nonnegative and L2-normalized; `log1p` of RNA-Seq must **not** be blindly copied.
- TruncatedSVD is used for scalable sparse reduction; subsequent centered PCA is run on the SVD representation. All geometry diagnostics currently operate on the **SVD representation**, not the full 130k-dimensional sparse ambient space.
- Euclidean distance is used in theorem-driven experiments; cosine similarity should be added as a separate text-domain sensitivity analysis.
- Marchenko–Pastur edge is valid only for independent, unit-variance data; the shuffled-column null is more informative for real text data.
- Correlation dimension on 800 points is unstable; the slope is descriptive, not a resolved intrinsic dimension.
- Mahalanobis distances are ranked without unjustified chi-square p-values.
- The code is an **initial reproducible pipeline**, not a completed empirical study. Additional sensitivity studies, bootstraps and careful interpretation are required before submission.

## Reproducibility

- Seeded document subsampling, synthetic baselines, embeddings, and random projections.
- Actual results are written only after the experiments run.
- Do not cite the original cancer report as a source of text-domain measurements.

## Sources

- scikit-learn 20 Newsgroups: https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
- Vectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Random projection: https://scikit-learn.org/stable/modules/random_projection.html
