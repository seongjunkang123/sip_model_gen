import os
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
REAL_DATA_PATH = "../sip_data/res1/combined_data.csv"
SYNTHETIC_DIR = "./synthetic_data(50)"
OUTPUT_PATH = "./nn_test_results.csv"

CONDITIONS = {
    "asthma": "asthma",
    "bronchi": "bronchi",
    "copd": "copd",
}

def euclidean_distances(A, B):
    # Compute pairwise Euclidean distances between rows of A and B.
    A2 = np.sum(A ** 2, axis=1, keepdims=True)
    B2 = np.sum(B ** 2, axis=1, keepdims=True)
    AB = A @ B.T
    dists_sq = A2 + B2.T - 2 * AB
    # clamp numerical noise
    dists_sq = np.maximum(dists_sq, 0.0)
    return np.sqrt(dists_sq)

def nn_adversarial_accuracy(real, synthetic):
    n_real = len(real)
    n_syn = len(synthetic)
    combined = np.vstack([real, synthetic])
    labels = np.array([0] * n_real + [1] * n_syn)  # 0=real, 1=synthetic

    dists = euclidean_distances(combined, combined)
    # set self-distance to infinity
    np.fill_diagonal(dists, np.inf)

    # nearest neighbor index for each sample
    nn_idx = np.argmin(dists, axis=1)
    nn_labels = labels[nn_idx]

    # calculate accuracy
    correct = (labels == nn_labels).sum()
    accuracy = correct / len(labels)

    # break down by class
    real_mask = labels == 0
    syn_mask = labels == 1

    syn_nn_syn = (nn_labels[syn_mask] == 1).sum()
    precision_proxy = syn_nn_syn / n_syn

    real_nn_syn = (nn_labels[real_mask] == 1).sum()
    recall_proxy = real_nn_syn / n_real

    return accuracy, precision_proxy, recall_proxy


def distance_to_closest_record(real, synthetic):
    dists = euclidean_distances(synthetic, real)
    min_dists = np.min(dists, axis=1)

    median_dcr = np.median(min_dists)
    p5_dcr = np.percentile(min_dists, 5)
    mean_dcr = np.mean(min_dists)

    # also compute real-to-real NN distances for reference
    real_dists = euclidean_distances(real, real)
    np.fill_diagonal(real_dists, np.inf)
    real_nn_dists = np.min(real_dists, axis=1)
    median_real_nn = np.median(real_nn_dists)

    # "copy" threshold: if a synthetic sample is closer to its nearest real
    copy_fraction = np.mean(min_dists < median_real_nn * 0.5)

    return {
        "median_dcr": median_dcr,
        "p5_dcr": p5_dcr,
        "mean_dcr": mean_dcr,
        "median_real_nn": median_real_nn,
        "copy_fraction": copy_fraction,
    }


# ── load data ────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Nearest Neighbor Tests: Real vs Synthetic VOC Data")
print("=" * 70)

real_df = pd.read_csv(REAL_DATA_PATH)
voc_features = real_df.columns[2:].tolist()

print(f"\nReal data     : {real_df.shape[0]} samples, {len(voc_features)} VOC features")

# normalize for distance computation: log1p, minmax
all_real_vals = real_df[voc_features].values.astype(np.float64)
all_real_log = np.log1p(all_real_vals)

# fit min-max on all real data
data_min = all_real_log.min(axis=0)
data_max = all_real_log.max(axis=0)
data_range = data_max - data_min
data_range[data_range == 0] = 1.0  # avoid division by zero

def normalize(data):
    return (np.log1p(data.astype(np.float64)) - data_min) / data_range

# ── run tests ────────────────────────────────────────────────────────────────
all_results = []

for condition, label in CONDITIONS.items():
    syn_path = os.path.join(SYNTHETIC_DIR, f"synthetic_data_{condition}.csv")
    syn_df = pd.read_csv(syn_path)

    real_cond = real_df[real_df["Disease"] == condition][voc_features].values
    syn_cond = syn_df[voc_features].values

    # normalize
    real_norm = normalize(real_cond)
    syn_norm = normalize(syn_cond)

    # NNAA
    nnaa, prec_proxy, recall_proxy = nn_adversarial_accuracy(real_norm, syn_norm)

    # DCR
    dcr = distance_to_closest_record(real_norm, syn_norm)

    print(f"\n{'─' * 70}")
    print(f"  Condition: {condition.upper()}")
    print(f"  Real: {len(real_cond)} samples | Synthetic: {len(syn_cond)} samples")
    print(f"{'─' * 70}")
    print(f"  NN Adversarial Accuracy : {nnaa:.4f}  (ideal ≈ 0.50)")
    print(f"  Precision proxy         : {prec_proxy:.4f}  (lower = more overlap = better)")
    print(f"  Recall proxy            : {recall_proxy:.4f}  (higher = better coverage)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Median DCR (syn→real)   : {dcr['median_dcr']:.4f}")
    print(f"  5th %ile DCR            : {dcr['p5_dcr']:.4f}")
    print(f"  Mean DCR                : {dcr['mean_dcr']:.4f}")
    print(f"  Median real→real NN dist: {dcr['median_real_nn']:.4f}")
    print(f"  Copy fraction (<50% of  : {dcr['copy_fraction']:.4f}  (ideal ≈ 0.00)")
    print(f"   median real NN dist)   ")

    result = {
        "condition": condition,
        "n_real": len(real_cond),
        "n_synthetic": len(syn_cond),
        "nnaa": round(nnaa, 4),
        "precision_proxy": round(prec_proxy, 4),
        "recall_proxy": round(recall_proxy, 4),
        "median_dcr": round(dcr["median_dcr"], 4),
        "p5_dcr": round(dcr["p5_dcr"], 4),
        "mean_dcr": round(dcr["mean_dcr"], 4),
        "median_real_nn": round(dcr["median_real_nn"], 4),
        "copy_fraction": round(dcr["copy_fraction"], 4),
    }
    all_results.append(result)

# ── overall summary ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)

print(f"\n{'=' * 70}")
print("  Overall Summary")
print(f"{'=' * 70}")
print(f"  Avg NNAA              : {results_df['nnaa'].mean():.4f}  (ideal ≈ 0.50)")
print(f"  Avg Precision proxy   : {results_df['precision_proxy'].mean():.4f}")
print(f"  Avg Recall proxy      : {results_df['recall_proxy'].mean():.4f}")
print(f"  Avg Median DCR        : {results_df['median_dcr'].mean():.4f}")
print(f"  Avg Copy fraction     : {results_df['copy_fraction'].mean():.4f}")

quality = results_df["nnaa"].mean()
if quality < 0.55:
    verdict = "EXCELLENT — synthetic data is nearly indistinguishable from real"
elif quality < 0.65:
    verdict = "GOOD — synthetic data closely resembles real data"
elif quality < 0.75:
    verdict = "FAIR — some distributional differences remain"
else:
    verdict = "POOR — classifier can easily distinguish real from synthetic"

print(f"\n  Verdict: {verdict}")

# ── save ─────────────────────────────────────────────────────────────────────
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Results saved to: {OUTPUT_PATH}")
print(f"{'=' * 70}\n")
