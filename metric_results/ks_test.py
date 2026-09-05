import os

import numpy as np
import pandas as pd
from scipy import stats

# paths
REAL_DATA_PATH = "../sip_data/res1/combined_data.csv"
SYNTHETIC_DIR = "./synthetic_data(50)"
OUTPUT_PATH = "./ks_test_results.csv"

# significance levels to report
ALPHA_LEVELS = [0.05, 0.01]

CONDITIONS = {
    "asthma": "asthma",
    "bronchi": "bronchi",
    "copd": "copd",
}

# load data
print("=" * 65)
print("  Kolmogorov-Smirnov Test: Real vs Synthetic VOC Data")
print("=" * 65)

real_df = pd.read_csv(REAL_DATA_PATH)
voc_features = real_df.columns[2:].tolist()  # drop Patient ID, Disease

print(f"\nReal data     : {real_df.shape[0]} samples, {len(voc_features)} VOC features")
print(f"Conditions    : {list(CONDITIONS.keys())}")
print(f"Real samples per condition:")
for cond, count in real_df["Disease"].value_counts().items():
    print(f"  {cond:<10} {count}")

# run KS tests
all_results = []

for condition, label in CONDITIONS.items():
    syn_path = os.path.join(SYNTHETIC_DIR, f"synthetic_data_{condition}.csv")
    syn_df = pd.read_csv(syn_path)

    # filter real data to this condition
    real_cond = real_df[real_df["Disease"] == condition][voc_features]
    syn_cond = syn_df[voc_features]

    print(f"\n{'─' * 65}")
    print(f"  Condition : {condition.upper()}")
    print(f"  Real      : {len(real_cond)} samples")
    print(f"  Synthetic : {len(syn_cond)} samples")
    print(f"{'─' * 65}")
    print(
        f"  {'Feature':<15} {'KS Stat':>10} {'p-value':>12} {'Pass α=0.05':>12} {'Pass α=0.01':>12}"
    )
    print(
        f"  {'─' * 14:<15} {'─' * 9:>10} {'─' * 11:>12} {'─' * 10:>12} {'─' * 10:>12}"
    )

    for feature in voc_features:
        real_vals = real_cond[feature].dropna().values
        syn_vals = syn_cond[feature].dropna().values

        ks_stat, p_value = stats.ks_2samp(real_vals, syn_vals)

        row = {
            "condition": condition,
            "feature": feature,
            "ks_stat": round(ks_stat, 4),
            "p_value": round(p_value, 6),
        }
        for alpha in ALPHA_LEVELS:
            # pass = distributions are NOT significantly different (p > alpha)
            row[f"pass_a{int(alpha * 100):02d}"] = p_value > alpha

        all_results.append(row)

        pass_05 = "✓ PASS" if p_value > 0.05 else "✗ FAIL"
        pass_01 = "✓ PASS" if p_value > 0.01 else "✗ FAIL"
        print(
            f"  {str(feature):<15} {ks_stat:>10.4f} {p_value:>12.6f} {pass_05:>12} {pass_01:>12}"
        )

# ── per-condition summary ─────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)

print(f"\n{'=' * 65}")
print("  Per-Condition Summary")
print(f"{'=' * 65}")
print(
    f"  {'Condition':<10} {'Avg KS':>9} {'Avg p':>10} {'% Pass α=0.05':>15} {'% Pass α=0.01':>15}"
)
print(f"  {'─' * 9:<10} {'─' * 8:>9} {'─' * 9:>10} {'─' * 13:>15} {'─' * 13:>15}")

for condition in CONDITIONS:
    cond_df = results_df[results_df["condition"] == condition]
    avg_ks = cond_df["ks_stat"].mean()
    avg_p = cond_df["p_value"].mean()
    pct_05 = cond_df["pass_a05"].mean() * 100
    pct_01 = cond_df["pass_a01"].mean() * 100
    print(
        f"  {condition:<10} {avg_ks:>9.4f} {avg_p:>10.6f} {pct_05:>14.1f}% {pct_01:>14.1f}%"
    )

# ── overall summary ───────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("  Overall Summary (all conditions & features)")
print(f"{'=' * 65}")
print(f"  Total feature tests  : {len(results_df)}")
print(f"  Avg KS statistic     : {results_df['ks_stat'].mean():.4f}")
print(f"  Avg p-value          : {results_df['p_value'].mean():.6f}")
print(f"  Median p-value       : {results_df['p_value'].median():.6f}")

for alpha in ALPHA_LEVELS:
    col = f"pass_a{int(alpha * 100):02d}"
    n_pass = results_df[col].sum()
    n_total = len(results_df)
    print(
        f"  Pass rate (α={alpha}) : {n_pass}/{n_total}  ({n_pass / n_total * 100:.1f}%)"
    )

# ── worst & best features (across all conditions) ─────────────────────────────
print(f"\n{'=' * 65}")
print("  Top 5 WORST features (highest KS statistic = most different)")
print(f"{'=' * 65}")
worst = results_df.nlargest(5, "ks_stat")[
    ["condition", "feature", "ks_stat", "p_value"]
]
print(worst.to_string(index=False))

print(f"\n{'=' * 65}")
print("  Top 5 BEST features (lowest KS statistic = most similar)")
print(f"{'=' * 65}")
best = results_df.nsmallest(5, "ks_stat")[
    ["condition", "feature", "ks_stat", "p_value"]
]
print(best.to_string(index=False))

# ── save full results ─────────────────────────────────────────────────────────
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Full results saved to: {OUTPUT_PATH}")
print(f"{'=' * 65}\n")
