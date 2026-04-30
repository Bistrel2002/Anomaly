"""Visualize data drift detection across simulated scenarios.

Generates publication-quality charts that demonstrate how the KS-test based
drift detector behaves under three conditions:

    1. **No drift** — live data drawn from the same distribution as training.
    2. **Gradual drift** — live mean shifts slowly over successive windows.
    3. **Sudden drift** — live data drawn from a completely different
       distribution.

All plots are saved to ``<project_root>/drift_output/``.

Usage:
    PYTHONPATH=. python scripts/visualize_drift.py
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Ensure project root is on sys.path so ``app`` package is importable.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from app.drift import FEATURES, DriftDetector, P_VALUE_THRESHOLD  # noqa: E402

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "drift_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_temp_baseline(rng: np.random.Generator, n: int = 2000) -> dict:
    """Create a synthetic baseline dict (mean≈0, std≈1 per feature)."""
    return {f: rng.normal(0, 1, n) for f in FEATURES}


def _records_from_arrays(arrays: dict[str, np.ndarray]) -> list[dict]:
    """Convert {feature: array} → list of record dicts."""
    n = min(len(v) for v in arrays.values())
    return [{f: float(arrays[f][i]) for f in FEATURES} for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Distribution comparison (KDE plots)
# ---------------------------------------------------------------------------

def plot_distribution_comparison(
    baseline: dict,
    live_no_drift: dict,
    live_drifted: dict,
) -> str:
    """KDE overlay: baseline vs no-drift vs drifted for every feature."""
    n_features = len(FEATURES)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(FEATURES):
        ax = axes[idx]
        sns.kdeplot(baseline[feat], ax=ax, label="Training baseline",
                    color="#3498db", fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(live_no_drift[feat], ax=ax, label="Live (no drift)",
                    color="#2ecc71", fill=True, alpha=0.3, linewidth=2)
        sns.kdeplot(live_drifted[feat], ax=ax, label="Live (drifted)",
                    color="#e74c3c", fill=True, alpha=0.3, linewidth=2)
        ax.set_title(feat)
        ax.legend(fontsize=8)

    # Hide the extra subplot (7 features → 8 subplots).
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Distributions — Baseline vs Live Data", fontsize=16, y=1.02)
    path = os.path.join(OUTPUT_DIR, "drift_distribution_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 2. P-value heatmap across scenarios
# ---------------------------------------------------------------------------

def plot_pvalue_heatmap(results: dict[str, dict]) -> str:
    """Heatmap of p-values (features × scenarios)."""
    scenarios = list(results.keys())
    data = np.zeros((len(FEATURES), len(scenarios)))

    for j, scenario in enumerate(scenarios):
        for i, feat in enumerate(FEATURES):
            data[i, j] = results[scenario]["results"][feat]["p_value"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".4f",
        xticklabels=scenarios,
        yticklabels=FEATURES,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("KS Test p-values per Feature & Scenario")
    ax.axvline(x=0, color="black", linewidth=2)

    # Draw the threshold line annotation.
    ax.text(
        len(scenarios) + 0.3, len(FEATURES) / 2,
        f"Threshold = {P_VALUE_THRESHOLD}",
        fontsize=10, rotation=90, va="center", color="#e74c3c", fontweight="bold",
    )

    path = os.path.join(OUTPUT_DIR, "drift_pvalue_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Gradual drift timeline
# ---------------------------------------------------------------------------

def plot_gradual_drift_timeline(
    baseline: dict,
    detector: DriftDetector,
    n_windows: int = 20,
    window_size: int = 150,
) -> str:
    """Simulate gradual drift and plot p-value evolution over time."""
    rng = np.random.Generator(np.random.PCG64(123))

    # Mean shifts linearly from 0 to 3 over n_windows.
    mean_shifts = np.linspace(0, 3, n_windows)

    timeline: dict[str, list[float]] = {f: [] for f in FEATURES}
    drift_flags: list[bool] = []

    for shift in mean_shifts:
        live = {f: rng.normal(shift, 1, window_size) for f in FEATURES}
        records = _records_from_arrays(live)
        result = detector.compute_drift(records)
        drift_flags.append(result["drift_detected"])
        for f in FEATURES:
            timeline[f].append(result["results"][f]["p_value"])

    fig, ax = plt.subplots(figsize=(14, 7))
    for feat in FEATURES:
        ax.plot(mean_shifts, timeline[feat], marker="o", markersize=4, label=feat)

    ax.axhline(y=P_VALUE_THRESHOLD, color="#e74c3c", linestyle="--",
               linewidth=2, label=f"Threshold ({P_VALUE_THRESHOLD})")

    # Shade drift-detected windows.
    for i, (shift, drifted) in enumerate(zip(mean_shifts, drift_flags)):
        if drifted:
            ax.axvspan(
                shift - (mean_shifts[1] - mean_shifts[0]) / 2,
                shift + (mean_shifts[1] - mean_shifts[0]) / 2,
                alpha=0.08, color="red",
            )

    ax.set_xlabel("Mean Shift Applied to Live Data")
    ax.set_ylabel("KS Test p-value")
    ax.set_title("Gradual Drift Timeline — p-value vs Increasing Mean Shift")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    path = os.path.join(OUTPUT_DIR, "drift_gradual_timeline.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Summary bar chart (drift detected per feature)
# ---------------------------------------------------------------------------

def plot_summary_bar(results: dict[str, dict]) -> str:
    """Grouped bar chart: KS statistic per feature across scenarios."""
    scenarios = list(results.keys())
    x = np.arange(len(FEATURES))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for j, scenario in enumerate(scenarios):
        stats_vals = [
            results[scenario]["results"][f]["statistic"] for f in FEATURES
        ]
        bars = ax.bar(x + j * width, stats_vals, width, label=scenario,
                      color=colors[j % len(colors)], edgecolor="white")
        # Mark drifted bars.
        for i, bar in enumerate(bars):
            if results[scenario]["results"][FEATURES[i]]["drift_detected"]:
                bar.set_edgecolor("#e74c3c")
                bar.set_linewidth(2)

    ax.set_xlabel("Feature")
    ax.set_ylabel("KS Statistic")
    ax.set_title("KS Statistic per Feature — Red border = drift detected")
    ax.set_xticks(x + width)
    ax.set_xticklabels(FEATURES)
    ax.legend()

    path = os.path.join(OUTPUT_DIR, "drift_ks_statistic_bars.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all drift visualizations."""
    print("=" * 60)
    print("  DRIFT DETECTION — VISUALIZATION & TESTING")
    print("=" * 60)

    rng = np.random.Generator(np.random.PCG64(42))

    # --- Build synthetic baseline & detector --------------------------------
    baseline = _build_temp_baseline(rng)
    baseline_path = os.path.join(OUTPUT_DIR, "_temp_baseline.pkl")
    joblib.dump(baseline, baseline_path)
    detector = DriftDetector(baseline_path=baseline_path)

    print(f"\n📦 Baseline: {len(baseline[FEATURES[0]])} samples, "
          f"{len(FEATURES)} features")

    # --- Scenario 1: No drift -----------------------------------------------
    print("\n🟢 Scenario 1 — No Drift (same distribution)")
    live_no_drift = {f: rng.normal(0, 1, 300) for f in FEATURES}
    records_no_drift = _records_from_arrays(live_no_drift)
    result_no_drift = detector.compute_drift(records_no_drift)
    print(f"   drift_detected: {result_no_drift['drift_detected']}")
    for f in FEATURES:
        r = result_no_drift["results"][f]
        status = "⚠️  DRIFT" if r["drift_detected"] else "✅ OK"
        print(f"   {f:>8s}  stat={r['statistic']:.4f}  "
              f"p={r['p_value']:.4f}  {status}")

    # --- Scenario 2: Gradual drift -------------------------------------------
    print("\n🟡 Scenario 2 — Gradual Drift (mean shifted +1.5)")
    live_gradual = {f: rng.normal(1.5, 1, 300) for f in FEATURES}
    records_gradual = _records_from_arrays(live_gradual)
    result_gradual = detector.compute_drift(records_gradual)
    print(f"   drift_detected: {result_gradual['drift_detected']}")
    for f in FEATURES:
        r = result_gradual["results"][f]
        status = "⚠️  DRIFT" if r["drift_detected"] else "✅ OK"
        print(f"   {f:>8s}  stat={r['statistic']:.4f}  "
              f"p={r['p_value']:.4f}  {status}")

    # --- Scenario 3: Sudden drift --------------------------------------------
    print("\n🔴 Scenario 3 — Sudden Drift (mean=10, std=0.5)")
    live_sudden = {f: rng.normal(10, 0.5, 300) for f in FEATURES}
    records_sudden = _records_from_arrays(live_sudden)
    result_sudden = detector.compute_drift(records_sudden)
    print(f"   drift_detected: {result_sudden['drift_detected']}")
    for f in FEATURES:
        r = result_sudden["results"][f]
        status = "⚠️  DRIFT" if r["drift_detected"] else "✅ OK"
        print(f"   {f:>8s}  stat={r['statistic']:.4f}  "
              f"p={r['p_value']:.4f}  {status}")

    # --- Plots ---------------------------------------------------------------
    print("\n📊 Generating plots…\n")

    all_results = {
        "No Drift": result_no_drift,
        "Gradual (+1.5σ)": result_gradual,
        "Sudden (+10σ)": result_sudden,
    }

    plot_distribution_comparison(baseline, live_no_drift, live_sudden)
    plot_pvalue_heatmap(all_results)
    plot_gradual_drift_timeline(baseline, detector)
    plot_summary_bar(all_results)

    # Cleanup temp baseline.
    os.remove(baseline_path)

    print(f"\n✅ All plots saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
