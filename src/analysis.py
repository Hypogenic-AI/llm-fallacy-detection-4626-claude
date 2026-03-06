"""
Analysis script for LLM fallacy self-detection experiments.
Computes per-category metrics, statistical tests, and generates visualizations.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats

BASE_DIR = Path("/workspaces/llm-fallacy-detection-4626-claude")
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


def analyze_experiment1():
    """Analyze Attribution Bias experiment."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Experiment 1 - Attribution Bias")
    print("=" * 60)

    fpath = RESULTS_DIR / "exp1_attribution_gpt-4.1.json"
    if not fpath.exists():
        print("  Results file not found, skipping")
        return None

    with open(fpath) as f:
        results = json.load(f)

    # Overall detection rates
    self_det = [r["self_detected"] for r in results if r["self_detected"] is not None]
    other_det = [r["other_detected"] for r in results if r["other_detected"] is not None]
    neutral_det = [r["neutral_detected"] for r in results if r["neutral_detected"] is not None]

    # All examples are fallacious (from LOGIC dataset), so detection = TP
    self_rate = np.mean(self_det)
    other_rate = np.mean(other_det)
    neutral_rate = np.mean(neutral_det)

    print(f"\nOverall Detection Rates (all examples are fallacious):")
    print(f"  Self-attributed:    {self_rate:.3f} ({sum(self_det)}/{len(self_det)})")
    print(f"  Other-attributed:   {other_rate:.3f} ({sum(other_det)}/{len(other_det)})")
    print(f"  Neutral:            {neutral_rate:.3f} ({sum(neutral_det)}/{len(neutral_det)})")
    print(f"  Attribution bias:   {other_rate - self_rate:+.3f}")

    # McNemar's test: paired comparison of self vs other
    # Build contingency table
    paired = [(r["self_detected"], r["other_detected"]) for r in results
              if r["self_detected"] is not None and r["other_detected"] is not None]
    a = sum(1 for s, o in paired if s and o)       # both detect
    b = sum(1 for s, o in paired if s and not o)   # self detects, other misses
    c = sum(1 for s, o in paired if not s and o)   # self misses, other detects
    d = sum(1 for s, o in paired if not s and not o) # both miss

    print(f"\n  McNemar contingency: a={a}, b={b}, c={c}, d={d}")
    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 25 else None
        # Use exact binomial test for small samples
        p_value = stats.binom_test(min(b, c), b + c, 0.5) if hasattr(stats, 'binom_test') else \
                  stats.binomtest(min(b, c), b + c, 0.5).pvalue
        print(f"  McNemar test p-value: {p_value:.4f}")
    else:
        p_value = 1.0
        print("  McNemar test: no discordant pairs")

    # Per-category analysis
    by_cat = defaultdict(lambda: {"self": [], "other": [], "neutral": []})
    for r in results:
        cat = r["true_label"]
        if r["self_detected"] is not None:
            by_cat[cat]["self"].append(r["self_detected"])
        if r["other_detected"] is not None:
            by_cat[cat]["other"].append(r["other_detected"])
        if r["neutral_detected"] is not None:
            by_cat[cat]["neutral"].append(r["neutral_detected"])

    print(f"\nPer-Category Detection Rates:")
    print(f"{'Category':<25} {'Self':>8} {'Other':>8} {'Neutral':>8} {'Bias':>8}")
    print("-" * 65)

    cat_data = []
    for cat in sorted(by_cat.keys()):
        s = np.mean(by_cat[cat]["self"]) if by_cat[cat]["self"] else 0
        o = np.mean(by_cat[cat]["other"]) if by_cat[cat]["other"] else 0
        n = np.mean(by_cat[cat]["neutral"]) if by_cat[cat]["neutral"] else 0
        bias = o - s
        print(f"  {cat:<23} {s:>8.3f} {o:>8.3f} {n:>8.3f} {bias:>+8.3f}")
        cat_data.append({
            "category": cat,
            "self_rate": s,
            "other_rate": o,
            "neutral_rate": n,
            "bias": bias,
            "n_self": len(by_cat[cat]["self"]),
            "n_other": len(by_cat[cat]["other"]),
        })

    # Identify blindspots (>15% bias)
    blindspots = [c for c in cat_data if c["bias"] > 0.15]
    if blindspots:
        print(f"\nBlindspot Categories (bias > 15%):")
        for b in blindspots:
            print(f"  {b['category']}: {b['bias']:+.1%}")

    # Cohen's h effect size
    if self_rate > 0 and other_rate > 0:
        h = 2 * np.arcsin(np.sqrt(other_rate)) - 2 * np.arcsin(np.sqrt(self_rate))
        print(f"\nCohen's h effect size: {h:.3f}")

    # Wilson confidence intervals
    def wilson_ci(p, n, z=1.96):
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        spread = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
        return max(0, center - spread), min(1, center + spread)

    n_total = len(self_det)
    self_ci = wilson_ci(self_rate, n_total)
    other_ci = wilson_ci(other_rate, n_total)
    print(f"\n95% CI (Wilson):")
    print(f"  Self:  [{self_ci[0]:.3f}, {self_ci[1]:.3f}]")
    print(f"  Other: [{other_ci[0]:.3f}, {other_ci[1]:.3f}]")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Overall detection rates
    conditions = ["Self-attributed", "Other-attributed", "Neutral"]
    rates = [self_rate, other_rate, neutral_rate]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    bars = axes[0].bar(conditions, rates, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Detection Rate (Recall)")
    axes[0].set_title("Exp 1: Overall Fallacy Detection by Attribution")
    axes[0].set_ylim(0, 1.05)
    for bar, rate in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{rate:.1%}", ha="center", fontsize=10)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")

    # Plot 2: Per-category bias
    cats = [c["category"] for c in cat_data]
    biases = [c["bias"] for c in cat_data]
    sorted_idx = np.argsort(biases)
    cats_sorted = [cats[i] for i in sorted_idx]
    biases_sorted = [biases[i] for i in sorted_idx]
    bar_colors = ["#e74c3c" if b < -0.05 else "#2ecc71" if b > 0.05 else "#95a5a6" for b in biases_sorted]
    axes[1].barh(cats_sorted, biases_sorted, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Attribution Bias (Other - Self)")
    axes[1].set_title("Exp 1: Per-Category Attribution Bias")
    axes[1].axvline(x=0, color="black", linewidth=0.5)
    axes[1].axvline(x=0.15, color="red", linestyle="--", alpha=0.5, label=">15% blindspot")
    axes[1].axvline(x=-0.15, color="red", linestyle="--", alpha=0.5)

    # Plot 3: Self vs Other scatter
    self_rates = [c["self_rate"] for c in cat_data]
    other_rates = [c["other_rate"] for c in cat_data]
    axes[2].scatter(self_rates, other_rates, s=80, c="#3498db", edgecolor="black", zorder=5)
    for c in cat_data:
        axes[2].annotate(c["category"][:12], (c["self_rate"], c["other_rate"]),
                         fontsize=7, ha="center", va="bottom")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x (no bias)")
    axes[2].set_xlabel("Self-attributed Detection Rate")
    axes[2].set_ylabel("Other-attributed Detection Rate")
    axes[2].set_title("Exp 1: Self vs Other Detection by Category")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_attribution_bias.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {PLOTS_DIR / 'exp1_attribution_bias.png'}")

    return {
        "overall": {
            "self_rate": self_rate, "other_rate": other_rate, "neutral_rate": neutral_rate,
            "bias": other_rate - self_rate, "mcnemar_p": p_value,
            "n": len(self_det),
        },
        "per_category": cat_data,
        "blindspots": blindspots,
    }


def analyze_experiment2():
    """Analyze Cross-Model Evaluation experiment."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Experiment 2 - Cross-Model Evaluation")
    print("=" * 60)

    fpath = RESULTS_DIR / "exp2_crossmodel.json"
    if not fpath.exists():
        print("  Results file not found, skipping")
        return None

    with open(fpath) as f:
        all_results = json.load(f)

    # Compute detection rates for each gen-eval pair
    summary = {}
    for key, results in all_results.items():
        valid = [r for r in results if r["detected"] is not None]
        detected = sum(1 for r in valid if r["detected"])
        rate = detected / len(valid) if valid else 0

        parts = key.split("_gen_")
        gen_model = parts[0]
        eval_model = parts[1].replace("_eval", "")
        is_self = gen_model == eval_model

        summary[key] = {
            "generator": gen_model,
            "evaluator": eval_model,
            "is_self_eval": is_self,
            "detection_rate": rate,
            "n_detected": detected,
            "n_total": len(valid),
        }

    print(f"\nCross-Model Detection Matrix:")
    print(f"{'Generator':<20} {'Evaluator':<20} {'Self?':>6} {'Rate':>8} {'N':>5}")
    print("-" * 65)
    for key, s in summary.items():
        marker = " *" if s["is_self_eval"] else ""
        print(f"  {s['generator']:<18} {s['evaluator']:<18} {str(s['is_self_eval']):>6} "
              f"{s['detection_rate']:>8.3f} {s['n_total']:>5}{marker}")

    # Self vs cross comparison
    self_rates = [s["detection_rate"] for s in summary.values() if s["is_self_eval"]]
    cross_rates = [s["detection_rate"] for s in summary.values() if not s["is_self_eval"]]
    if self_rates and cross_rates:
        print(f"\nSelf-eval avg:  {np.mean(self_rates):.3f}")
        print(f"Cross-eval avg: {np.mean(cross_rates):.3f}")
        print(f"Difference:     {np.mean(cross_rates) - np.mean(self_rates):+.3f}")

    # Per-category analysis for self vs cross
    by_cat_self = defaultdict(list)
    by_cat_cross = defaultdict(list)
    for key, results in all_results.items():
        is_self = "gpt-4.1_gen_gpt-4.1_eval" in key or "gpt-4o-mini_gen_gpt-4o-mini_eval" in key
        for r in results:
            if r["detected"] is not None:
                cat = r["true_label"]
                if is_self:
                    by_cat_self[cat].append(r["detected"])
                else:
                    by_cat_cross[cat].append(r["detected"])

    if by_cat_self and by_cat_cross:
        print(f"\nPer-Category Self vs Cross (Exp 2):")
        print(f"{'Category':<25} {'Self':>8} {'Cross':>8} {'Bias':>8}")
        print("-" * 55)
        cat_data_2 = []
        for cat in sorted(set(list(by_cat_self.keys()) + list(by_cat_cross.keys()))):
            s = np.mean(by_cat_self[cat]) if by_cat_self[cat] else 0
            c = np.mean(by_cat_cross[cat]) if by_cat_cross[cat] else 0
            bias = c - s
            print(f"  {cat:<23} {s:>8.3f} {c:>8.3f} {bias:>+8.3f}")
            cat_data_2.append({"category": cat, "self_rate": s, "cross_rate": c, "bias": bias})

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of detection matrix
    models = sorted(set(s["generator"] for s in summary.values()))
    matrix = np.zeros((len(models), len(models)))
    for s in summary.values():
        i = models.index(s["generator"])
        j = models.index(s["evaluator"])
        matrix[i][j] = s["detection_rate"]

    short_labels = [m.replace("gpt-4.1", "GPT-4.1").replace("gpt-4o-mini", "GPT-4o-mini") for m in models]
    sns.heatmap(matrix, annot=True, fmt=".3f", xticklabels=short_labels, yticklabels=short_labels,
                cmap="RdYlGn", vmin=0, vmax=1, ax=axes[0])
    axes[0].set_xlabel("Evaluator Model")
    axes[0].set_ylabel("Generator Model")
    axes[0].set_title("Exp 2: Cross-Model Detection Rate Matrix")

    # Self vs Cross bar chart
    if self_rates and cross_rates:
        conditions = ["Self-Evaluation", "Cross-Evaluation"]
        means = [np.mean(self_rates), np.mean(cross_rates)]
        bars = axes[1].bar(conditions, means, color=["#e74c3c", "#2ecc71"],
                           edgecolor="black", linewidth=0.5)
        axes[1].set_ylabel("Detection Rate")
        axes[1].set_title("Exp 2: Self vs Cross Evaluation")
        axes[1].set_ylim(0, 1.05)
        for bar, m in zip(bars, means):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{m:.1%}", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_crossmodel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {PLOTS_DIR / 'exp2_crossmodel.png'}")

    return summary


def analyze_experiment3():
    """Analyze Prompting Intervention experiment."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Experiment 3 - Prompting Intervention")
    print("=" * 60)

    fpath = RESULTS_DIR / "exp3_intervention_gpt-4.1.json"
    if not fpath.exists():
        print("  Results file not found, skipping")
        return None

    with open(fpath) as f:
        int_results = json.load(f)

    # Also load Exp 1 for comparison (self-attributed without intervention)
    exp1_path = RESULTS_DIR / "exp1_attribution_gpt-4.1.json"
    if not exp1_path.exists():
        print("  Exp 1 results not found for comparison")
        return None

    with open(exp1_path) as f:
        exp1_results = json.load(f)

    # Match samples by text (first 80)
    exp1_self = {r["text"]: r["self_detected"] for r in exp1_results[:80]}
    int_det = {r["text"]: r["intervention_detected"] for r in int_results}

    matched = []
    for text in exp1_self:
        if text in int_det and exp1_self[text] is not None and int_det[text] is not None:
            matched.append((exp1_self[text], int_det[text]))

    if not matched:
        print("  No matched samples found")
        return None

    base_rate = np.mean([m[0] for m in matched])
    int_rate = np.mean([m[1] for m in matched])
    print(f"\nSelf-attributed (no intervention): {base_rate:.3f}")
    print(f"Self-attributed (with intervention): {int_rate:.3f}")
    print(f"Improvement: {int_rate - base_rate:+.3f}")

    # McNemar on matched pairs
    b = sum(1 for s, i in matched if s and not i)
    c = sum(1 for s, i in matched if not s and i)
    if b + c > 0:
        p_val = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        print(f"McNemar p-value: {p_val:.4f}")
    else:
        p_val = 1.0

    # Per-category for intervention
    by_cat_int = defaultdict(list)
    by_cat_base = defaultdict(list)
    for r in int_results:
        if r["intervention_detected"] is not None:
            by_cat_int[r["true_label"]].append(r["intervention_detected"])
    for r in exp1_results[:80]:
        if r["self_detected"] is not None:
            by_cat_base[r["true_label"]].append(r["self_detected"])

    print(f"\nPer-Category Intervention Effect:")
    print(f"{'Category':<25} {'Base':>8} {'Interv':>8} {'Change':>8}")
    print("-" * 55)
    for cat in sorted(by_cat_int.keys()):
        b_rate = np.mean(by_cat_base.get(cat, [0]))
        i_rate = np.mean(by_cat_int[cat])
        print(f"  {cat:<23} {b_rate:>8.3f} {i_rate:>8.3f} {i_rate - b_rate:>+8.3f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = ["Self (No Intervention)", "Self (With Intervention)"]
    rates = [base_rate, int_rate]
    colors = ["#e74c3c", "#f39c12"]
    bars = ax.bar(conditions, rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Detection Rate")
    ax.set_title("Exp 3: Effect of Explicit Fallacy-Checking Prompt")
    ax.set_ylim(0, 1.05)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp3_intervention.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {PLOTS_DIR / 'exp3_intervention.png'}")

    return {
        "base_rate": base_rate,
        "intervention_rate": int_rate,
        "improvement": int_rate - base_rate,
        "mcnemar_p": p_val,
        "n_matched": len(matched),
    }


def create_summary_visualization(exp1, exp2, exp3):
    """Create a combined summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Overall rates across conditions
    conditions = []
    rates = []
    colors = []
    if exp1:
        conditions.extend(["Self\n(Exp1)", "Other\n(Exp1)", "Neutral\n(Exp1)"])
        rates.extend([exp1["overall"]["self_rate"], exp1["overall"]["other_rate"],
                       exp1["overall"]["neutral_rate"]])
        colors.extend(["#e74c3c", "#2ecc71", "#3498db"])
    if exp3:
        conditions.append("Self+Interv\n(Exp3)")
        rates.append(exp3["intervention_rate"])
        colors.append("#f39c12")

    if conditions:
        bars = axes[0, 0].bar(conditions, rates, color=colors, edgecolor="black", linewidth=0.5)
        axes[0, 0].set_ylabel("Detection Rate")
        axes[0, 0].set_title("Summary: Detection Rates Across Conditions")
        axes[0, 0].set_ylim(0, 1.05)
        for bar, rate in zip(bars, rates):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"{rate:.1%}", ha="center", fontsize=9)

    # Top-right: Per-category bias heatmap
    if exp1 and exp1["per_category"]:
        cat_df = pd.DataFrame(exp1["per_category"])
        cat_df = cat_df.sort_values("bias", ascending=True)
        vals = cat_df[["self_rate", "other_rate", "neutral_rate"]].values
        sns.heatmap(vals, annot=True, fmt=".2f", cmap="RdYlGn",
                    yticklabels=cat_df["category"].values,
                    xticklabels=["Self", "Other", "Neutral"],
                    ax=axes[0, 1], vmin=0, vmax=1)
        axes[0, 1].set_title("Per-Category Detection Rates (Exp 1)")

    # Bottom-left: Bias distribution
    if exp1 and exp1["per_category"]:
        biases = [c["bias"] for c in exp1["per_category"]]
        axes[1, 0].hist(biases, bins=10, color="#3498db", edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(x=0, color="black", linewidth=1)
        axes[1, 0].axvline(x=np.mean(biases), color="red", linestyle="--",
                           label=f"Mean={np.mean(biases):.3f}")
        axes[1, 0].set_xlabel("Attribution Bias (Other - Self)")
        axes[1, 0].set_ylabel("Number of Categories")
        axes[1, 0].set_title("Distribution of Per-Category Attribution Bias")
        axes[1, 0].legend()

    # Bottom-right: Summary statistics text
    axes[1, 1].axis("off")
    summary_text = "KEY FINDINGS\n" + "=" * 40 + "\n\n"
    if exp1:
        summary_text += f"Exp 1: Attribution Bias\n"
        summary_text += f"  Self detection rate:  {exp1['overall']['self_rate']:.1%}\n"
        summary_text += f"  Other detection rate: {exp1['overall']['other_rate']:.1%}\n"
        summary_text += f"  Bias: {exp1['overall']['bias']:+.1%}\n"
        summary_text += f"  McNemar p = {exp1['overall']['mcnemar_p']:.4f}\n"
        summary_text += f"  N = {exp1['overall']['n']}\n\n"
    if exp2:
        self_r = np.mean([s["detection_rate"] for s in exp2.values() if s["is_self_eval"]])
        cross_r = np.mean([s["detection_rate"] for s in exp2.values() if not s["is_self_eval"]])
        summary_text += f"Exp 2: Cross-Model Evaluation\n"
        summary_text += f"  Self-eval avg:  {self_r:.1%}\n"
        summary_text += f"  Cross-eval avg: {cross_r:.1%}\n"
        summary_text += f"  Difference: {cross_r - self_r:+.1%}\n\n"
    if exp3:
        summary_text += f"Exp 3: Prompting Intervention\n"
        summary_text += f"  Without intervention: {exp3['base_rate']:.1%}\n"
        summary_text += f"  With intervention:    {exp3['intervention_rate']:.1%}\n"
        summary_text += f"  Improvement: {exp3['improvement']:+.1%}\n"
        summary_text += f"  McNemar p = {exp3['mcnemar_p']:.4f}\n"

    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("LLM Self-Detection of Logical Fallacies: Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSummary plot saved to {PLOTS_DIR / 'summary.png'}")


def main():
    print("=" * 60)
    print("ANALYSIS: LLM Fallacy Self-Detection Experiments")
    print("=" * 60)

    exp1 = analyze_experiment1()
    exp2 = analyze_experiment2()
    exp3 = analyze_experiment3()

    # Summary visualization
    create_summary_visualization(exp1, exp2, exp3)

    # Save combined analysis
    analysis_results = {
        "exp1_attribution": exp1,
        "exp2_crossmodel": {k: v for k, v in (exp2 or {}).items()},
        "exp3_intervention": exp3,
    }
    with open(RESULTS_DIR / "analysis_summary.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)

    print("\nAnalysis complete. Results saved to results/")


if __name__ == "__main__":
    main()
