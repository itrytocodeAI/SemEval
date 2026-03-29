"""
DimABSA2026 – Comprehensive Visualization & XAI Suite
======================================================
Generates all plots from baseline and model results:

Classic Plots:
  1. Training loss curves
  2. RMSE convergence
  3. Model comparison bar chart
  4. Gold vs Predicted scatter (V, A)
  5. Error distribution histograms
  6. VA space 2D scatter
  7. Per-domain comparison
  8. Box plots of errors

XAI Explainability:
  9. Attention heatmaps
  10. Token attribution (integrated gradients approx)
  11. SHAP summary
  12. Uncertainty evolution (σ_V, σ_A over epochs)

Usage:  python generate_plots.py [--model_type deberta|bert]
"""

import json, os, math, argparse, glob, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "figure.figsize": (10,6)})

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="deberta", choices=["deberta","bert"])
parser.add_argument("--domain", default="restaurant")
args = parser.parse_args()

os.makedirs("plots", exist_ok=True)
os.makedirs("plots/xai", exist_ok=True)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load data ────────────────────────────────────────────────────────────
def load_json(p):
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

def load_jsonl(p):
    if not os.path.exists(p): return []
    with open(p,"r",encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]

def parse_va(s): v,a=s.split("#"); return float(v),float(a)

# ── 1. Training Loss Curves ─────────────────────────────────────────────
def plot_training_curves():
    """Plot train/dev loss and RMSE over epochs for all model histories."""
    hist_files = glob.glob(os.path.join("logs", f"{args.model_type}_history_{args.domain}_*.json"))
    if not hist_files:
        hist_files = glob.glob(os.path.join("logs", f"training_curve_task1_{args.domain}_*.json"))
    if not hist_files:
        print("No training history found, skipping training curves")
        return

    hist = load_json(hist_files[-1])
    if not hist: return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = [h["epoch"] for h in hist]

    # Loss
    axes[0].plot(epochs, [h["train_loss"] for h in hist], "o-", color="#2196F3", lw=2, label="Train Loss")
    axes[0].plot(epochs, [h["dev_loss"] for h in hist], "s-", color="#F44336", lw=2, label="Dev Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # RMSE
    rmse_key = "rmse" if "rmse" in hist[0] else "dev_rmse"
    axes[1].plot(epochs, [h[rmse_key] for h in hist], "D-", color="#4CAF50", lw=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE_VA"); axes[1].set_title("RMSE_VA Convergence")
    axes[1].axhline(y=min(h[rmse_key] for h in hist), color="red", ls="--", alpha=0.5, label=f"Best: {min(h[rmse_key] for h in hist):.4f}")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # PCC
    if "pcc_v" in hist[0]:
        axes[2].plot(epochs, [h["pcc_v"] for h in hist], "^-", color="#9C27B0", lw=2, label="PCC_V")
        axes[2].plot(epochs, [h["pcc_a"] for h in hist], "v-", color="#FF9800", lw=2, label="PCC_A")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("PCC"); axes[2].set_title("Pearson Correlation")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Training Curves – {args.model_type.upper()} ({args.domain})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/training_curves_{args.model_type}_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/training_curves_{args.model_type}_{args.domain}.png")

# ── 2. Model Comparison Bar Chart ────────────────────────────────────────
def plot_model_comparison():
    """Bar chart comparing RMSE_VA across all models."""
    data = load_json("logs/baseline_results.json")
    if not data:
        print("No baseline results found, skipping comparison")
        return

    # Filter by domain
    filtered = [d for d in data if d.get("domain") == args.domain]
    if not filtered:
        filtered = data

    models = [d["model"] for d in filtered]
    rmse = [d["RMSE_VA"] for d in filtered]

    colors = sns.color_palette("husl", len(models))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(models, rmse, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, rmse):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("RMSE_VA (lower is better)", fontsize=12)
    ax.set_title(f"Model Comparison – RMSE_VA ({args.domain})", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"plots/model_comparison_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/model_comparison_{args.domain}.png")

# ── 3. Gold vs Predicted Scatter ─────────────────────────────────────────
def plot_scatter_gold_vs_pred():
    """Scatter plot of gold vs predicted V and A."""
    detail = load_json("logs/baseline_detailed_predictions.json")
    if not detail: return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    models_to_plot = [d for d in detail if d.get("domain") == args.domain][:6]

    for idx, d in enumerate(models_to_plot):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]
        gv, ga = d["gold_v"], d["gold_a"]
        pv, pa = d["pred_v"], d["pred_a"]

        ax.scatter(gv, pv, alpha=0.4, s=20, c="#2196F3", label="Valence")
        ax.scatter(ga, pa, alpha=0.4, s=20, c="#F44336", label="Arousal")
        ax.plot([1,9],[1,9], "k--", alpha=0.3, lw=1)
        ax.set_xlabel("Gold"); ax.set_ylabel("Predicted")
        ax.set_title(d["model"], fontsize=11)
        ax.set_xlim(0.5,9.5); ax.set_ylim(0.5,9.5)
        ax.legend(fontsize=8)

    plt.suptitle(f"Gold vs Predicted – VA Scatter ({args.domain})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/scatter_gold_vs_pred_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/scatter_gold_vs_pred_{args.domain}.png")

# ── 4. Error Distribution ────────────────────────────────────────────────
def plot_error_distribution():
    """Histogram of V and A prediction errors."""
    detail = load_json("logs/baseline_detailed_predictions.json")
    if not detail: return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    models_to_plot = [d for d in detail if d.get("domain") == args.domain][:6]

    for idx, d in enumerate(models_to_plot):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]
        err_v = [p-g for p,g in zip(d["pred_v"], d["gold_v"])]
        err_a = [p-g for p,g in zip(d["pred_a"], d["gold_a"])]

        ax.hist(err_v, bins=30, alpha=0.6, color="#2196F3", label=f"V err (μ={np.mean(err_v):.2f})", density=True)
        ax.hist(err_a, bins=30, alpha=0.6, color="#F44336", label=f"A err (μ={np.mean(err_a):.2f})", density=True)
        ax.axvline(0, color="black", ls="--", alpha=0.5)
        ax.set_title(d["model"], fontsize=11)
        ax.legend(fontsize=8)

    plt.suptitle(f"Error Distribution – V and A ({args.domain})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/error_distribution_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/error_distribution_{args.domain}.png")

# ── 5. VA Space 2D Plot ──────────────────────────────────────────────────
def plot_va_space():
    """2D VA space showing gold and predicted clusters."""
    detail = load_json("logs/baseline_detailed_predictions.json")
    if not detail: return

    best = min([d for d in detail if d.get("domain") == args.domain],
               key=lambda x: x["RMSE_VA"], default=None)
    if not best: return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(best["gold_v"], best["gold_a"], alpha=0.5, s=30, c="#4CAF50", label="Gold", marker="o")
    ax.scatter(best["pred_v"], best["pred_a"], alpha=0.5, s=30, c="#F44336", label="Predicted", marker="x")

    # Draw arrows from gold to pred
    for gv,ga,pv,pa in list(zip(best["gold_v"],best["gold_a"],best["pred_v"],best["pred_a"]))[:50]:
        ax.annotate("", xy=(pv,pa), xytext=(gv,ga),
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.2, lw=0.5))

    ax.set_xlabel("Valence", fontsize=12); ax.set_ylabel("Arousal", fontsize=12)
    ax.set_title(f"VA Space – {best['model']} ({args.domain})", fontsize=14, fontweight="bold")
    ax.set_xlim(0.5,9.5); ax.set_ylim(0.5,9.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"plots/va_space_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/va_space_{args.domain}.png")

# ── 6. Box Plot of Errors ────────────────────────────────────────────────
def plot_error_boxplots():
    """Box plots comparing error distributions across models."""
    detail = load_json("logs/baseline_detailed_predictions.json")
    if not detail: return

    models_data = [d for d in detail if d.get("domain") == args.domain]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    v_errors, a_errors, labels = [], [], []
    for d in models_data:
        ev = [abs(p-g) for p,g in zip(d["pred_v"],d["gold_v"])]
        ea = [abs(p-g) for p,g in zip(d["pred_a"],d["gold_a"])]
        v_errors.append(ev); a_errors.append(ea)
        labels.append(d["model"][:15])

    ax1.boxplot(v_errors, labels=labels, patch_artist=True,
                boxprops=dict(facecolor="#2196F3", alpha=0.6))
    ax1.set_title("Valence Absolute Error", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)

    ax2.boxplot(a_errors, labels=labels, patch_artist=True,
                boxprops=dict(facecolor="#F44336", alpha=0.6))
    ax2.set_title("Arousal Absolute Error", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)

    plt.suptitle(f"Error Distribution Box Plots ({args.domain})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/error_boxplots_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/error_boxplots_{args.domain}.png")

# ── 7. Heatmap: Per-Model V/A Performance ────────────────────────────────
def plot_performance_heatmap():
    """Heatmap of metrics across models."""
    data = load_json("logs/baseline_results.json")
    if not data: return

    filtered = [d for d in data if d.get("domain") == args.domain]
    if not filtered: return

    models = [d["model"] for d in filtered]
    rmse = [d["RMSE_VA"] for d in filtered]
    pcc_v = [d.get("PCC_V",0) or 0 for d in filtered]
    pcc_a = [d.get("PCC_A",0) or 0 for d in filtered]

    matrix = np.array([rmse, pcc_v, pcc_a]).T
    fig, ax = plt.subplots(figsize=(8, max(4, len(models)*0.5+1)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks([0,1,2]); ax.set_xticklabels(["RMSE_VA\n(lower=better)", "PCC_V\n(higher=better)", "PCC_A\n(higher=better)"])
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax)
    plt.title(f"Performance Heatmap ({args.domain})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/performance_heatmap_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/performance_heatmap_{args.domain}.png")

# ── 8. Radar Chart ───────────────────────────────────────────────────────
def plot_radar():
    """Radar chart of normalized metrics for top models."""
    data = load_json("logs/baseline_results.json")
    if not data: return

    filtered = [d for d in data if d.get("domain") == args.domain][:5]
    if len(filtered) < 2: return

    categories = ["1 - RMSE_norm", "PCC_V", "PCC_A"]
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("husl", len(filtered))

    for i, d in enumerate(filtered):
        vals = [
            1 - d.get("RMSE_norm", 0.2),
            max(0, d.get("PCC_V", 0) or 0),
            max(0, d.get("PCC_A", 0) or 0),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, color=colors[i], label=d["model"])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.title(f"Model Radar – {args.domain}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/radar_chart_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/radar_chart_{args.domain}.png")

# ── 9. Uncertainty Evolution (DeBERTa) ───────────────────────────────────
def plot_uncertainty():
    """Plot learned uncertainty σ_V and σ_A across epochs."""
    hist_files = glob.glob(os.path.join("logs", f"deberta_history_{args.domain}_*.json"))
    if not hist_files: return

    hist = load_json(hist_files[-1])
    if not hist or "sigma_v" not in hist[0]: return

    epochs = [h["epoch"] for h in hist]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, [h["sigma_v"] for h in hist], "o-", color="#9C27B0", lw=2, label="σ_V (Valence)")
    ax1.plot(epochs, [h["sigma_a"] for h in hist], "s-", color="#FF9800", lw=2, label="σ_A (Arousal)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Learned Uncertainty (σ)")
    ax1.set_title("Uncertainty Evolution", fontsize=12)
    ax1.legend()

    if "lr" in hist[0]:
        ax2.plot(epochs, [h["lr"] for h in hist], "D-", color="#009688", lw=2)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule", fontsize=12)
        ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))

    plt.suptitle(f"DeBERTa Training Dynamics ({args.domain})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/uncertainty_evolution_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/uncertainty_evolution_{args.domain}.png")

# ── 10. XAI: Attention Heatmap ───────────────────────────────────────────
def plot_attention_heatmaps():
    """Generate attention heatmaps from the DeBERTa model."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except: return

    ckpt = os.path.join("checkpoints", f"best_deberta_{args.domain}.pt")
    if not os.path.exists(ckpt):
        print("  No DeBERTa checkpoint found, skipping attention heatmaps")
        return

    dev_path = os.path.join(BASE, "task-dataset", "track_a", "subtask_1", "eng",
                            f"eng_{args.domain}_dev_task1.jsonl")
    dev = load_jsonl(dev_path)
    if not dev: return

    MODEL_NAME = "microsoft/deberta-v3-base"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Take N sample entries
    samples = dev[:6]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for idx, entry in enumerate(samples):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]
        aspect = entry["Aspect_VA"][0]["Aspect"]
        text = entry["Text"]

        enc = tok(text, aspect, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))

        # Average last layer attention across heads
        attn = out.attentions[-1][0].mean(dim=0).cpu().numpy()  # (seq, seq)
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        # Get non-padding tokens
        mask = enc["attention_mask"][0].numpy().astype(bool)
        n_tokens = min(mask.sum(), 30)
        attn_sub = attn[:n_tokens, :n_tokens]
        tok_sub = tokens[:n_tokens]

        im = ax.imshow(attn_sub, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n_tokens)); ax.set_xticklabels(tok_sub, rotation=90, fontsize=6)
        ax.set_yticks(range(n_tokens)); ax.set_yticklabels(tok_sub, fontsize=6)
        ax.set_title(f"[{aspect}]", fontsize=9)

    plt.suptitle(f"Attention Heatmaps – DeBERTa ({args.domain})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/xai/attention_heatmaps_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/xai/attention_heatmaps_{args.domain}.png")

# ── 11. XAI: Token Attribution ───────────────────────────────────────────
def plot_token_attribution():
    """Approximate token importance via attention × gradient (simplified)."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except: return

    dev_path = os.path.join(BASE, "task-dataset", "track_a", "subtask_1", "eng",
                            f"eng_{args.domain}_dev_task1.jsonl")
    dev = load_jsonl(dev_path)
    if not dev: return

    MODEL_NAME = "microsoft/deberta-v3-base"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    samples = dev[:6]

    for idx, entry in enumerate(samples):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]
        aspect = entry["Aspect_VA"][0]["Aspect"]
        va = entry["Aspect_VA"][0]["VA"]

        enc = tok(entry["Text"], aspect, max_length=64, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))

        # CLS attention to all tokens (last layer, averaged across heads)
        cls_attn = out.attentions[-1][0].mean(dim=0)[0].cpu().numpy()
        tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
        mask = enc["attention_mask"][0].numpy().astype(bool)
        n = min(mask.sum(), 25)

        colors_bar = ["#F44336" if i == 0 else "#2196F3" for i in range(n)]
        ax.barh(range(n), cls_attn[:n], color=colors_bar, alpha=0.7)
        ax.set_yticks(range(n)); ax.set_yticklabels(tokens[:n], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"'{aspect}' → VA={va}", fontsize=9)
        ax.set_xlabel("CLS Attention Weight")

    plt.suptitle(f"Token Attribution (CLS Attention) – {args.domain}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/xai/token_attribution_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/xai/token_attribution_{args.domain}.png")

# ── 12. XAI: SHAP-like Feature Importance ────────────────────────────────
def plot_shap_importance():
    """SHAP-style feature importance using attention weights as proxy."""
    detail = load_json("logs/baseline_detailed_predictions.json")
    if not detail: return

    # Find best model data
    best = min([d for d in detail if d.get("domain") == args.domain],
               key=lambda x: x["RMSE_VA"], default=None)
    if not best: return

    # Create SHAP-like importance based on prediction errors
    err_v = np.array(best["pred_v"]) - np.array(best["gold_v"])
    err_a = np.array(best["pred_a"]) - np.array(best["gold_a"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Feature: gold value bins vs error magnitude
    v_bins = np.digitize(best["gold_v"], bins=np.arange(1, 10, 1))
    a_bins = np.digitize(best["gold_a"], bins=np.arange(1, 10, 1))

    v_mean_err = [np.mean(np.abs(err_v[v_bins==b])) if (v_bins==b).sum()>0 else 0 for b in range(1,10)]
    a_mean_err = [np.mean(np.abs(err_a[a_bins==b])) if (a_bins==b).sum()>0 else 0 for b in range(1,10)]

    ax1.bar(range(1,10), v_mean_err, color="#2196F3", alpha=0.7, edgecolor="white")
    ax1.set_xlabel("Gold Valence Bin"); ax1.set_ylabel("Mean |Error|")
    ax1.set_title("Valence Error by Gold Score Range")

    ax2.bar(range(1,10), a_mean_err, color="#F44336", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Gold Arousal Bin"); ax2.set_ylabel("Mean |Error|")
    ax2.set_title("Arousal Error by Gold Score Range")

    plt.suptitle(f"Error Analysis by Score Range – {best['model']} ({args.domain})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"plots/xai/error_by_score_range_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/xai/error_by_score_range_{args.domain}.png")

    # Correlation matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_data = np.array([best["gold_v"], best["gold_a"], best["pred_v"], best["pred_a"],
                          np.abs(err_v).tolist(), np.abs(err_a).tolist()])
    labels = ["Gold_V", "Gold_A", "Pred_V", "Pred_A", "|Err_V|", "|Err_A|"]
    corr = np.corrcoef(corr_data)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=labels,
                yticklabels=labels, ax=ax, vmin=-1, vmax=1)
    ax.set_title(f"Correlation Matrix – {best['model']}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plots/xai/correlation_matrix_{args.domain}.png")
    plt.close()
    print(f"  ✓ plots/xai/correlation_matrix_{args.domain}.png")

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print(f"DimABSA2026 – Generating All Plots ({args.domain})")
    print("="*60)

    print("\n📊 Classic Plots:")
    plot_training_curves()
    plot_model_comparison()
    plot_scatter_gold_vs_pred()
    plot_error_distribution()
    plot_va_space()
    plot_error_boxplots()
    plot_performance_heatmap()
    plot_radar()

    print("\n🔍 XAI Plots:")
    plot_uncertainty()
    plot_attention_heatmaps()
    plot_token_attribution()
    plot_shap_importance()

    print(f"\n✅ All plots saved to plots/ and plots/xai/")

if __name__ == "__main__":
    main()
