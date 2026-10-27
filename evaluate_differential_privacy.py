import os
import sys
import argparse
import time
import math
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from data.preprocessing import NBaIoTDataPreprocessor
from data.download_nbaiot import get_nbaiot_feature_names, prepare_nbaiot_data
from data.nbaiot_dataset import create_dataloaders, NBaIoTDataset
from data.non_iid_partition import build_non_iid_benchmark_data
from models.lstm import LSTMIDS
from security.runner import run_experiment
from security.attacks.base_attack import select_malicious_clients
from security.differential_privacy import (
    compute_rdp_cumulative_privacy,
    calibrate_analytic_gaussian_sigma,
    DPAccountant
)
from xai.shap_explainer import SHAPExplainer
from xai.explanation_metrics import (
    compute_explanation_cosine_similarity,
    compute_top_k_jaccard,
    compute_explanation_fidelity
)

logger = get_logger("StageI_DifferentialPrivacy")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage I: Differential Privacy Evaluation Suite")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default="results/differential_privacy_stage_i", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    parser.add_argument("--epsilons", type=str, default="20.0,10.0,5.0,2.0,1.0", help="Comma-separated target cumulative epsilons")
    parser.add_argument("--include-ref", action="store_true", help="Include weak reference epsilon=100.0")
    parser.add_argument("--seeds", type=str, default="1", help="Comma-separated random seeds (e.g. '1' or '1,2,3')")
    parser.add_argument("--rounds", type=int, default=3, help="Number of communication rounds")
    parser.add_argument("--alphas", type=str, default="1.0", help="Dirichlet alpha for Non-IID + DP scenario")
    parser.add_argument("--clipping-norm", type=float, default=1.0, help="L2 sensitivity clipping threshold C")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target privacy delta (default 1e-5)")
    parser.add_argument("--sanity", action="store_true", help="Run quick sanity check (1 seed, 1 round, epsilons=[inf, 10.0])")
    return parser.parse_args()


# =========================================================================
# PLOTTING ROUTINES (10 PUBLICATION-QUALITY FIGURES)
# =========================================================================

def plot_fig1_f1_vs_epsilon(summary_df: pd.DataFrame, save_path: str):
    """Fig 1: Global IDS F1-score vs epsilon across 4 aggregators for clean scenario."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = summary_df[(summary_df["scenario"].str.contains("Clean")) & (summary_df["epsilon"] != "inf")].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    palette = {"fedavg": "#e74c3c", "model_trust": "#f39c12", "joint_trust": "#3498db", "temporal_trust": "#2ecc71"}
    markers = {"fedavg": "x", "model_trust": "s", "joint_trust": "^", "temporal_trust": "o"}
    
    plt.figure(figsize=(9, 5))
    for meth in ["fedavg", "model_trust", "joint_trust", "temporal_trust"]:
        m_df = df[df["aggregation_method"] == meth].sort_values("eps_float")
        if not m_df.empty:
            plt.plot(m_df["eps_float"], m_df["f1_mean"] * 100.0, marker=markers[meth], color=palette[meth],
                     label=meth.replace("_", " ").title(), linewidth=2.2, markersize=8)
            
    # Reference line for DP OFF
    ref_df = summary_df[(summary_df["scenario"].str.contains("Clean")) & (summary_df["epsilon"] == "inf")]
    if not ref_df.empty:
        ref_f1 = ref_df["f1_mean"].mean() * 100.0
        plt.axhline(ref_f1, color="gray", linestyle="--", alpha=0.7, label=f"DP OFF Reference ({ref_f1:.1f}%)")
        
    plt.title("IDS F1-Score vs. Privacy Budget (Clean Network)", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy / More Noise)", fontsize=10)
    plt.ylabel("Global Test F1-Score (%)", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(df["eps_float"].unique()), [f"{x:.1f}" for x in sorted(df["eps_float"].unique())])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(80, 102)
    plt.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 1 (F1 vs Epsilon) to: {save_path}")


def plot_fig2_accuracy_vs_epsilon(summary_df: pd.DataFrame, save_path: str):
    """Fig 2: Global IDS Accuracy vs epsilon across 4 aggregators for clean scenario."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = summary_df[(summary_df["scenario"].str.contains("Clean")) & (summary_df["epsilon"] != "inf")].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    palette = {"fedavg": "#e74c3c", "model_trust": "#f39c12", "joint_trust": "#3498db", "temporal_trust": "#2ecc71"}
    markers = {"fedavg": "x", "model_trust": "s", "joint_trust": "^", "temporal_trust": "o"}
    
    plt.figure(figsize=(9, 5))
    for meth in ["fedavg", "model_trust", "joint_trust", "temporal_trust"]:
        m_df = df[df["aggregation_method"] == meth].sort_values("eps_float")
        if not m_df.empty:
            plt.plot(m_df["eps_float"], m_df["accuracy_mean"] * 100.0, marker=markers[meth], color=palette[meth],
                     label=meth.replace("_", " ").title(), linewidth=2.2, markersize=8)
            
    plt.title("IDS Detection Accuracy vs. Privacy Budget (Clean Network)", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy / More Noise)", fontsize=10)
    plt.ylabel("Global Test Accuracy (%)", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(df["eps_float"].unique()), [f"{x:.1f}" for x in sorted(df["eps_float"].unique())])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(80, 102)
    plt.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 2 (Accuracy vs Epsilon) to: {save_path}")


def plot_fig3_shap_similarity_vs_epsilon(expl_df: pd.DataFrame, save_path: str):
    """Fig 3: SHAP Cosine Similarity vs epsilon comparing DP models against DP OFF ground truth."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = expl_df[expl_df["epsilon"] != "inf"].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x="eps_float", y="shap_cosine_similarity", hue="aggregation_method", marker="o", linewidth=2.4)
    plt.title("Explanation Alignment vs. Privacy: SHAP Cosine Similarity to DP OFF Baseline", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy)", fontsize=10)
    plt.ylabel("Cosine Similarity $S_{\cos}(E_{dp}, E_{clean}) \in [-1, 1]$", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(df["eps_float"].unique()), [f"{x:.1f}" for x in sorted(df["eps_float"].unique())])
    plt.ylim(0.40, 1.05)
    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Exact Match (1.0)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 3 (SHAP Similarity vs Epsilon) to: {save_path}")


def plot_fig4_topk_stability_vs_epsilon(expl_df: pd.DataFrame, save_path: str):
    """Fig 4: Top-K feature stability (Jaccard Index) vs epsilon."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = expl_df[expl_df["epsilon"] != "inf"].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x="eps_float", y="topk_jaccard", hue="aggregation_method", marker="s", linewidth=2.4)
    plt.title("Top-10 Decisive Feature Stability (Jaccard Index) vs. Privacy Budget", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy)", fontsize=10)
    plt.ylabel("Top-10 Feature Jaccard Index $J(Top_{dp}, Top_{clean}) \in [0, 1]$", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(df["eps_float"].unique()), [f"{x:.1f}" for x in sorted(df["eps_float"].unique())])
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 4 (Top-K Stability vs Epsilon) to: {save_path}")


def plot_fig5_explanation_fidelity_vs_epsilon(expl_df: pd.DataFrame, save_path: str):
    """Fig 5: Explanation Prediction Shift Fidelity vs epsilon."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = expl_df[expl_df["epsilon"] != "inf"].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x="eps_float", y="explanation_fidelity", hue="aggregation_method", marker="^", linewidth=2.4)
    plt.title("Explanation Fidelity (Decisive Feature Perturbation Shift) vs. Privacy Budget", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy)", fontsize=10)
    plt.ylabel("Prediction Probability Shift $\Delta P_{mask}$", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(df["eps_float"].unique()), [f"{x:.1f}" for x in sorted(df["eps_float"].unique())])
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 5 (Fidelity vs Epsilon) to: {save_path}")


def plot_fig6_honest_trust_vs_epsilon(trust_df: pd.DataFrame, save_path: str):
    """Fig 6: Honest client trust component breakdown (M_i, E_i, T_i, F_i) vs epsilon."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = trust_df[(~trust_df["is_malicious"]) & (trust_df["epsilon"] != "inf") & (trust_df["aggregation_method"] == "temporal_trust")].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    # Filter for final round
    max_round_idx = df.groupby(["scenario", "epsilon", "aggregation_method", "seed", "client_id"])["round"].idxmax()
    df = df.loc[max_round_idx]
    
    grouped = df.groupby("eps_float")[["model_trust", "expl_trust", "joint_trust", "final_trust"]].mean().reset_index()
    
    plt.figure(figsize=(9, 5))
    plt.plot(grouped["eps_float"], grouped["model_trust"], marker="o", label="Model Trust ($M_i$)", color="#3498db", linewidth=2.2)
    plt.plot(grouped["eps_float"], grouped["expl_trust"], marker="s", label="Explanation Trust ($E_i$)", color="#2ecc71", linewidth=2.2)
    plt.plot(grouped["eps_float"], grouped["joint_trust"], marker="^", label="Joint Trust ($T_i$)", color="#f39c12", linewidth=2.2)
    plt.plot(grouped["eps_float"], grouped["final_trust"], marker="d", label="Final Temporal Trust ($F_i$)", color="#9b59b6", linewidth=2.5)
    
    plt.axhline(0.50, color="red", linestyle="--", linewidth=1.5, label="Trust Defense Threshold (0.50)")
    plt.title("Honest Client Trust Scores Under Differential Privacy Noise", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy / High Noise)", fontsize=10)
    plt.ylabel("Trust Score $[0, 1]$", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(grouped["eps_float"].unique()), [f"{x:.1f}" for x in sorted(grouped["eps_float"].unique())])
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 6 (Honest Trust vs Epsilon) to: {save_path}")


def plot_fig7_malicious_trust_vs_epsilon(trust_df: pd.DataFrame, save_path: str):
    """Fig 7: Trust gap between honest and malicious clients vs epsilon under adversarial conditions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = trust_df[(trust_df["scenario"].str.contains("Poisoning|Flipping|Sign")) & 
                  (trust_df["epsilon"] != "inf") & 
                  (trust_df["aggregation_method"] == "temporal_trust")].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    max_round_idx = df.groupby(["scenario", "epsilon", "aggregation_method", "seed", "client_id"])["round"].idxmax()
    df = df.loc[max_round_idx]
    
    honest_df = df[~df["is_malicious"]].groupby(["scenario", "eps_float"])["final_trust"].mean().reset_index()
    mal_df = df[df["is_malicious"]].groupby(["scenario", "eps_float"])["final_trust"].mean().reset_index()
    
    merged = pd.merge(honest_df, mal_df, on=["scenario", "eps_float"], suffixes=("_honest", "_malicious"))
    merged["trust_gap"] = merged["final_trust_honest"] - merged["final_trust_malicious"]
    
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=merged, x="eps_float", y="trust_gap", hue="scenario", marker="o", linewidth=2.4)
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    plt.title("Trust Separation Gap ($\Delta F = F_{honest} - F_{mal}$) vs. Privacy Budget", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy)", fontsize=10)
    plt.ylabel("Trust Gap $\Delta F$", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(merged["eps_float"].unique()), [f"{x:.1f}" for x in sorted(merged["eps_float"].unique())])
    plt.ylim(-0.1, 0.6)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 7 (Malicious Trust Gap vs Epsilon) to: {save_path}")


def plot_fig8_detection_rate_vs_epsilon(det_df: pd.DataFrame, save_path: str):
    """Fig 8: Malicious Client Detection ROC-AUC vs epsilon."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = det_df[(det_df["scenario"].str.contains("Poisoning|Flipping|Sign")) & 
                (det_df["epsilon"] != "inf") & 
                (det_df["aggregation_method"] == "temporal_trust")].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x="eps_float", y="detection_auc", hue="scenario", marker="s", linewidth=2.4)
    plt.title("Malicious Client Detection ROC-AUC vs. Privacy Budget", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy)", fontsize=10)
    plt.ylabel("Detection ROC-AUC", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(df["eps_float"].unique()), [f"{x:.1f}" for x in sorted(df["eps_float"].unique())])
    plt.ylim(0.40, 1.05)
    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Perfect Detection (1.0)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 8 (Detection Rate vs Epsilon) to: {save_path}")


def plot_fig9_honest_fpr_vs_epsilon(det_df: pd.DataFrame, save_path: str):
    """Fig 9: Honest Client False Rejection Rate (False Alarm FPR) vs epsilon."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = det_df[(det_df["epsilon"] != "inf") & (det_df["aggregation_method"] == "temporal_trust")].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x="eps_float", y="detection_fpr", hue="scenario", marker="^", linewidth=2.4)
    plt.title("Honest Client False Alarm Rate (False Rejection by $\\tau = 0.50$) vs. $\\epsilon$", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Decreasing $\epsilon$ $\to$ Stronger Privacy)", fontsize=10)
    plt.ylabel("Honest Client False Positive Rate", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(df["eps_float"].unique()), [f"{x:.1f}" for x in sorted(df["eps_float"].unique())])
    plt.ylim(-0.05, 0.40)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 9 (Honest FPR vs Epsilon) to: {save_path}")


def plot_fig10_privacy_utility_tradeoff(summary_df: pd.DataFrame, save_path: str):
    """Fig 10: Privacy-Utility Pareto Frontier (Global F1 vs Cumulative epsilon)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = summary_df[(summary_df["scenario"].str.contains("Clean")) & (summary_df["epsilon"] != "inf")].copy()
    if df.empty:
        return
    df["eps_float"] = df["epsilon"].astype(float)
    
    palette = {"fedavg": "#e74c3c", "temporal_trust": "#2ecc71"}
    
    plt.figure(figsize=(9, 5))
    for meth in ["fedavg", "temporal_trust"]:
        m_df = df[df["aggregation_method"] == meth].sort_values("eps_float")
        if not m_df.empty:
            plt.plot(m_df["eps_float"], m_df["f1_mean"] * 100.0, marker="o", color=palette[meth],
                     label=f"{meth.replace('_', ' ').title()} Pareto Curve", linewidth=2.5, markersize=8)
            
    plt.title("Privacy-Utility Pareto Frontier: IDS F1-Score Across Formal Privacy Budgets", fontsize=13, fontweight="bold")
    plt.xlabel(r"Cumulative Privacy Budget $\epsilon$ (Higher $\to$ Lower Noise / Weaker Privacy)", fontsize=10)
    plt.ylabel("IDS Utility (F1-Score %)", fontsize=10)
    plt.xscale("log")
    plt.xticks(sorted(df["eps_float"].unique()), [f"{x:.1f}" for x in sorted(df["eps_float"].unique())])
    plt.ylim(80, 102)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Fig 10 (Privacy-Utility Tradeoff) to: {save_path}")


# =========================================================================
# MAIN EVALUATION SUITE
# =========================================================================

def run_stage_i_differential_privacy_suite(
    output_dir: str = "results/differential_privacy_stage_i",
    epsilons: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    num_rounds: int = 3,
    alpha_non_iid: float = 1.0,
    clipping_norm: float = 1.0,
    delta: float = 1e-5,
    device: str = "cuda",
    sanity_check: bool = False,
    include_ref: bool = False
):
    """
    Executes the comprehensive Stage I Differential Privacy Evaluation suite.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if sanity_check:
        eps_list = [float("inf"), 10.0]
        seed_list = [1]
        num_rounds = 1
    else:
        if epsilons is None:
            eps_list = [20.0, 10.0, 5.0, 2.0, 1.0]
        else:
            eps_list = epsilons
        if include_ref and (100.0 not in eps_list):
            eps_list = [100.0] + eps_list
        # Prepend float('inf') for DP OFF reference
        if float("inf") not in eps_list:
            eps_list = [float("inf")] + eps_list
        seed_list = seeds if seeds is not None else [1]

    dev_type = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    logger.info("=" * 95)
    logger.info("   STAGE I: DIFFERENTIAL PRIVACY (DP) EVALUATION SUITE")
    logger.info(f"   Device: [{dev_type.upper()}] | Epsilons: {eps_list} | Delta: {delta} | Rounds: {num_rounds}")
    logger.info(f"   Seeds: {seed_list} | Clipping Norm C: {clipping_norm} | Sanity Check: {sanity_check}")
    logger.info("=" * 95)

    # 1. Prepare Base Clean Data & Non-IID Partition
    data_dir = "data/nbaiot"
    prepare_nbaiot_data(data_dir=data_dir, sample_size=3000, seed=42)
    device_names = [
        "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera"
    ]
    dev_to_id = {dev: i for i, dev in enumerate(device_names)}

    preprocessor = NBaIoTDataPreprocessor(
        data_dir=data_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    client_data_raw, (global_test_x, global_test_y) = preprocessor.load_and_preprocess_all(device_names)
    clean_cached_data = (client_data_raw, (global_test_x, global_test_y))

    # Reference background for SHAP explanations
    benign_idx = np.where(global_test_y == 0)[0][:10]
    bg_samples = global_test_x[benign_idx] if len(benign_idx) >= 10 else global_test_x[:10]
    eval_x_sample = global_test_x[:20]
    feature_names = get_nbaiot_feature_names()

    # Generate Non-IID Partition for Non-IID + DP Evaluation
    logger.info(f"\n>>> Generating Dirichlet Non-IID partition (alpha={alpha_non_iid})...")
    non_iid_clients, _ = build_non_iid_benchmark_data(
        processed_client_data=client_data_raw,
        global_test=(global_test_x, global_test_y),
        device_names=device_names,
        alpha=alpha_non_iid,
        seed=42,
        batch_size=64
    )
    non_iid_cached_data = (non_iid_clients, (global_test_x, global_test_y))

    # 2. Calibrate Exact Gaussian Noise Multiplier for Cumulative Privacy Budgets
    logger.info("\n>>> Performing RDP Privacy Accounting & Noise Multiplier Calibration...")
    privacy_accounting_records = []
    sigma_map: Dict[float, float] = {}

    for eps in eps_list:
        if math.isinf(eps):
            sigma_map[eps] = 0.0
            rec = {
                "epsilon": "inf",
                "delta": delta,
                "sigma": 0.0,
                "clipping_norm": clipping_norm,
                "num_rounds": num_rounds,
                "cumulative_epsilon_rdp": float("inf"),
                "optimal_alpha_rdp": 1.0,
                "advanced_comp_epsilon": float("inf"),
                "basic_comp_epsilon": float("inf"),
                "privacy_accounting_status": "Non-Private Reference (DP OFF)"
            }
        else:
            # Calibrate sigma guaranteeing target cumulative epsilon over num_rounds
            sig = calibrate_analytic_gaussian_sigma(
                target_epsilon=eps,
                target_delta=delta,
                num_rounds=num_rounds
            )
            sigma_map[eps] = sig
            
            # Verify and record composition bounds
            act_rdp_eps, opt_a, _ = compute_rdp_cumulative_privacy(
                num_rounds=num_rounds,
                sigma=sig,
                delta=delta
            )
            
            # Record via DPAccountant
            acc = DPAccountant(target_delta=delta)
            eps_round = (math.sqrt(2.0 * math.log(1.25 / delta)) / sig) if sig > 0 else float("inf")
            for _ in range(num_rounds):
                acc.step(epsilon_round=eps_round, sigma=sig)
            cum_eps_acc, _ = acc.get_cumulative_privacy()
            
            rec = {
                "epsilon": eps,
                "delta": delta,
                "sigma": sig,
                "clipping_norm": clipping_norm,
                "num_rounds": num_rounds,
                "cumulative_epsilon_rdp": act_rdp_eps,
                "optimal_alpha_rdp": opt_a,
                "advanced_comp_epsilon": cum_eps_acc,
                "basic_comp_epsilon": num_rounds * eps_round,
                "privacy_accounting_status": "Formally Guaranteed via Analytical RDP Composition"
            }
            logger.info(f"   Target Cumulative Eps: {eps:4.1f} -> Calibrated Sigma: {sig:6.4f} (RDP Verification: {act_rdp_eps:.2f})")
        privacy_accounting_records.append(rec)

    accounting_df = pd.DataFrame(privacy_accounting_records)
    accounting_csv = os.path.join(output_dir, "dp_privacy_accounting.csv")
    accounting_df.to_csv(accounting_csv, index=False)
    logger.info(f"Saved privacy accounting summary to: {accounting_csv}")

    # 3. Define Evaluation Scenarios
    scenarios = [
        {
            "name": "Scenario 1: Clean Data + DP",
            "is_non_iid": False,
            "attack_type": "none",
            "malicious_ratio": 0.0,
            "attack_kwargs": {}
        },
        {
            "name": "Scenario 2: Non-IID + DP",
            "is_non_iid": True,
            "attack_type": "none",
            "malicious_ratio": 0.0,
            "attack_kwargs": {}
        },
        {
            "name": "Scenario 3: Model Poisoning + DP",
            "is_non_iid": False,
            "attack_type": "model_poisoning",
            "malicious_ratio": 0.10,
            "attack_kwargs": {"intensity": 4.0, "start_round": 1}
        },
        {
            "name": "Scenario 4: Sign Flipping + DP",
            "is_non_iid": False,
            "attack_type": "sign_flipping",
            "malicious_ratio": 0.10,
            "attack_kwargs": {"intensity": 1.0, "start_round": 1}
        }
    ]

    aggregation_methods = ["fedavg", "model_trust", "joint_trust", "temporal_trust"]

    global_perf_records = []
    trust_score_records = []
    agg_weight_records = []
    detection_records = []
    explanation_records = []

    # Store Clean Non-Private (DP OFF, Temporal Trust) Reference Explanation
    clean_ground_truth_expl = None
    clean_ground_truth_top10 = None

    total_runs = len(scenarios) * len(eps_list) * len(seed_list) * len(aggregation_methods)
    current_run = 0
    t_start = time.time()
    logger.info(f"\n>>> Starting Stage I Evaluation Suite ({total_runs} total runs planned)...")

    # 4. Main Evaluation Loop
    for scen in scenarios:
        scen_name = scen["name"]
        cached_data = non_iid_cached_data if scen["is_non_iid"] else clean_cached_data
        client_dataset = cached_data[0]

        for eps in eps_list:
            eps_label = "inf" if math.isinf(eps) else str(eps)
            sig_val = sigma_map[eps]
            dp_enabled = not math.isinf(eps)

            dp_cfg = {
                "enabled": dp_enabled,
                "epsilon": eps if dp_enabled else 1000.0,
                "delta": delta,
                "clipping_norm": clipping_norm,
                "sigma": sig_val,
                "apply_to_malicious": False,  # Honest clients protect data; adversaries bypass DP
                "update_client_model": True
            }

            for seed in seed_list:
                if scen["attack_type"] == "none" or scen["malicious_ratio"] <= 0.0:
                    malicious_devices = []
                else:
                    malicious_devices = select_malicious_clients(
                        total_clients=len(device_names),
                        malicious_ratio=scen["malicious_ratio"],
                        seed=seed,
                        client_ids=device_names
                    )

                for method in aggregation_methods:
                    current_run += 1
                    logger.info(
                        f"[{current_run}/{total_runs}] {scen_name} | eps={eps_label} | seed={seed} | method={method}..."
                    )

                    t_run0 = time.time()
                    res = run_experiment(
                        attack_type=scen["attack_type"],
                        aggregation_method=method,
                        malicious_ratio=scen["malicious_ratio"],
                        seed=seed,
                        num_rounds=num_rounds,
                        attack_kwargs=scen["attack_kwargs"],
                        device=dev_type,
                        preprocessed_data=cached_data,
                        dp_config=dp_cfg
                    )
                    run_dur = time.time() - t_run0

                    rounds_data = res["rounds"]
                    fin_rnd = rounds_data[-1]

                    # A. Global Performance Record
                    for rnd in rounds_data:
                        r_idx = rnd["round"]
                        perf_rec = {
                            "scenario": scen_name,
                            "epsilon": eps_label,
                            "delta": delta,
                            "sigma": sig_val,
                            "aggregation_method": method,
                            "seed": seed,
                            "round": r_idx,
                            "accuracy": rnd["accuracy"],
                            "precision": rnd["precision"],
                            "recall": rnd["recall"],
                            "f1": rnd["f1"],
                            "fpr": rnd["fpr"],
                            "fnr": rnd["fnr"],
                            "roc_auc": rnd["roc_auc"],
                            "asr": rnd["asr"],
                            "malicious_weight_sum": rnd["malicious_weight_sum"],
                            "honest_weight_sum": rnd["honest_weight_sum"],
                            "avg_honest_model_trust": rnd["avg_honest_model_trust"],
                            "avg_malicious_model_trust": rnd["avg_malicious_model_trust"],
                            "avg_honest_expl_trust": rnd["avg_honest_expl_trust"],
                            "avg_malicious_expl_trust": rnd["avg_malicious_expl_trust"],
                            "avg_honest_joint_trust": rnd["avg_honest_joint_trust"],
                            "avg_malicious_joint_trust": rnd["avg_malicious_joint_trust"],
                            "avg_honest_final_trust": rnd["avg_honest_final_trust"],
                            "avg_malicious_final_trust": rnd["avg_malicious_final_trust"],
                            "trust_gap": rnd["trust_gap"],
                            "detection_precision": rnd["detection_precision"],
                            "detection_recall": rnd["detection_recall"],
                            "detection_f1": rnd["detection_f1"],
                            "detection_auc": rnd["detection_auc"],
                            "detection_fpr": rnd["detection_fpr"],
                            "round_total_time": rnd["round_total_time"],
                            "privacy_accounting_status": "DP OFF" if math.isinf(eps) else "RDP-Guaranteed"
                        }
                        global_perf_records.append(perf_rec)

                        # B. Malicious Detection Record
                        det_rec = {
                            "scenario": scen_name,
                            "epsilon": eps_label,
                            "aggregation_method": method,
                            "seed": seed,
                            "round": r_idx,
                            "detection_precision": rnd["detection_precision"],
                            "detection_recall": rnd["detection_recall"],
                            "detection_f1": rnd["detection_f1"],
                            "detection_fpr": rnd["detection_fpr"],
                            "detection_auc": rnd["detection_auc"],
                            "threshold": 0.50
                        }
                        detection_records.append(det_rec)

                        # C. Client-Level Trust Scores & Aggregation Weights
                        for dev in device_names:
                            is_mal = dev in malicious_devices
                            role = "Malicious" if is_mal else "Honest"
                            cid = dev_to_id[dev]

                            if "num_train" in client_dataset[dev]:
                                sample_cnt = int(client_dataset[dev]["num_train"])
                            else:
                                sample_cnt = len(client_dataset[dev]["y_train"])

                            weight_val = float(rnd["client_weights"].get(dev, 0.0))

                            trust_score_records.append({
                                "scenario": scen_name,
                                "epsilon": eps_label,
                                "sigma": sig_val,
                                "aggregation_method": method,
                                "seed": seed,
                                "round": r_idx,
                                "client_id": cid,
                                "device_name": dev,
                                "is_malicious": is_mal,
                                "role": role,
                                "sample_count": sample_cnt,
                                "model_trust": float(rnd["client_model_trusts"].get(dev, 0.0)),
                                "expl_trust": float(rnd["client_expl_trusts"].get(dev, 0.0)),
                                "joint_trust": float(rnd["client_joint_trusts"].get(dev, 0.0)),
                                "final_trust": float(rnd["client_final_trusts"].get(dev, 0.0))
                            })

                            agg_weight_records.append({
                                "scenario": scen_name,
                                "epsilon": eps_label,
                                "sigma": sig_val,
                                "aggregation_method": method,
                                "seed": seed,
                                "round": r_idx,
                                "client_id": cid,
                                "device_name": dev,
                                "is_malicious": is_mal,
                                "sample_count": sample_cnt,
                                "aggregation_weight": weight_val,
                                "weight_pct": weight_val * 100.0
                            })

                    # D. Explanation Quality Evaluation on Global Model
                    # Reconstruct global model from weights
                    global_model = LSTMIDS(input_dim=115, hidden_dim=64, num_layers=2, dense_dim=32, dropout=0.2).to("cpu")
                    global_model.load_state_dict(res["global_model_weights"])
                    global_model.eval()

                    c_expl = SHAPExplainer(model=global_model, background_data=bg_samples, feature_names=feature_names, device="cpu")
                    s_vals = c_expl.compute_shap_values(eval_x_sample)
                    global_shap_rep = np.mean(np.abs(s_vals), axis=0)
                    global_shap_rep = np.nan_to_num(global_shap_rep, nan=0.0)

                    top_10_indices = list(np.argsort(global_shap_rep)[::-1][:10])
                    top_10_features = [feature_names[i] for i in top_10_indices]

                    # Establish ground truth reference explanation from Clean DP OFF Temporal Trust
                    if scen_name == "Scenario 1: Clean Data + DP" and math.isinf(eps) and method == "temporal_trust":
                        clean_ground_truth_expl = global_shap_rep.copy()
                        clean_ground_truth_top10 = top_10_features.copy()

                    if clean_ground_truth_expl is not None:
                        cos_sim = compute_explanation_cosine_similarity(global_shap_rep, clean_ground_truth_expl)
                        jacc = compute_top_k_jaccard(top_10_features, clean_ground_truth_top10)
                    else:
                        cos_sim = 1.0
                        jacc = 1.0

                    fidelity = compute_explanation_fidelity(
                        model=global_model,
                        x_test=eval_x_sample,
                        top_k_indices=top_10_indices,
                        device="cpu"
                    )

                    explanation_records.append({
                        "scenario": scen_name,
                        "epsilon": eps_label,
                        "sigma": sig_val,
                        "aggregation_method": method,
                        "seed": seed,
                        "shap_cosine_similarity": cos_sim,
                        "topk_jaccard": jacc,
                        "explanation_fidelity": fidelity
                    })

                    logger.info(
                        f"   Done in {run_dur:.1f}s | Acc: {fin_rnd['accuracy']*100:.2f}% | F1: {fin_rnd['f1']*100:.2f}% | CosSim: {cos_sim:.4f} | Jaccard: {jacc:.4f}"
                    )

    elapsed_suite = time.time() - t_start
    logger.info(f"\nAll {total_runs} runs completed in {elapsed_suite:.1f}s ({elapsed_suite/60:.2f}m).")

    # 5. Save DataFrames to CSVs
    perf_df = pd.DataFrame(global_perf_records)
    perf_csv = os.path.join(output_dir, "dp_global_performance.csv")
    perf_df.to_csv(perf_csv, index=False)
    logger.info(f"Saved global performance to: {perf_csv}")

    trust_df = pd.DataFrame(trust_score_records)
    trust_csv = os.path.join(output_dir, "dp_trust_scores.csv")
    trust_df.to_csv(trust_csv, index=False)
    logger.info(f"Saved trust scores to: {trust_csv}")

    weights_df = pd.DataFrame(agg_weight_records)
    weights_csv = os.path.join(output_dir, "dp_aggregation_weights.csv")
    weights_df.to_csv(weights_csv, index=False)
    logger.info(f"Saved aggregation weights to: {weights_csv}")

    detection_df = pd.DataFrame(detection_records)
    det_csv = os.path.join(output_dir, "dp_detection_metrics.csv")
    detection_df.to_csv(det_csv, index=False)
    logger.info(f"Saved detection metrics to: {det_csv}")

    expl_df = pd.DataFrame(explanation_records)
    expl_csv = os.path.join(output_dir, "dp_explanation_quality.csv")
    expl_df.to_csv(expl_csv, index=False)
    logger.info(f"Saved explanation quality to: {expl_csv}")

    # 6. Generate Summary Table (Final Round multi-seed mean +/- std)
    final_rounds_df = perf_df.loc[perf_df.groupby(["scenario", "epsilon", "aggregation_method", "seed"])["round"].idxmax()]
    summary_records = []
    for (scen, eps_val, meth), group in final_rounds_df.groupby(["scenario", "epsilon", "aggregation_method"]):
        n_s = len(group)
        summary_records.append({
            "scenario": scen,
            "epsilon": eps_val,
            "aggregation_method": meth,
            "num_seeds": n_s,
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_std": group["accuracy"].std(ddof=0) if n_s > 1 else 0.0,
            "precision_mean": group["precision"].mean(),
            "precision_std": group["precision"].std(ddof=0) if n_s > 1 else 0.0,
            "recall_mean": group["recall"].mean(),
            "recall_std": group["recall"].std(ddof=0) if n_s > 1 else 0.0,
            "f1_mean": group["f1"].mean(),
            "f1_std": group["f1"].std(ddof=0) if n_s > 1 else 0.0,
            "fpr_mean": group["fpr"].mean(),
            "fpr_std": group["fpr"].std(ddof=0) if n_s > 1 else 0.0,
            "roc_auc_mean": group["roc_auc"].mean(),
            "roc_auc_std": group["roc_auc"].std(ddof=0) if n_s > 1 else 0.0,
            "malicious_weight_mean": group["malicious_weight_sum"].mean(),
            "malicious_weight_std": group["malicious_weight_sum"].std(ddof=0) if n_s > 1 else 0.0,
            "honest_weight_mean": group["honest_weight_sum"].mean(),
            "honest_weight_std": group["honest_weight_sum"].std(ddof=0) if n_s > 1 else 0.0,
            "trust_gap_mean": group["trust_gap"].mean(),
            "trust_gap_std": group["trust_gap"].std(ddof=0) if n_s > 1 else 0.0,
            "detection_auc_mean": group["detection_auc"].mean(),
            "detection_auc_std": group["detection_auc"].std(ddof=0) if n_s > 1 else 0.0,
            "detection_f1_mean": group["detection_f1"].mean(),
            "detection_f1_std": group["detection_f1"].std(ddof=0) if n_s > 1 else 0.0
        })

    summary_df = pd.DataFrame(summary_records)
    sum_csv = os.path.join(output_dir, "dp_summary.csv")
    summary_df.to_csv(sum_csv, index=False)
    logger.info(f"Saved DP evaluation summary to: {sum_csv}")

    # 7. Generate 10 Publication-Quality Plots
    logger.info("\n>>> Generating 10 Publication-Ready Visualizations in plots/ ...")
    plot_fig1_f1_vs_epsilon(summary_df, os.path.join(plots_dir, "fig1_f1_vs_epsilon.png"))
    plot_fig2_accuracy_vs_epsilon(summary_df, os.path.join(plots_dir, "fig2_accuracy_vs_epsilon.png"))
    plot_fig3_shap_similarity_vs_epsilon(expl_df, os.path.join(plots_dir, "fig3_shap_similarity_vs_epsilon.png"))
    plot_fig4_topk_stability_vs_epsilon(expl_df, os.path.join(plots_dir, "fig4_topk_stability_vs_epsilon.png"))
    plot_fig5_explanation_fidelity_vs_epsilon(expl_df, os.path.join(plots_dir, "fig5_explanation_fidelity_vs_epsilon.png"))
    plot_fig6_honest_trust_vs_epsilon(trust_df, os.path.join(plots_dir, "fig6_honest_trust_vs_epsilon.png"))
    plot_fig7_malicious_trust_vs_epsilon(trust_df, os.path.join(plots_dir, "fig7_malicious_trust_vs_epsilon.png"))
    plot_fig8_detection_rate_vs_epsilon(detection_df, os.path.join(plots_dir, "fig8_detection_rate_vs_epsilon.png"))
    plot_fig9_honest_fpr_vs_epsilon(detection_df, os.path.join(plots_dir, "fig9_honest_fpr_vs_epsilon.png"))
    plot_fig10_privacy_utility_tradeoff(summary_df, os.path.join(plots_dir, "fig10_privacy_utility_tradeoff.png"))

    logger.info("\n" + "=" * 95)
    logger.info("   STAGE I: DIFFERENTIAL PRIVACY EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info(f"   Outputs available in: {output_dir}")
    logger.info("=" * 95)


if __name__ == "__main__":
    args = parse_args()
    eps_parsed = [float(e.strip()) for e in args.epsilons.split(",") if e.strip()] if args.epsilons else None
    seeds_parsed = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] if args.seeds else None

    run_stage_i_differential_privacy_suite(
        output_dir=args.output_dir,
        epsilons=eps_parsed,
        seeds=seeds_parsed,
        num_rounds=args.rounds,
        alpha_non_iid=float(args.alphas),
        clipping_norm=args.clipping_norm,
        delta=args.delta,
        device=args.device,
        sanity_check=args.sanity,
        include_ref=args.include_ref
    )
