import os
import sys
import yaml
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Ensure root directory is on path
sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from utils.metrics import compute_binary_metrics, print_metrics_table
from data.preprocessing import NBaIoTDataPreprocessor
from data.download_nbaiot import get_nbaiot_feature_names, prepare_nbaiot_data
from data.nbaiot_dataset import create_dataloaders, NBaIoTDataset
from models.lstm import LSTMIDS
from federated.client import IoTClient
from federated.aggregation import (
    compute_model_trust_weights,
    compute_joint_trust_weights,
    compute_temporal_trust_weights,
    aggregate_models
)
from security.model_trust import (
    ModelTrustEvaluator,
    calculate_client_update
)
from security.explanation_trust import (
    ExplanationTrustEvaluator,
    get_shap_representation
)
from security.joint_trust import (
    calculate_joint_trust,
    compute_client_joint_trust
)
from security.temporal_trust import (
    TemporalTrustManager,
    update_temporal_trust,
    calculate_final_trust
)
from security.attacks import (
    BaseAttack,
    NoAttack,
    ModelPoisoningAttack,
    apply_model_poisoning,
    select_malicious_clients,
    create_attack
)
from xai.shap_explainer import SHAPExplainer

logger = get_logger("StageG3_Model_Poisoning")

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_lstm_model(model_config: dict, device: str = "cpu") -> LSTMIDS:
    return LSTMIDS(
        input_dim=model_config.get("input_dim", 115),
        hidden_dim=model_config.get("hidden_dim", 64),
        num_layers=model_config.get("num_layers", 2),
        dense_dim=model_config.get("dense_dim", 32),
        dropout=model_config.get("dropout", 0.2)
    ).to(device)

def evaluate_model_on_test_loader(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Evaluates a PyTorch model on test_loader and computes binary IDS metrics."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(dev)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(dev)
            probs = torch.sigmoid(model(x_b)).cpu().numpy().ravel()
            all_preds.extend(probs)
            all_targets.extend(y_b.numpy().ravel())
            
    return compute_binary_metrics(all_targets, all_preds)

def compute_detection_metrics(
    true_malicious_mask: List[int],
    final_trust_scores: List[float],
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Evaluates malicious client detection where anomaly score = (1.0 - FinalTrust).
    A client is flagged as malicious if FinalTrust <= threshold.
    """
    y_true = np.asarray(true_malicious_mask, dtype=int)
    anomaly_scores = 1.0 - np.asarray(final_trust_scores, dtype=float)
    y_pred = (anomaly_scores >= (1.0 - threshold)).astype(int)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, anomaly_scores))
        else:
            auc = 1.0 if tp > 0 else 0.5
    except Exception:
        auc = 0.5
        
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec,
        "recall_detection_rate": rec,
        "false_positive_rate": fpr,
        "f1": f1,
        "auc": auc
    }

def run_model_poisoning_benchmark(
    config_path: str = "configs/config.yaml",
    rounds: int = 3,
    malicious_ratio: float = 0.10,
    intensity: float = 1.0,
    start_round: int = 1,
    seed: int = 42,
    device: str = "cuda"
):
    output_dir = "results/attacks/model_poisoning"
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    feature_names = get_nbaiot_feature_names()
    dev_type = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    
    logger.info("=" * 115)
    logger.info("   STAGE G3: SECURITY / ADVERSARIAL EVALUATION — MODEL POISONING ATTACK")
    logger.info(f"   Attack: Delta_i' = -{intensity:.1f} * Delta_i | Ratio: {malicious_ratio:.2f} | Start Round: {start_round} | Seed: {seed}")
    logger.info("=" * 115)
    
    # 1. Load Data
    data_dir = cfg["dataset"]["data_dir"]
    prepare_nbaiot_data(data_dir=data_dir, sample_size=cfg["dataset"].get("sample_size_per_client", 3000), seed=seed)
    
    preprocessor = NBaIoTDataPreprocessor(
        data_dir=data_dir,
        train_ratio=cfg["dataset"]["train_ratio"],
        val_ratio=cfg["dataset"]["val_ratio"],
        test_ratio=cfg["dataset"]["test_ratio"],
        seed=seed
    )
    device_names = cfg["clients"]["device_names"]
    num_clients = len(device_names)
    client_data, (global_test_x, global_test_y) = preprocessor.load_and_preprocess_all(device_names)
    
    batch_size = cfg["federated"]["batch_size"]
    global_test_dataset = NBaIoTDataset(global_test_x, global_test_y)
    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size, shuffle=False)
    
    # Background and evaluation samples for SHAP
    benign_idx = np.where(global_test_y == 0)[0][:40]
    attack_idx = np.where(global_test_y == 1)[0][:20]
    bg_samples = global_test_x[benign_idx]
    eval_x_sample = np.vstack([global_test_x[benign_idx[:20]], global_test_x[attack_idx]])

    # 2. Malicious Client Selection
    malicious_device_names = select_malicious_clients(
        total_clients=num_clients,
        malicious_ratio=malicious_ratio,
        seed=seed,
        client_ids=device_names
    )
    logger.info(f"Selected Malicious Clients ({len(malicious_device_names)}/{num_clients}): {malicious_device_names}")
    malicious_mask = [1 if dev in malicious_device_names else 0 for dev in device_names]
    
    # 3. Instantiate Attack
    attack = ModelPoisoningAttack(
        malicious_client_ids=malicious_device_names,
        intensity=intensity,
        start_round=start_round,
        seed=seed
    )

    # 4. Client Pool Factory
    def build_client_pool():
        clients = []
        for client_id, dev in enumerate(device_names):
            splits = client_data[dev]
            train_l, val_l, test_l = create_dataloaders(
                splits["x_train"], splits["y_train"],
                splits["x_val"], splits["y_val"],
                splits["x_test"], splits["y_test"],
                batch_size=batch_size
            )
            c = IoTClient(
                client_id=client_id,
                device_name=dev,
                train_loader=train_l,
                val_loader=val_l,
                test_loader=test_l,
                model_config=cfg["model"],
                device=dev_type
            )
            clients.append(c)
        return clients

    initial_model = create_lstm_model(cfg["model"], device=dev_type)
    initial_weights = {k: v.cpu().clone() for k, v in initial_model.state_dict().items()}
    
    clean_init_metrics = evaluate_model_on_test_loader(initial_model, global_test_loader, dev_type)
    logger.info(f"Initial Untrained Global Model F1: {clean_init_metrics['f1'] * 100:.2f}% | Acc: {clean_init_metrics['accuracy'] * 100:.2f}%")

    # =========================================================================
    # PHASE A: CLEAN BASELINE EXPERIMENT (0% Malicious Clients)
    # =========================================================================
    logger.info("\n>>> PHASE A: Running Clean Baseline Experiment (0% Malicious, FedAvg)...")
    clean_clients = build_client_pool()
    clean_server_model = create_lstm_model(cfg["model"], device=dev_type)
    clean_server_model.load_state_dict(initial_weights)
    
    clean_round_metrics = []
    for r in range(1, rounds + 1):
        g_weights = {k: v.cpu().clone() for k, v in clean_server_model.state_dict().items()}
        updates, counts = [], []
        for client in clean_clients:
            client.receive_global_weights(g_weights)
            client.train_local(epochs=1, lr=cfg["federated"]["learning_rate"])
            w_i, n_i = client.get_model_update()
            updates.append(w_i)
            counts.append(n_i)
        agg_w, _ = aggregate_models(updates, counts, None, method="fedavg")
        clean_server_model.load_state_dict(agg_w)
        m = evaluate_model_on_test_loader(clean_server_model, global_test_loader, dev_type)
        m["round"] = r
        clean_round_metrics.append(m)
        logger.info(f"Clean Round {r} FedAvg -> F1: {m['f1']*100:.2f}% | Acc: {m['accuracy']*100:.2f}% | FPR: {m['fpr']*100:.2f}%")

    final_clean_f1 = clean_round_metrics[-1]["f1"]
    final_clean_acc = clean_round_metrics[-1]["accuracy"]

    # =========================================================================
    # PHASE B: ADVERSARIAL EVALUATION UNDER MODEL POISONING
    # Comparing: FedAvg vs Model Trust vs Joint Trust vs Temporal Trust
    # =========================================================================
    logger.info("\n>>> PHASE B: Multi-Round Comparative Evaluation across 4 Aggregation Schemes...")
    aggregation_schemes = ["fedavg", "model_trust", "joint_trust", "temporal_trust"]
    
    comparison_results = []
    client_trajectories = []
    detection_records = []

    scheme_models = {
        scheme: create_lstm_model(cfg["model"], device=dev_type) for scheme in aggregation_schemes
    }
    for sm in scheme_models.values():
        sm.load_state_dict(initial_weights)
        
    scheme_temporal_managers = {
        scheme: TemporalTrustManager(lambda_decay=0.70, beta=0.50) for scheme in aggregation_schemes
    }

    model_evaluator = ModelTrustEvaluator(w_direction=0.50, w_norm=0.20, w_layer=0.30, reference_strategy="leave_one_out")
    expl_evaluator = ExplanationTrustEvaluator(w_similarity=0.40, w_jaccard=0.30, w_fidelity=0.30, top_k=5, reference_strategy="leave_one_out")
    
    clients = build_client_pool()

    for r in range(1, rounds + 1):
        print(f"\n" + "=" * 125)
        print(f"   --- COMMUNICATION ROUND {r}/{rounds} UNDER MODEL POISONING (c={intensity:.1f}) ---")
        print("=" * 125)
        
        main_global_weights = {k: v.cpu().clone() for k, v in scheme_models["temporal_trust"].state_dict().items()}
        
        manipulated_deltas = {}
        manipulated_weights = {}
        sample_counts = []
        client_models_dict = {}
        client_eval_x_dict = {}

        for i, client in enumerate(clients):
            dev = client.device_name
            client.receive_global_weights(main_global_weights)
            client.train_local(epochs=1, lr=cfg["federated"]["learning_rate"])
            honest_w, n_i = client.get_model_update()
            sample_counts.append(n_i)
            
            # 1. Calculate Honest Delta relative to pre-round global model
            honest_delta = calculate_client_update(main_global_weights, honest_w)
            
            # 2. Apply Model Poisoning Attack: Delta'_i = -c * Delta_i
            adv_delta, adv_w = attack.apply_update_attack(
                delta=honest_delta,
                global_weights=main_global_weights,
                client_id=dev,
                round_idx=r
            )
            
            # Verify numerical safety
            for pk, pv in adv_delta.items():
                assert not torch.isnan(pv).any(), f"NaN in poisoned delta '{pk}' for {dev}"
                assert not torch.isinf(pv).any(), f"Inf in poisoned delta '{pk}' for {dev}"
                
            manipulated_deltas[dev] = adv_delta
            manipulated_weights[dev] = adv_w
            
            # Load poisoned weights into client model for local SHAP evaluation
            client.model.load_state_dict({k: v.to(dev_type) for k, v in adv_w.items()})
            client_models_dict[dev] = client.model
            client_eval_x_dict[dev] = eval_x_sample

        # Trust Calculations
        # 1. Model Trust (M_i)
        m_scores, _ = model_evaluator.evaluate_all(manipulated_deltas)
        m_trust_list = [m_scores[c.device_name]["model_trust"] for c in clients]

        # 2. Explanation Trust (E_i)
        client_explanations = {}
        for client in clients:
            explainer = SHAPExplainer(model=client.model, background_data=bg_samples, feature_names=feature_names, device="cpu")
            shap_vals = explainer.compute_shap_values(eval_x_sample)
            client_explanations[client.device_name] = get_shap_representation(shap_vals)

        e_scores, _ = expl_evaluator.evaluate_all(
            client_explanations=client_explanations,
            client_models=client_models_dict,
            x_eval_dict=client_eval_x_dict,
            feature_names=feature_names,
            device=dev_type
        )
        e_trust_list = [e_scores[c.device_name]["explanation_trust"] for c in clients]

        # 3. Joint Trust (T_i)
        j_trust_dict = compute_client_joint_trust(m_scores, e_scores, alpha=0.50)
        j_trust_list = [j_trust_dict[c.device_name] for c in clients]

        # 4. Historical (H_i) and Final Trust (F_i)
        h_trust_list = []
        f_trust_list = []
        for i, client in enumerate(clients):
            rec = scheme_temporal_managers["temporal_trust"].update_client(
                client_name=client.device_name,
                joint_trust=j_trust_list[i],
                round_idx=r,
                model_trust=m_trust_list[i],
                explanation_trust=e_trust_list[i]
            )
            h_trust_list.append(rec["historical_trust"])
            f_trust_list.append(rec["final_trust"])

        # Compute Weights for All 4 Methods
        fedavg_w, model_w = compute_model_trust_weights(sample_counts, m_trust_list)
        _, joint_w = compute_joint_trust_weights(sample_counts, j_trust_list)
        _, final_w = compute_temporal_trust_weights(sample_counts, f_trust_list)

        # Print Client Trust Table
        header = f"{'ID':<2} | {'Device':<32} | {'Malicious?':<10} | {'Model M_i':<9} | {'Expl E_i':<9} | {'Joint T_i':<9} | {'Hist H_i':<9} | {'Final F_i':<9} | {'FedAvg W':<8} | {'Final W':<8}"
        print(header)
        print("-" * len(header))
        for i, client in enumerate(clients):
            is_m_str = "MALICIOUS" if client.device_name in malicious_device_names else "Honest"
            print(
                f"{i:<2} | {client.device_name:<32} | {is_m_str:<10} | "
                f"{m_trust_list[i]:9.4f} | {e_trust_list[i]:9.4f} | {j_trust_list[i]:9.4f} | "
                f"{h_trust_list[i]:9.4f} | {f_trust_list[i]:9.4f} | "
                f"{fedavg_w[i]:8.4f} | {final_w[i]:8.4f}"
            )
            client_trajectories.append({
                "round": r,
                "client_id": i,
                "device_name": client.device_name,
                "malicious": 1 if client.device_name in malicious_device_names else 0,
                "attack_type": "model_poisoning",
                "attack_intensity": intensity,
                "attack_start_round": start_round,
                "sample_count": sample_counts[i],
                "model_trust": m_trust_list[i],
                "explanation_trust": e_trust_list[i],
                "joint_trust": j_trust_list[i],
                "historical_trust": h_trust_list[i],
                "final_trust": f_trust_list[i],
                "fedavg_weight": fedavg_w[i],
                "model_trust_weight": model_w[i],
                "joint_trust_weight": joint_w[i],
                "final_trust_weight": final_w[i]
            })

        # Malicious Client Detection Metrics
        det = compute_detection_metrics(malicious_mask, f_trust_list, threshold=0.50)
        det["round"] = r
        det["intensity"] = intensity
        detection_records.append(det)
        logger.info(
            f"Malicious Client Detection (Threshold 0.50): "
            f"TP={det['tp']}, FP={det['fp']}, Precision={det['precision']*100:.1f}%, "
            f"Recall={det['recall_detection_rate']*100:.1f}%, AUC={det['auc']:.4f}"
        )

        # Multi-Scheme Aggregation & Evaluation
        updates_list = [manipulated_weights[c.device_name] for c in clients]
        scheme_weights_map = {
            "fedavg": fedavg_w,
            "model_trust": model_w,
            "joint_trust": joint_w,
            "temporal_trust": final_w
        }

        print(f"\n   >>> Performance Comparison across Aggregation Schemes (Round {r}):")
        for scheme in aggregation_schemes:
            if scheme == "fedavg":
                agg_dict, _ = aggregate_models(updates_list, sample_counts, None, method="fedavg")
            elif scheme == "model_trust":
                agg_dict, _ = aggregate_models(updates_list, sample_counts, m_trust_list, method="model_trust")
            elif scheme == "joint_trust":
                agg_dict, _ = aggregate_models(updates_list, sample_counts, j_trust_list, method="joint_trust")
            elif scheme == "temporal_trust":
                agg_dict, _ = aggregate_models(updates_list, sample_counts, f_trust_list, method="temporal_trust")
                
            scheme_models[scheme].load_state_dict(agg_dict)
            eval_metrics = evaluate_model_on_test_loader(scheme_models[scheme], global_test_loader, dev_type)
            
            mal_w_sum = sum([scheme_weights_map[scheme][idx] for idx, is_m in enumerate(malicious_mask) if is_m == 1])
            hon_w_sum = sum([scheme_weights_map[scheme][idx] for idx, is_m in enumerate(malicious_mask) if is_m == 0])
            
            avg_mal_trust = float(np.mean([f_trust_list[idx] for idx, is_m in enumerate(malicious_mask) if is_m == 1])) if any(malicious_mask) else 0.0
            avg_hon_trust = float(np.mean([f_trust_list[idx] for idx, is_m in enumerate(malicious_mask) if is_m == 0]))
            
            f1_degradation = max(0.0, final_clean_f1 - eval_metrics["f1"])
            
            comparison_results.append({
                "round": r,
                "aggregation_method": scheme,
                "malicious_ratio": malicious_ratio,
                "attack_intensity": intensity,
                "malicious_weight_sum": mal_w_sum,
                "honest_weight_sum": hon_w_sum,
                "avg_malicious_trust": avg_mal_trust,
                "avg_honest_trust": avg_hon_trust,
                "accuracy": eval_metrics["accuracy"],
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "f1": eval_metrics["f1"],
                "fpr": eval_metrics["fpr"],
                "fnr": eval_metrics["fnr"],
                "roc_auc": eval_metrics["roc_auc"],
                "clean_f1_baseline": final_clean_f1,
                "f1_degradation": f1_degradation
            })
            
            print(
                f"   * {scheme:<14} | F1: {eval_metrics['f1']*100:6.2f}% | "
                f"Acc: {eval_metrics['accuracy']*100:6.2f}% | "
                f"FPR: {eval_metrics['fpr']*100:6.2f}% | "
                f"Malicious Wt: {mal_w_sum*100:5.2f}% | "
                f"F1 Drop: {f1_degradation*100:5.2f}%"
            )

    # 5. Save Structured CSV Results
    df_traj = pd.DataFrame(client_trajectories)
    traj_path = os.path.join(output_dir, "model_poisoning_client_trajectories.csv")
    df_traj.to_csv(traj_path, index=False)
    logger.info(f"Saved client trajectories to: {traj_path}")

    df_comp = pd.DataFrame(comparison_results)
    comp_path = os.path.join(output_dir, "model_poisoning_aggregation_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    logger.info(f"Saved aggregation comparisons to: {comp_path}")

    df_det = pd.DataFrame(detection_records)
    det_path = os.path.join(output_dir, "model_poisoning_detection_metrics.csv")
    df_det.to_csv(det_path, index=False)
    logger.info(f"Saved detection metrics to: {det_path}")

    # 6. Generate 5-Panel Validation Plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2])

    palette = {"fedavg": "#d9534f", "model_trust": "#f0ad4e", "joint_trust": "#5bc0de", "temporal_trust": "#2e7d32"}

    # Panel 1: Global F1 vs Round
    for scheme in aggregation_schemes:
        sub = df_comp[df_comp["aggregation_method"] == scheme]
        ax1.plot(sub["round"], sub["f1"] * 100, marker='o', linewidth=2.5, label=scheme.replace("_", " ").title(), color=palette.get(scheme))
    ax1.axhline(final_clean_f1 * 100, color="#333333", linestyle="--", alpha=0.7, label=f"Clean Baseline ({final_clean_f1*100:.1f}%)")
    ax1.set_title(f"1. Global IDS F1 vs Round (c={intensity:.1f})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Communication Round", fontsize=9)
    ax1.set_ylabel("Global F1-Score (%)", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower left", frameon=True, fontsize=8)

    # Panel 2: Malicious Client Final Trust vs Round
    mal_traj = df_traj[df_traj["malicious"] == 1].groupby("round")["final_trust"].mean()
    if not mal_traj.empty:
        ax2.plot(mal_traj.index, mal_traj.values, marker='^', linewidth=2.5, color="#b71c1c", label=f"Malicious Trust (c={intensity:.1f})")
    ax2.axhline(0.50, color="gray", linestyle=":", label="Detection Threshold (0.50)")
    ax2.set_title("2. Malicious Client Final Trust vs Round", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Communication Round", fontsize=9)
    ax2.set_ylabel("Final Trust F_i", fontsize=9)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(loc="upper right", frameon=True, fontsize=8)

    # Panel 3: Honest Client Final Trust vs Round
    honest_traj = df_traj[df_traj["malicious"] == 0].groupby("round")["final_trust"].mean()
    ax3.plot(honest_traj.index, honest_traj.values, marker='s', linewidth=2.5, color="#1b5e20", label="Honest Clients Trust")
    ax3.set_title("3. Honest Client Final Trust vs Round", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Communication Round", fontsize=9)
    ax3.set_ylabel("Final Trust F_i", fontsize=9)
    ax3.set_ylim(0.0, 1.05)
    ax3.legend(loc="lower right", frameon=True, fontsize=8)

    # Panel 4: Total Malicious Aggregation Weight vs Round
    for scheme in aggregation_schemes:
        sub = df_comp[df_comp["aggregation_method"] == scheme]
        ax4.plot(sub["round"], sub["malicious_weight_sum"] * 100, marker='d', linewidth=2.5, label=scheme.replace("_", " ").title(), color=palette.get(scheme))
    ax4.set_title("4. Total Malicious Aggregation Weight vs Round", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Communication Round", fontsize=9)
    ax4.set_ylabel("Weight Allocated to Attacker (%)", fontsize=9)
    max_w = df_comp["malicious_weight_sum"].max() * 100
    ax4.set_ylim(0, max(15.0, max_w * 1.35))
    ax4.legend(loc="upper right", frameon=True, fontsize=9)

    # Panel 5: Final Round Malicious Weight Comparison Bar Chart
    r_last = rounds
    last_comp = df_comp[df_comp["round"] == r_last]
    schemes_disp = [s.replace("_", " ").title() for s in last_comp["aggregation_method"]]
    weights_pct = last_comp["malicious_weight_sum"] * 100
    bars = ax5.bar(schemes_disp, weights_pct, color=[palette[s] for s in last_comp["aggregation_method"]], width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax5.set_title(f"5. Malicious Influence at Round {r_last}", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Aggregation Defense", fontsize=9)
    ax5.set_ylabel("Weight (%)", fontsize=9)
    ax5.set_ylim(0, max(15.0, max(weights_pct) * 1.35))

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "stage_g3_model_poisoning_validation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved Stage G3 Model-Poisoning validation plot to: {plot_path}")

    # Copy to artifacts directory
    artifact_plot = "C:/Users/mg631/.gemini/antigravity-ide/brain/2fc60168-5d84-4e19-92ec-ec184bfb8405/stage_g3_model_poisoning_validation.png"
    import shutil
    shutil.copyfile(plot_path, artifact_plot)
    logger.info(f"Copied plot to artifact directory: {artifact_plot}")

    print("\n" + "=" * 115)
    print("   STAGE G3 MODEL-POISONING VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 115)
    return df_comp, df_traj, df_det

def run_intensity_sweep(
    config_path: str = "configs/config.yaml",
    rounds: int = 3,
    malicious_ratio: float = 0.10,
    intensities: List[float] = [1.0, 2.0, 5.0, 10.0],
    seed: int = 42,
    device: str = "cuda"
):
    """Executes the Section 8 intensity sweep across c in {1, 2, 5, 10}."""
    output_dir = "results/attacks/model_poisoning"
    os.makedirs(output_dir, exist_ok=True)
    sweep_records = []
    
    logger.info("=" * 115)
    logger.info(f"   STARTING INTENSITY SWEEP: c in {intensities}")
    logger.info("=" * 115)
    
    for c in intensities:
        logger.info(f"\n>>> Running Model Poisoning at Intensity c = {c:.1f}...")
        df_comp, df_traj, df_det = run_model_poisoning_benchmark(
            config_path=config_path,
            rounds=rounds,
            malicious_ratio=malicious_ratio,
            intensity=c,
            seed=seed,
            device=device
        )
        
        # Take last round summary
        last_round_comp = df_comp[df_comp["round"] == rounds]
        for _, row in last_round_comp.iterrows():
            rec = dict(row)
            rec["intensity"] = c
            sweep_records.append(rec)
            
    df_sweep = pd.DataFrame(sweep_records)
    sweep_path = os.path.join(output_dir, "model_poisoning_intensity_sweep.csv")
    df_sweep.to_csv(sweep_path, index=False)
    logger.info(f"Saved complete intensity sweep to: {sweep_path}")
    return df_sweep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage G3: Model-Poisoning Adversarial Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--start_round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sweep", action="store_true", help="Run full intensity sweep across c in {1, 2, 5, 10}")
    args = parser.parse_args()

    if args.sweep:
        run_intensity_sweep(
            config_path=args.config,
            rounds=args.rounds,
            malicious_ratio=args.ratio,
            intensities=[1.0, 2.0, 5.0, 10.0],
            seed=args.seed,
            device=args.device
        )
    else:
        run_model_poisoning_benchmark(
            config_path=args.config,
            rounds=args.rounds,
            malicious_ratio=args.ratio,
            intensity=args.intensity,
            start_round=args.start_round,
            seed=args.seed,
            device=args.device
        )
