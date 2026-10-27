import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from utils.metrics import compute_binary_metrics
from data.preprocessing import NBaIoTDataPreprocessor
from data.download_nbaiot import get_nbaiot_feature_names, prepare_nbaiot_data
from data.nbaiot_dataset import create_dataloaders, NBaIoTDataset
from models.lstm import LSTMIDS
from security.stage_j_runner import run_stage_j_experiment
from xai.shap_explainer import SHAPExplainer
from xai.explanation_metrics import (
    compute_explanation_cosine_similarity,
    compute_top_k_jaccard,
    compute_explanation_fidelity
)

logger = get_logger("Phase5_Ablation")

def run_phase5_ablation(seed: int = 1, rounds: int = 3, device: str = "cuda"):
    logger.info("=" * 85)
    logger.info(f"   PHASE 5: COMPONENT ABLATION STUDY (Seed={seed}, Threat=Model Poisoning)")
    logger.info("=" * 85)

    output_dir = "results/final_stage_j/ablation"
    os.makedirs(output_dir, exist_ok=True)
    dev_type = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # 1. Data Preparation
    data_dir = "data/nbaiot"
    prepare_nbaiot_data(data_dir=data_dir, sample_size=3000, seed=seed)
    preprocessor = NBaIoTDataPreprocessor(
        data_dir=data_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    device_names = [
        "Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell",
        "Philips_B120N10_Baby_Monitor", "Provision_PT_737E_Security_Camera",
        "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
        "SimpleHome_XCS7_1002_WHT_Security_Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera"
    ]
    client_data, (global_test_x, global_test_y) = preprocessor.load_and_preprocess_all(device_names)
    feature_names = get_nbaiot_feature_names()

    # Precompute Reference Clean Attribution on an untainted baseline model
    checkpoint_path = "results/global_lstm_step1.pt"
    ref_model = LSTMIDS(115, 64, 2, 32).to("cpu")
    if os.path.exists(checkpoint_path):
        ref_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    ref_model.eval()

    benign_idx = np.where(global_test_y == 0)[0][:10]
    bg_samples = global_test_x[benign_idx]
    eval_x_sample = global_test_x[:20]

    ref_explainer = SHAPExplainer(model=ref_model, background_data=bg_samples, feature_names=feature_names, device="cpu")
    ref_shap = ref_explainer.compute_shap_values(eval_x_sample)
    ref_expl_vector = np.mean(np.abs(ref_shap), axis=0)

    # Conceptual Ablation Configurations
    ablation_setups = [
        {
            "id": "A0",
            "name": "FedAvg (No Trust)",
            "aggregation": "fedavg",
            "dp": None
        },
        {
            "id": "A1",
            "name": "FedAvg + Model Trust (M)",
            "aggregation": "model_trust",
            "dp": None
        },
        {
            "id": "A2",
            "name": "Model + Explanation Trust (M+E static)",
            "aggregation": "joint_trust",
            "dp": None
        },
        {
            "id": "A3",
            "name": "Joint Trust (T)",
            "aggregation": "joint_trust",
            "dp": None
        },
        {
            "id": "A4",
            "name": "Joint Trust + Temporal/Historical Trust (Full Proposed)",
            "aggregation": "temporal_trust",
            "dp": None
        },
        {
            "id": "A5",
            "name": "Full Proposed + Differential Privacy (eps=10.0)",
            "aggregation": "temporal_trust",
            "dp": {"enabled": True, "epsilon": 10.0, "delta": 1e-5, "clipping_norm": 1.0}
        }
    ]

    ablation_results = []
    t_start = time.time()

    for config in ablation_setups:
        cfg_id = config["id"]
        cfg_name = config["name"]
        agg_method = config["aggregation"]
        dp_cfg = config["dp"]

        logger.info(f"\n=========================================================================")
        logger.info(f"   EVALUATING ABLATION CONFIG {cfg_id}: [{cfg_name}]")
        logger.info(f"=========================================================================")

        exp_id = f"ablation_{cfg_id.lower()}_seed{seed}"
        res = run_stage_j_experiment(
            experiment_id=exp_id,
            attack_type="model_poisoning",
            aggregation_method=agg_method,
            malicious_ratio=0.10,  # 1 attacker
            seed=seed,
            num_rounds=rounds,
            attack_kwargs={"intensity": 4.0},
            dp_config=dp_cfg,
            device=dev_type,
            preprocessed_data=(client_data, (global_test_x, global_test_y)),
            phase="PHASE_5_ABLATION",
            save_results_dir=output_dir
        )

        fin = res["final_summary"]

        # Compute post-training SHAP metrics on final global model
        final_model = LSTMIDS(115, 64, 2, 32).to("cpu")
        final_model.load_state_dict(res["global_model_weights"])
        final_model.eval()

        explainer = SHAPExplainer(model=final_model, background_data=bg_samples, feature_names=feature_names, device="cpu")
        shap_vals = explainer.compute_shap_values(eval_x_sample)
        final_expl_vector = np.mean(np.abs(shap_vals), axis=0)

        cos_sim = compute_explanation_cosine_similarity(ref_expl_vector, final_expl_vector)
        top_k_ref = np.argsort(ref_expl_vector)[-10:]
        top_k_final = np.argsort(final_expl_vector)[-10:]
        jaccard = compute_top_k_jaccard(top_k_ref, top_k_final)
        fid = compute_explanation_fidelity(final_model, eval_x_sample, list(top_k_final), device="cpu")

        # False positive rate for honest client detection
        honest_fpr = fin["detection_fpr"]
        mal_det_rate = fin["detection_recall"]

        record = {
            "config_id": cfg_id,
            "configuration": cfg_name,
            "accuracy": fin["accuracy"],
            "precision": fin["precision"],
            "recall": fin["recall"],
            "f1": fin["f1"],
            "fpr": fin["fpr"],
            "fnr": fin["fnr"],
            "roc_auc": fin["roc_auc"],
            "shap_similarity": cos_sim,
            "top_k_stability": jaccard,
            "explanation_fidelity": fid,
            "malicious_detection_rate": mal_det_rate,
            "honest_false_positive_rate": honest_fpr,
            "malicious_aggregation_weight": fin["malicious_weight_sum"],
            "trust_gap": fin["trust_gap"],
            "runtime_s": res["overhead"]["total_experiment_time"]
        }
        ablation_results.append(record)

    total_time = time.time() - t_start

    # Save summary dataframe
    df_ablation = pd.DataFrame(ablation_results)
    summary_csv = os.path.join(output_dir, f"ablation_study_seed{seed}.csv")
    df_ablation.to_csv(summary_csv, index=False)

    summary_json = os.path.join(output_dir, f"ablation_study_seed{seed}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # Print Publication Table
    print("\n" + "=" * 125)
    print("   PHASE 5: COMPONENT ABLATION STUDY RESULTS (MODEL POISONING THREAT, SEED 1)")
    print("=" * 125)
    fmt = "{:<5} {:<38} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10} {:<12}"
    print(fmt.format("ID", "Configuration", "Acc", "F1", "AUC", "SHAP-Sim", "TopK-Stab", "Fidelity", "Mal-Detect", "Mal-Weight"))
    print("-" * 125)
    for r in ablation_results:
        print(fmt.format(
            r["config_id"],
            r["configuration"],
            f"{r['accuracy']*100:.1f}%",
            f"{r['f1']*100:.1f}%",
            f"{r['roc_auc']*100:.1f}%",
            f"{r['shap_similarity']:.3f}",
            f"{r['top_k_stability']:.3f}",
            f"{r['explanation_fidelity']:.3f}",
            f"{r['malicious_detection_rate']*100:.1f}%",
            f"{r['malicious_aggregation_weight']:.4f}"
        ))
    print("-" * 125)
    print(f"Total Phase 5 Execution Runtime: {total_time:.2f} seconds.")
    print(f"Saved results to: {summary_csv}")
    print("=" * 125 + "\n")

    return ablation_results

if __name__ == "__main__":
    run_phase5_ablation()
