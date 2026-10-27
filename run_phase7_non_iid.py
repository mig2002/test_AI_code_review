import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from data.preprocessing import NBaIoTDataPreprocessor
from data.download_nbaiot import prepare_nbaiot_data
from data.non_iid_partition import build_non_iid_benchmark_data
from security.stage_j_runner import run_stage_j_experiment

logger = get_logger("Phase7_NonIID")

def run_phase7_non_iid(seed: int = 1, rounds: int = 3, device: str = "cuda"):
    logger.info("=" * 90)
    logger.info(f"   PHASE 7: NON-IID / DATA HETEROGENEITY BENCHMARK (Seed={seed}, Rounds={rounds})")
    logger.info("=" * 90)

    output_dir = "results/final_stage_j/non_iid"
    os.makedirs(output_dir, exist_ok=True)
    dev_type = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    alphas = [10.0, 1.0, 0.5, 0.1]
    methods = [
        ("FedAvg", "fedavg"),
        ("Model Trust", "model_trust"),
        ("Joint Trust", "joint_trust"),
        ("Temporal Trust (Proposed)", "temporal_trust")
    ]

    # 1. Prepare Base Preprocessed Data
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
    base_client_data, (global_test_x, global_test_y) = preprocessor.load_and_preprocess_all(device_names)

    non_iid_results = []
    t_start = time.time()

    for alpha in alphas:
        logger.info(f"\n=========================================================================")
        logger.info(f"   EVALUATING DIRICHLET CONCENTRATION: alpha = {alpha}")
        logger.info(f"=========================================================================")

        # Construct Dirichlet Non-IID Partition
        non_iid_client_data, dist_stats = build_non_iid_benchmark_data(
            processed_client_data=base_client_data,
            global_test=(global_test_x, global_test_y),
            device_names=device_names,
            alpha=alpha,
            seed=seed
        )

        for disp_name, agg_key in methods:
            exp_id = f"non_iid_a{str(alpha).replace('.', '_')}_{agg_key}_seed{seed}"
            logger.info(f"Running {disp_name} at alpha={alpha}...")

            res = run_stage_j_experiment(
                experiment_id=exp_id,
                attack_type="model_poisoning",
                aggregation_method=agg_key,
                malicious_ratio=0.10,
                seed=seed,
                num_rounds=rounds,
                attack_kwargs={"intensity": 4.0},
                alpha_dirichlet=alpha,
                preprocessed_data=(non_iid_client_data, (global_test_x, global_test_y)),
                device=dev_type,
                phase="PHASE_7_NON_IID",
                save_results_dir=output_dir
            )

            fin = res["final_summary"]

            record = {
                "alpha": alpha,
                "method": disp_name,
                "aggregation_key": agg_key,
                "seed": seed,
                "accuracy": fin["accuracy"],
                "precision": fin["precision"],
                "recall": fin["recall"],
                "f1": fin["f1"],
                "fpr": fin["fpr"],
                "fnr": fin["fnr"],
                "roc_auc": fin["roc_auc"],
                "avg_honest_trust": fin["avg_honest_final_trust"],
                "avg_malicious_trust": fin["avg_malicious_final_trust"],
                "trust_gap": fin["trust_gap"],
                "malicious_detection_rate": fin["detection_recall"],
                "honest_false_positive_rate": fin["detection_fpr"],
                "malicious_aggregation_weight": fin["malicious_weight_sum"],
                "honest_aggregation_weight": fin["honest_weight_sum"],
                "runtime_s": res["overhead"]["total_experiment_time"]
            }
            non_iid_results.append(record)

    total_time = time.time() - t_start

    # Save summary dataframe
    df_non_iid = pd.DataFrame(non_iid_results)
    summary_csv = os.path.join(output_dir, f"non_iid_benchmark_seed{seed}.csv")
    df_non_iid.to_csv(summary_csv, index=False)

    summary_json = os.path.join(output_dir, f"non_iid_benchmark_seed{seed}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(non_iid_results, f, indent=2)

    # Print Table
    print("\n" + "=" * 125)
    print(f"   PHASE 7: NON-IID / DATA HETEROGENEITY BENCHMARK RESULTS (SEED {seed})")
    print("=" * 125)
    fmt = "{:<8} {:<26} {:<8} {:<8} {:<8} {:<12} {:<12} {:<10} {:<12} {:<12}"
    print(fmt.format("Alpha", "Method", "Acc", "F1", "AUC", "Honest-Tr", "Mal-Tr", "Trust-Gap", "Mal-Detect", "Mal-Weight"))
    print("-" * 125)
    for r in non_iid_results:
        print(fmt.format(
            str(r["alpha"]),
            r["method"],
            f"{r['accuracy']*100:.1f}%",
            f"{r['f1']*100:.1f}%",
            f"{r['roc_auc']*100:.1f}%",
            f"{r['avg_honest_trust']:.4f}",
            f"{r['avg_malicious_trust']:.4f}",
            f"{r['trust_gap']:.4f}",
            f"{r['malicious_detection_rate']*100:.1f}%",
            f"{r['malicious_aggregation_weight']:.4f}"
        ))
    print("-" * 125)
    print(f"Total Phase 7 Execution Runtime: {total_time:.2f} seconds.")
    print(f"Saved results to: {summary_csv}")
    print("=" * 125 + "\n")

    return non_iid_results

if __name__ == "__main__":
    run_phase7_non_iid()
