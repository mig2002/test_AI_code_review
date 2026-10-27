import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from utils.logger import get_logger
from security.stage_j_runner import run_stage_j_experiment

logger = get_logger("Phase6_Security")

def run_phase6_security(seed: int = 1, rounds: int = 3, device: str = "cuda"):
    logger.info("=" * 90)
    logger.info(f"   PHASE 6: COMPREHENSIVE SECURITY BENCHMARK (Seed={seed}, Rounds={rounds})")
    logger.info("=" * 90)

    output_dir = "results/final_stage_j/security"
    os.makedirs(output_dir, exist_ok=True)
    dev_type = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # Stage-G Validated Threat Environments
    threat_scenarios = [
        {
            "name": "Clean (No Attack)",
            "attack_type": "none",
            "malicious_ratio": 0.0,
            "attack_kwargs": {}
        },
        {
            "name": "Sign Flipping",
            "attack_type": "sign_flipping",
            "malicious_ratio": 0.10,
            "attack_kwargs": {"intensity": 2.0}
        },
        {
            "name": "Model Poisoning",
            "attack_type": "model_poisoning",
            "malicious_ratio": 0.10,
            "attack_kwargs": {"intensity": 4.0}
        },
        {
            "name": "Label Flipping",
            "attack_type": "label_flipping",
            "malicious_ratio": 0.10,
            "attack_kwargs": {}
        },
        {
            "name": "Backdoor Attack",
            "attack_type": "backdoor",
            "malicious_ratio": 0.10,
            "attack_kwargs": {
                "trigger_features": [105, 106, 107],
                "trigger_val": 3.0,
                "target_label": 0
            }
        },
        {
            "name": "Explanation Manipulation",
            "attack_type": "explanation_manipulation",
            "malicious_ratio": 0.10,
            "attack_kwargs": {
                "strategy": "decoy_swap",
                "top_k": 5,
                "manipulation_strength": 0.25
            }
        }
    ]

    # Defense Methods to Benchmark
    defense_methods = [
        ("FedAvg", "fedavg"),
        ("Model Trust", "model_trust"),
        ("Joint Trust", "joint_trust"),
        ("Temporal Trust (Proposed)", "temporal_trust"),
        ("Coordinate Median", "median"),
        ("Trimmed Mean", "trimmed_mean"),
        ("Krum", "krum"),
        ("Multi-Krum", "multi_krum")
    ]

    security_results = []
    t_start = time.time()

    for threat in threat_scenarios:
        threat_name = threat["name"]
        atk_type = threat["attack_type"]
        mal_ratio = threat["malicious_ratio"]
        atk_kwargs = threat["attack_kwargs"]

        logger.info(f"\n=========================================================================")
        logger.info(f"   EVALUATING THREAT ENVIRONMENT: [{threat_name}]")
        logger.info(f"=========================================================================")

        for disp_name, agg_key in defense_methods:
            exp_id = f"sec_{atk_type}_{agg_key}_seed{seed}"
            logger.info(f"Running Defense: {disp_name} against {threat_name}...")

            res = run_stage_j_experiment(
                experiment_id=exp_id,
                attack_type=atk_type,
                aggregation_method=agg_key,
                malicious_ratio=mal_ratio,
                seed=seed,
                num_rounds=rounds,
                attack_kwargs=atk_kwargs,
                device=dev_type,
                phase="PHASE_6_SECURITY",
                save_results_dir=output_dir
            )

            fin = res["final_summary"]

            record = {
                "threat_scenario": threat_name,
                "attack_type": atk_type,
                "defense_method": disp_name,
                "aggregation_key": agg_key,
                "seed": seed,
                "accuracy": fin["accuracy"],
                "precision": fin["precision"],
                "recall": fin["recall"],
                "f1": fin["f1"],
                "fpr": fin["fpr"],
                "fnr": fin["fnr"],
                "roc_auc": fin["roc_auc"],
                "asr": fin.get("asr", 0.0),
                "malicious_detection_rate": fin["detection_recall"],
                "honest_false_positive_rate": fin["detection_fpr"],
                "malicious_aggregation_weight": fin["malicious_weight_sum"],
                "trust_gap": fin["trust_gap"],
                "runtime_s": res["overhead"]["total_experiment_time"]
            }
            security_results.append(record)

    total_time = time.time() - t_start

    # Save summary dataframe
    df_sec = pd.DataFrame(security_results)
    summary_csv = os.path.join(output_dir, f"security_benchmark_seed{seed}.csv")
    df_sec.to_csv(summary_csv, index=False)

    summary_json = os.path.join(output_dir, f"security_benchmark_seed{seed}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(security_results, f, indent=2)

    # Print Table Summary
    print("\n" + "=" * 125)
    print(f"   PHASE 6: COMPREHENSIVE SECURITY BENCHMARK RESULTS (SEED {seed})")
    print("=" * 125)
    fmt = "{:<25} {:<24} {:<8} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10}"
    print(fmt.format("Threat Scenario", "Defense Method", "Acc", "F1", "AUC", "ASR", "Mal-Detect", "Mal-Weight", "Trust Gap"))
    print("-" * 125)
    for r in security_results:
        asr_str = f"{r['asr']*100:.1f}%" if r["attack_type"] == "backdoor" else "N/A"
        print(fmt.format(
            r["threat_scenario"],
            r["defense_method"],
            f"{r['accuracy']*100:.1f}%",
            f"{r['f1']*100:.1f}%",
            f"{r['roc_auc']*100:.1f}%",
            asr_str,
            f"{r['malicious_detection_rate']*100:.1f}%",
            f"{r['malicious_aggregation_weight']:.4f}",
            f"{r['trust_gap']:.4f}"
        ))
    print("-" * 125)
    print(f"Total Phase 6 Execution Runtime: {total_time:.2f} seconds.")
    print(f"Saved results to: {summary_csv}")
    print("=" * 125 + "\n")

    return security_results

if __name__ == "__main__":
    run_phase6_security()
