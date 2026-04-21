# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, NoReturn

import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.report import report
from ts_benchmark.common.constant import CONFIG_PATH, THIRD_PARTY_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend

sys.path.insert(0, THIRD_PARTY_PATH)

warnings.filterwarnings("ignore")


def str_to_bool(value: str) -> bool:
    if value.lower() in ['true', '1', 't']:
        return True
    elif value.lower() in ['false', '0', 'f']:
        return False
    else:
        raise ValueError("Invalid boolean value.")


def build_data_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    data_config = config_data["data_config"]
    data_config["data_name_list"] = args.data_name_list
    if args.data_set_name is not None:
        data_config["data_set_name"] = args.data_set_name
    return data_config


def build_model_config(model_name: str, adapter: str, model_hyper_params: str, config_data: Dict) -> Dict:
    model_config = config_data.get("model_config", None)
    model_config["models"] = [
        {
            "adapter": adapter,
            "model_name": model_name,
            "model_hyper_params": json.loads(model_hyper_params) if model_hyper_params else {},
        }
    ]
    return model_config


def build_evaluation_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    evaluation_config = config_data["evaluation_config"]
    evaluation_config["save_path"] = args.save_path

    metric_list = []
    if args.metrics != "all" and args.metrics is not None:
        for metric in args.metrics:
            metric = json.loads(metric)
            metric_list.append(metric)
        evaluation_config["metrics"] = metric_list

    default_strategy_args = evaluation_config["strategy_args"]
    strategy_args_updates = json.loads(args.strategy_args) if args.strategy_args else None

    if strategy_args_updates is not None:
        default_strategy_args.update(strategy_args_updates)

    if args.seed is not None:
        default_strategy_args["seed"] = args.seed
    if args.save_true_pred is not None:
        default_strategy_args["save_true_pred"] = args.save_true_pred
    default_strategy_args["deterministic"] = args.deterministic

    return evaluation_config


def build_report_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    report_config = config_data["report_config"]
    report_config["aggregate_type"] = args.aggregate_type
    report_config["save_path"] = args.save_path
    return report_config


def init_worker(env: Dict) -> NoReturn:
    sys.path.insert(0, THIRD_PARTY_PATH)
    torch.set_num_threads(1)


def run_single_model(args, model_name, adapter, model_hyper_params, save_sub_path):
    with open(os.path.join(CONFIG_PATH, args.config_path), "r") as file:
        config_data = json.load(file)

    data_config = build_data_config(args, config_data)
    model_config = build_model_config(model_name, adapter, model_hyper_params, config_data)
    evaluation_config = build_evaluation_config(args, config_data)

    original_save_path = args.save_path
    args.save_path = save_sub_path
    report_config = build_report_config(args, config_data)
    args.save_path = original_save_path

    ParallelBackend().init(
        backend=args.eval_backend,
        n_workers=args.num_workers,
        n_cpus=args.num_cpus,
        gpu_devices=args.gpus,
        default_timeout=args.timeout,
        max_tasks_per_child=args.max_tasks_per_child,
        worker_initializers=[init_worker],
    )

    try:
        log_filenames = pipeline(data_config, model_config, evaluation_config)
    finally:
        ParallelBackend().close(force=True)

    report_config["log_files_list"] = log_filenames
    if args.report_method == "csv":
        filename = get_unique_file_suffix()
        leaderboard_file_name = model_name.split(".")[-1] + "_report" + filename
        report_config["leaderboard_file_name"] = leaderboard_file_name
    report(report_config, report_method=args.report_method)

    result_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "result", save_sub_path)
    csv_files = [f for f in os.listdir(result_dir) if f.endswith(".csv")] if os.path.isdir(result_dir) else []
    if csv_files:
        latest_csv = sorted(csv_files)[-1]
        result_df = pd.read_csv(os.path.join(result_dir, latest_csv))
        val_col = [c for c in result_df.columns if c not in ('strategy_args', 'metric_name')][0]
        print(f"\n{'='*50}")
        print(f"Results for {model_name.split('.')[-1]}:")
        print(f"{'='*50}")
        for _, row in result_df.iterrows():
            mn = str(row['metric_name']).lower()
            if 'mse' in mn or 'mae' in mn:
                v = row[val_col]
                print(f"  {row['metric_name']}: {v:.6f}" if pd.notna(v) else f"  {row['metric_name']}: NaN")
        print(f"{'='*50}")

    return log_filenames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MoFo_Compare - Run original MoFo and MoFo_Circulant side by side",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--data-name-list", type=str, nargs="+", default=None)
    parser.add_argument("--data-set-name", type=str, nargs="+", default=None)

    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--strategy-args", type=str, default='{"horizon": 96}')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", type=str, default="efficient", choices=["full", "efficient", "none"])
    parser.add_argument("--save-true-pred", type=str_to_bool, default=None)

    parser.add_argument("--eval-backend", type=str, default="sequential", choices=["sequential", "ray"])
    parser.add_argument("--num-cpus", type=int, default=os.cpu_count())
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tasks-per-child", type=int, default=100)

    parser.add_argument("--aggregate_type", default="mean")
    parser.add_argument("--report-method", type=str, default="csv", choices=["dash", "csv"])
    parser.add_argument("--save-path", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.set_num_threads(3)

    DATA_NAME = args.data_name_list[0] if args.data_name_list else "ETTh1"
    BASE_SAVE = args.save_path or DATA_NAME.replace(".csv", "")

    SEQ_LEN_LIST = [96, 336, 512]

    all_results = {}

    for seq_len in SEQ_LEN_LIST:
        print()
        print("#" * 60)
        print(f"# seq_len = {seq_len}")
        print("#" * 60)

        mofe_hyper = json.dumps({
            "batch_size": 16, "d_model": 24, "horizon": 96, "lr": 0.01,
            "norm": True, "seq_len": seq_len, "patience": 10,
            "periodic": 24, "bias": 1, "cias": 1
        })
        circulant_hyper = json.dumps({
            "batch_size": 16, "d_model": 24, "horizon": 96, "lr": 0.01,
            "norm": True, "seq_len": seq_len, "patience": 10,
            "periodic": 24, "bias": 1, "cias": 1,
            "lambda_init": 0.1, "use_causal_mask": False
        })
        dualpath_hyper = json.dumps({
            "batch_size": 16, "d_model": 24, "horizon": 96, "lr": 0.01,
            "norm": True, "seq_len": seq_len, "patience": 10,
            "periodic": 24, "bias": 1, "cias": 1,
            "lambda_init": 0.1, "use_causal_mask": False,
            "use_dual_path": True, "decomp_mode": "stl",
            "dual_path_period": 0, "trend_mode": "mlp",
            "dual_path_alpha_init": 0.5
        })
        circbias_hyper = json.dumps({
            "batch_size": 16, "d_model": 24, "horizon": 96, "lr": 0.01,
            "norm": True, "seq_len": seq_len, "patience": 10,
            "periodic": 24, "bias": 1, "cias": 1,
            "lambda_init": 0.1, "use_causal_mask": False
        })

        print(f"[1/4] Running Original MoFo (seq_len={seq_len}) ...")
        run_single_model(
            args,
            model_name="time_series_library.MoFo",
            adapter="MoFo_adapter",
            model_hyper_params=mofe_hyper,
            save_sub_path=f"{BASE_SAVE}/MoFo_sl{seq_len}",
        )

        print(f"[2/4] Running MoFo_Circulant (seq_len={seq_len}) ...")
        run_single_model(
            args,
            model_name="time_series_library.MoFo_Circulant",
            adapter="MoFo_Circulant_adapter",
            model_hyper_params=circulant_hyper,
            save_sub_path=f"{BASE_SAVE}/MoFo_Circulant_sl{seq_len}",
        )

        print(f"[3/4] Running MoFo_CircBias (seq_len={seq_len}) ...")
        run_single_model(
            args,
            model_name="time_series_library.MoFo_CircBias",
            adapter="MoFo_CircBias_adapter",
            model_hyper_params=circbias_hyper,
            save_sub_path=f"{BASE_SAVE}/MoFo_CircBias_sl{seq_len}",
        )

        print(f"[4/4] Running MoFo_Circulant_DualPath (seq_len={seq_len}) ...")
        run_single_model(
            args,
            model_name="time_series_library.MoFo_Circulant_DualPath",
            adapter="MoFo_Circulant_DP_adapter",
            model_hyper_params=dualpath_hyper,
            save_sub_path=f"{BASE_SAVE}/MoFo_Circulant_DP_sl{seq_len}",
        )

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        mofe_dir = os.path.join(root_dir, "result", f"{BASE_SAVE}/MoFo_sl{seq_len}")
        circ_dir = os.path.join(root_dir, "result", f"{BASE_SAVE}/MoFo_Circulant_sl{seq_len}")
        cb_dir = os.path.join(root_dir, "result", f"{BASE_SAVE}/MoFo_CircBias_sl{seq_len}")
        dp_dir = os.path.join(root_dir, "result", f"{BASE_SAVE}/MoFo_Circulant_DP_sl{seq_len}")

        def _load_latest_metrics(res_dir):
            if not os.path.isdir(res_dir):
                return None
            csvs = [f for f in os.listdir(res_dir) if f.endswith(".csv")]
            if not csvs:
                return None
            df = pd.read_csv(os.path.join(res_dir, sorted(csvs)[-1]))
            val_col = [c for c in df.columns if c not in ('strategy_args', 'metric_name')][0]
            metrics = {}
            for _, row in df.iterrows():
                mn = str(row['metric_name']).lower()
                if 'mse' in mn or 'mae' in mn:
                    v = row[val_col]
                    metrics[row['metric_name']] = v if pd.notna(v) else float('nan')
            return metrics

        m1 = _load_latest_metrics(mofe_dir)
        m2 = _load_latest_metrics(circ_dir)
        m3 = _load_latest_metrics(cb_dir)
        m4 = _load_latest_metrics(dp_dir)
        all_results[seq_len] = (m1, m2, m3, m4)

        if m1 and m2 and m3 and m4:
            all_keys = sorted(set(m1.keys()) | set(m2.keys()) | set(m3.keys()) | set(m4.keys()))
            print()
            print(f"--- seq_len={seq_len} ---")
            print(f"{'Metric':<25} {'MoFo':>10} {'Circulant':>10} {'CircBias':>10} {'DualPath':>10}")
            print("-" * 70)
            for k in all_keys:
                v1 = m1.get(k, float('nan'))
                v2 = m2.get(k, float('nan'))
                v3 = m3.get(k, float('nan'))
                v4 = m4.get(k, float('nan'))
                v1_str = f"{v1:.6f}" if not np.isnan(v1) else "NaN"
                v2_str = f"{v2:.6f}" if not np.isnan(v2) else "NaN"
                v3_str = f"{v3:.6f}" if not np.isnan(v3) else "NaN"
                v4_str = f"{v4:.6f}" if not np.isnan(v4) else "NaN"
                print(f"{k:<25} {v1_str:>10} {v2_str:>10} {v3_str:>10} {v4_str:>10}")

    print()
    print("=" * 70)
    print("SUMMARY: MoFo vs MoFo_Circulant vs MoFo_CircBias vs MoFo_Circulant_DualPath")
    print("=" * 70)
    for seq_len in SEQ_LEN_LIST:
        m1, m2, m3, m4 = all_results.get(seq_len, (None, None, None, None))
        if m1 and m2 and m3 and m4:
            all_keys = sorted(set(m1.keys()) | set(m2.keys()) | set(m3.keys()) | set(m4.keys()))
            print(f"\n  seq_len = {seq_len}")
            for k in all_keys:
                v1 = m1.get(k, float('nan'))
                v2 = m2.get(k, float('nan'))
                v3 = m3.get(k, float('nan'))
                v4 = m4.get(k, float('nan'))
                v1_str = f"{v1:.6f}" if not np.isnan(v1) else "NaN"
                v2_str = f"{v2:.6f}" if not np.isnan(v2) else "NaN"
                v3_str = f"{v3:.6f}" if not np.isnan(v3) else "NaN"
                v4_str = f"{v4:.6f}" if not np.isnan(v4) else "NaN"
                print(f"    {k:<25} MoFo: {v1_str}  Circulant: {v2_str}  CircBias: {v3_str}  DualPath: {v4_str}")
    print()
    print("=" * 70)
