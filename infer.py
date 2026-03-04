import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml


def merge_part_results(answer_dir: str, model_name: str, num_parts: int) -> None:
    """Merge result_0.json ... result_{num_parts-1}.json into result.json (sorted by _idx)."""
    out_dir = os.path.join(answer_dir, model_name)
    merged = []
    missing = []
    for p in range(num_parts):
        part_file = os.path.join(out_dir, f"result_{p}.json")
        if not os.path.isfile(part_file):
            missing.append(part_file)
            continue
        with open(part_file, "r", encoding="utf-8") as f:
            part_data = json.load(f)
        merged.extend(part_data)
    if missing:
        raise FileNotFoundError(
            f"Part result(s) not found: {missing}. "
            "Some GPU processes may have failed (e.g. MASTER_PORT conflict). Re-run with fixed ports or single GPU."
        )
    merged.sort(key=lambda x: x.pop("_idx", 0))
    result_file = os.path.join(out_dir, "result.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)
    print(f"[Merge] Wrote {result_file} ({len(merged)} samples)")


def run_one_part(part_index: int, num_parts: int, script: str, env: dict, gpu_id: int) -> subprocess.CompletedProcess:
    env = env.copy()
    env["PART_INDEX"] = str(part_index)
    env["NUM_PARTS"] = str(num_parts)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Each part uses a different port so parallel processes don't conflict (Address already in use)
    env["MASTER_PORT"] = str(12358 + part_index)
    return subprocess.run([sys.executable, script], env=env, capture_output=False)


def run_model(model_cfg, num_gpus: int = 1):
    script = model_cfg["script"]
    model_name = model_cfg["model_name"]
    ckpt_path = model_cfg.get("ckpt_path")
    qa_path = model_cfg.get("qa_path")
    answer_dir = model_cfg.get("answer_dir")

    modal_type = model_cfg.get("modal_type")

    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    if ckpt_path:
        env["MODEL_CKPT"] = ckpt_path
    if qa_path:
        env["QA_PATH"] = qa_path
    if answer_dir:
        env["ANSWER_DIR"] = answer_dir
    if modal_type:
        env["MODAL_TYPE"] = modal_type

    if num_gpus <= 1:
        env["PART_INDEX"] = "0"
        env["NUM_PARTS"] = "1"
        print(f"\n==== Running model: {model_cfg['model_name']} (1 GPU) ====")
        print("Command:", sys.executable, script)
        try:
            subprocess.run([sys.executable, script], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] model {model_cfg['model_name']} execution failed: {e}")
        return

    print(f"\n==== Running model: {model_cfg['model_name']} ({num_gpus} GPUs in parallel) ====")
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(run_one_part, p, num_gpus, script, env, p): p
            for p in range(num_gpus)
        }
        for fut in as_completed(futures):
            part = futures[fut]
            try:
                fut.result()
                print(f"  Part {part} finished.")
            except Exception as e:
                print(f"[ERROR] Part {part} failed: {e}")
                raise
    merge_part_results(answer_dir or ".", model_name, num_gpus)


def main(config_path: str, only: str | None = None, num_gpus: int = 1):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", [])
    if not models:
        print("[WARN] model not found in config.yaml")
        return

    for m in models:
        if only is not None and m.get("model_name") != only:
            continue
        run_model(m, num_gpus=num_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="config path",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="only run one model",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="split QA data into N parts and run N processes in parallel (one per GPU). Default 1.",
    )
    args = parser.parse_args()

    main(args.config, args.only, num_gpus=args.num_gpus)

