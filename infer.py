import argparse
import os
import subprocess
import sys

import yaml


def run_model(model_cfg):

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


    cmd = [sys.executable, script]

    print(f"\n==== Running model: {model_cfg['model_name']} ====")
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] model {model_cfg['model_name']} execution failed: {e}")


def main(config_path: str, only: str | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", [])
    if not models:
        print("[WARN] model not found in config.yaml")
        return

    for m in models:
        if only is not None and m.get("model_name") != only:
            continue
        run_model(m)


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
    args = parser.parse_args()

    main(args.config, args.only)

