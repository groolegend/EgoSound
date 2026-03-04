import os
import sys
import json
import time
import ast
import unicodedata
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# =========================
# Configuration
# =========================
# IMPORTANT: Do NOT hardcode keys in code. Use environment variables instead.
# export OPENAI_API_KEY="sk-xxxx"


BASE_URL = os.getenv("OPENAI_BASE_URL", "https://yibuapi.com/v1")
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. Please run:\n"
        "  export OPENAI_API_KEY='sk-xxxx'\n"
        "Optionally set OPENAI_BASE_URL, e.g.:\n"
        "  export OPENAI_BASE_URL='https://yibuapi.com/v1'"
    )

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# =========================
# OpenAI call helpers
# =========================
def interaction(_client: OpenAI, message_text):
    """Perform a single API call."""
    return _client.chat.completions.create(
        model="gpt-5",
        messages=message_text,
        max_completion_tokens=800,
    )


def ask_with_retries(_client: OpenAI, message, retries=3, backoff=1.0):
    """
    Call the LLM and retry on transient failures or empty responses.
    Returns the response content string.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            completion = interaction(_client, message)
            response_message = completion.choices[0].message.content
            if response_message is None or response_message.strip() == "":
                raise ValueError("empty response")
            return response_message
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
            else:
                raise last_err


# =========================
# Parsing helpers
# =========================
def parse_python_dict_string(text: str) -> dict:
    """
    Parse the judge output which should be a Python dict string, e.g.
    "{'binary_pred': 'correct', 'score': 4}"
    """
    if text is None or text.strip() == "":
        raise ValueError("empty_response")

    content = unicodedata.normalize("NFKC", text).strip()
    # If the model ever wraps in code fences, strip them defensively.
    if content.startswith("```"):
        content = content.strip("`").strip()
        # handle possible "json"/"python" tags on first line
        lines = content.splitlines()
        if lines and lines[0].lower() in ("json", "python"):
            content = "\n".join(lines[1:]).strip()

    return ast.literal_eval(content)


# =========================
# Judge wrappers
# =========================
def build_judge_messages(question: str, answer: str, pred: str):
    sys_msg = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs "
        "for video-based question-answer pairs. Your task is to compare the predicted answer with the "
        "correct answer and determine if the predicted answer is correct or not.\n"
        "------\n"
        "## INSTRUCTIONS:\n"
        "- Focus on correctness and factual accuracy.\n"
        "- The predicted answer must align with the video content implied by the question/answer.\n"
        "- The predicted answer and the correct answer may differ in language; translate mentally and compare semantics.\n"
        "- Synonyms/paraphrases are valid.\n"
        "- Output ONLY a Python dictionary string with keys 'binary_pred' and 'score'.\n"
        "- 'binary_pred' must be 'correct' or 'incorrect'.\n"
        "- 'score' must be an integer from 0 (fully wrong) to 5 (fully correct).\n"
        "- No extra text, no markdown.\n"
        "Example: {'binary_pred': 'correct', 'score': 4}"
    )

    user_msg = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Return ONLY a Python dictionary string like: {'binary_pred': 'correct', 'score': 4}."
    )

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


def get_score_videollama(sample: dict) -> dict:
    question = sample["question"]
    answer = sample["answer"]
    pred = sample["pred"]

    try:
        messages = build_judge_messages(question, answer, pred)
        response_message = ask_with_retries(client, messages, retries=3, backoff=1.0)

        # Uncomment for debugging:
        # print("RAW >>>", repr(response_message))

        response_dict = parse_python_dict_string(response_message)
        sample.update(response_dict)
        return sample
    except Exception as e:
        print(f"Stage 2 FAILED (videollama): {e}")
        return {"error": str(e), "question_id": sample.get("question_id")}


def get_score_videosalmonn(sample: dict) -> dict:
    # videosalmonn style input:
    # sample = { "prompt": {"value": "header\nquestion"}, "ref": "...", "pred": "...", ... }
    try:
        question = sample["prompt"]["value"].split("\n")[1]
    except Exception:
        # fallback: use the whole prompt if unexpected format
        question = sample.get("prompt", {}).get("value", "")

    answer = sample["ref"]
    pred = sample["pred"]

    try:
        messages = build_judge_messages(question, answer, pred)
        response_message = ask_with_retries(client, messages, retries=3, backoff=1.0)

        response_dict = parse_python_dict_string(response_message)
        sample["question"] = question
        sample.update(response_dict)
        return sample
    except Exception as e:
        print(f"Stage 2 FAILED (videosalmonn): {e}")
        return {"error": str(e), "question_id": sample.get("question_id")}


# =========================
# Multi-worker merge & launcher (same pattern as infer.py)
# =========================
def merge_eval_part_results(qa_eval_path: str, num_parts: int) -> None:
    """Merge eval_0.json ... eval_{num_parts-1}.json into eval.json (sorted by _idx)."""
    out_dir = os.path.dirname(qa_eval_path) or "."
    base_name = os.path.basename(qa_eval_path)
    name_stem, ext = os.path.splitext(base_name)
    merged = []
    missing = []
    for p in range(num_parts):
        part_file = os.path.join(out_dir, f"{name_stem}_{p}{ext}")
        if not os.path.isfile(part_file):
            missing.append(part_file)
            continue
        with open(part_file, "r", encoding="utf-8") as f:
            part_data = json.load(f)
        merged.extend(part_data)
    if missing:
        raise FileNotFoundError(
            f"Part result(s) not found: {missing}. "
            "Some worker processes may have failed. Re-run or use --num_gpus 1."
        )
    merged.sort(key=lambda x: x.pop("_idx", 0))
    with open(qa_eval_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[Merge] Wrote {qa_eval_path} ({len(merged)} samples)")


def run_one_eval_part(
    part_index: int,
    num_parts: int,
    answer_path: str,
    style: str,
    save_every: int,
    sleep: float,
) -> subprocess.CompletedProcess:
    """Run this script as one worker with PART_INDEX and NUM_PARTS set."""
    env = os.environ.copy()
    env["PART_INDEX"] = str(part_index)
    env["NUM_PARTS"] = str(num_parts)
    return subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--answer_path", answer_path,
            "--style", style,
            "--save_every", str(save_every),
            "--sleep", str(sleep),
        ],
        env=env,
        capture_output=False,
    )


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Evaluate QA predictions using an LLM-based judge.")
    parser.add_argument(
        "--answer_path",
        type=str,
        required=True,
        help="Path to the JSON file containing model predictions.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="videollama",
        choices=["videollama", "videosalmonn"],
        help="Input result style: 'videollama' or 'videosalmonn'.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save intermediate results every N samples.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Sleep (seconds) between requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Split data into N parts and run N processes in parallel (like infer.py). Default 1.",
    )
    args = parser.parse_args()

    answer_path = args.answer_path
    style = args.style
    save_every = args.save_every
    sleep_s = args.sleep
    num_gpus = args.num_gpus

    part_index = int(os.environ.get("PART_INDEX", "0"))
    num_parts = int(os.environ.get("NUM_PARTS", "1"))

    # Launcher: spawn N workers then merge (same pattern as infer.py)
    if num_gpus > 1:
        print(f"\n==== Running QA eval with {num_gpus} workers in parallel ====")
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(
                    run_one_eval_part,
                    p,
                    num_gpus,
                    answer_path,
                    style,
                    save_every,
                    sleep_s,
                ): p
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
        qa_eval_path = answer_path.replace("result", "eval")
        merge_eval_part_results(qa_eval_path, num_gpus)
        # Print stats from merged file
        with open(qa_eval_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        _print_stats(results, answer_path)
        return

    qa_eval_path = answer_path.replace("result", "eval")
    if num_parts > 1:
        base_name = os.path.basename(qa_eval_path)
        name_stem, ext = os.path.splitext(base_name)
        out_dir = os.path.dirname(qa_eval_path) or "."
        qa_eval_path = os.path.join(out_dir, f"{name_stem}_{part_index}{ext}")
    out_dir = os.path.dirname(qa_eval_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load input samples
    with open(answer_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    # Partition: this worker only processes indices where i % num_parts == part_index
    samples = [s for i, s in enumerate(all_samples) if i % num_parts == part_index]

    print(f"Loaded {len(samples)} samples (part {part_index}/{num_parts}) from: {answer_path}")
    results = []
    processed_ids = set()

    # Resume if eval file exists and is non-empty
    if os.path.exists(qa_eval_path) and os.path.getsize(qa_eval_path) > 0:
        try:
            with open(qa_eval_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                results = existing
                for r in results:
                    qid = r.get("question_id")
                    if qid is not None:
                        processed_ids.add(qid)
            print(f"Resuming: loaded {len(results)} existing evaluated samples from: {qa_eval_path}")
        except Exception as e:
            print(f"Warning: failed to read existing eval file {qa_eval_path}: {e}")

    # Evaluate loop
    for i, sample in enumerate(tqdm(samples, total=len(samples), desc=f"Part {part_index}/{num_parts}"), start=1):
        qid = sample.get("question_id")
        if qid is not None and qid in processed_ids:
            continue

        if style == "videollama":
            result = get_score_videollama(sample)
        else:
            result = get_score_videosalmonn(sample)

        if isinstance(result, dict) and "error" in result:
            print(f"---{qid if qid else i}--- Eval Failed. Skipping.")
            continue

        if num_parts > 1:
            result["_idx"] = part_index + (i - 1) * num_parts  # original index for merging
        results.append(result)
        if qid is not None:
            processed_ids.add(qid)

        # Save periodically
        if (len(results) % save_every) == 0:
            with open(qa_eval_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(sleep_s)

    # Final save
    with open(qa_eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluated results to: {qa_eval_path}")
    print(f"Total evaluated QA pairs: {len(results)}")

    if num_parts <= 1:
        _print_stats(results, answer_path)
    # When num_parts > 1 we're a worker; launcher will merge and print stats


def _print_stats(results: list, answer_path: str) -> None:
    """Print average score and accuracy from evaluated results."""
    score_sum = 0
    count = 0
    corr = 0
    for r in results:
        try:
            score = int(r["score"])
            score_sum += score
            count += 1
            if r.get("binary_pred") == "correct":
                corr += 1
        except Exception:
            continue
    if count > 0:
        average_score = score_sum / count
        acc = corr / count
    else:
        average_score = 0.0
        acc = 0.0
    print(f"The evaluated model result file is: {answer_path}")
    print("Average score for correctness:", average_score)
    print("Accuracy for correctness:", acc)


if __name__ == "__main__":
    main()
