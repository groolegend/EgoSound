import sys
sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import json
from tqdm import tqdm
import os


def inference():
    model_path = os.environ.get(
        "MODEL_CKPT",
        " "
    )
    modal_type = os.environ.get("MODAL_TYPE", "a")  # a / v / av

    model, processor, tokenizer = model_init(model_path)

    if modal_type == "a":
        model.model.vision_tower = None
    elif modal_type == "v":
        model.model.audio_tower = None
    elif modal_type == "av":
        pass
    else:
        raise NotImplementedError(f"Unsupported MODAL_TYPE: {modal_type}")

    qa_path = os.environ.get("QA_PATH", "")
    qa_data = json.load(open(qa_path, "r"))

    answer_root = os.environ.get("ANSWER_DIR", "")
    model_name = os.environ.get("MODEL_NAME", "videollama2_av")
    answer_dir = os.path.join(answer_root, model_name)
    os.makedirs(answer_dir, exist_ok=True)
    answer_file = os.path.join(answer_dir, "result.json")
    results = []

    for i, qa in enumerate(tqdm(qa_data), start=1):
        audio_video_path = qa["video_path"]
        audio_video_path = audio_video_path.replace("videos", "audios").replace(
            ".mp4", ".wav"
        )
        preprocess = processor["audio" if modal_type == "a" else "video"]
        if modal_type == "a":
            audio_video_tensor = preprocess(audio_video_path)
        else:
            audio_video_tensor = preprocess(
                audio_video_path, va=True if modal_type == "av" else False
            )
        question = qa["question"]

        output = mm_infer(
            audio_video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            modal="audio" if modal_type == "a" else "video",
            do_sample=False,
        )
        qa["pred"] = output
        results.append(qa)

    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    inference()
