from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import os
import sys
import json
import traceback
from tqdm import tqdm
import torch

MODEL_PATH = os.environ.get("MODEL_CKPT", " ")

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.disable_talker()
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

def eval_omni():
    qa_path = os.environ.get("QA_PATH", " ")
    qa_data = json.load(open(qa_path, "r"))
    qa_dir = os.path.dirname(os.path.abspath(qa_path))

    part_index = int(os.environ.get("PART_INDEX", "0"))
    num_parts = int(os.environ.get("NUM_PARTS", "1"))
    qa_data = [qa for i, qa in enumerate(qa_data) if i % num_parts == part_index]

    answer_root = os.environ.get("ANSWER_DIR", "./results")
    model_name = os.environ.get("MODEL_NAME", "qwen3_omni_thinking")
    answer_dir = os.path.join(answer_root, model_name)
    os.makedirs(answer_dir, exist_ok=True)
    if num_parts > 1:
        answer_file = os.path.join(answer_dir, f"result_{part_index}.json")
    else:
        answer_file = os.path.join(answer_dir, "result.json")
    results = []

    USE_AUDIO_IN_VIDEO = False

    n_total = len(qa_data)
    for i, qa in enumerate(tqdm(qa_data, desc=f"Part {part_index}/{num_parts}")):
        video_path = qa["video_path"]
        if not os.path.isabs(video_path):
            video_path = os.path.join(qa_dir, video_path)
        print(f"[{i+1}/{n_total}] Loading: {os.path.basename(video_path)}", flush=True)
        question = qa["question"]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question},
                ],
            },
        ]

        try:
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(model.device).to(model.dtype)
            print(f"[{i+1}/{n_total}] Running generate (may take minutes per sample)...", flush=True)
            gen_out = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False, thinker_return_dict_in_generate=True,max_new_tokens=256)
            # 返回可能是 (thinker_result,) 或 (thinker_result, audio)；thinker_result 可能是带 .sequences 的对象、张量或已解码 str
            if isinstance(gen_out, (tuple, list)) and len(gen_out) >= 2:
                text_ids, _ = gen_out[0], gen_out[1]
            else:
                text_ids = gen_out[0] if isinstance(gen_out, (tuple, list)) else gen_out
            if isinstance(text_ids, str):
                raw_text = text_ids
            elif hasattr(text_ids, "sequences"):
                text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                raw_text = text[0]
            elif isinstance(text_ids, torch.Tensor):
                text = processor.batch_decode(text_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                raw_text = text[0]
            else:
                raise AttributeError(f"generate returned unexpected type for text_ids: {type(text_ids)}")
            qa["pred"] = raw_text.split("</think>\n\n")[1]
            if num_parts > 1:
                qa["_idx"] = part_index + i * num_parts
            results.append(qa)
            print(f"[{i+1}/{n_total}] Done.", flush=True)
        except torch.cuda.OutOfMemoryError:
            traceback.print_exc()
            raise
        except Exception as e:
            traceback.print_exc()
            continue

    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    eval_omni()
