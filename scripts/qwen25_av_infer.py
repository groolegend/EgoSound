import soundfile as sf
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import os
import json
from tqdm import tqdm

# allow overriding 7B checkpoint via environment variable
MODEL_PATH_7B=os.environ.get(
    "MODEL_CKPT",
    " "
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH_7B,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH_7B)

def eval_omni():
    qa_path = os.environ.get("QA_PATH", " ")
    qa_data = json.load(open(qa_path, "r"))
    qa_dir = os.path.dirname(os.path.abspath(qa_path))

    part_index = int(os.environ.get("PART_INDEX", "0"))
    num_parts = int(os.environ.get("NUM_PARTS", "1"))
    qa_data = [qa for i, qa in enumerate(qa_data) if i % num_parts == part_index]

    answer_root = os.environ.get("ANSWER_DIR", " ")
    model_name = os.environ.get("MODEL_NAME", "qwen25_omni")
    answer_dir = os.path.join(answer_root, model_name)
    os.makedirs(answer_dir, exist_ok=True)
    if num_parts > 1:
        answer_file = os.path.join(answer_dir, f"result_{part_index}.json")
    else:
        answer_file = os.path.join(answer_dir, "result.json")
    results = []

    for i, qa in enumerate(tqdm(qa_data, desc=f"Part {part_index}/{num_parts}")):
        video_path = qa["video_path"]
        if not os.path.isabs(video_path):
            video_path = os.path.join(qa_dir, video_path)
        question = qa["question"]
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question}
                ],
            },
        ]
        # set use audio in video
        USE_AUDIO_IN_VIDEO = True

        # Preparation for inference
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)

        # Inference: Generation of the output text and audio
        text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,return_audio=False)

        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        qa["pred"] = text[0].split("assistant\n")[1]
        if num_parts > 1:
            qa["_idx"] = part_index + i * num_parts
        results.append(qa)

    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    eval_omni()
