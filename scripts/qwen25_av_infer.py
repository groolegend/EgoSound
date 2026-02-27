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
    qa_path=os.environ.get(
        "QA_PATH",
        " "
    )
    qa_data=json.load(open(qa_path, "r"))

    answer_root = os.environ.get(
        "ANSWER_DIR",
        " "
    )
    model_name = os.environ.get("MODEL_NAME", "qwen25_omni")
    answer_dir = os.path.join(answer_root, model_name)
    os.makedirs(answer_dir, exist_ok=True)
    answer_file = os.path.join(answer_dir, "result.json")
    results=[]

    for i, qa in enumerate(tqdm(qa_data), start=1):
        video_path=qa['video_path']
        question=qa['question']
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
        # print(text[0].split('assistant\n')[1])
        qa['pred']= text[0].split('assistant\n')[1]
        results.append(qa)
        if i % 50 == 0:
            print(qa)
    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
eval_omni()