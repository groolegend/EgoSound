import math
import shutil
import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import librosa
import torch
import json
from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm

MINICPM_PATH = os.environ.get("MODEL_CKPT", " ").strip()


def _sync_trust_remote_code_to_cache(model_path: str) -> None:
    """Copy local .py files into HF transformers_modules cache so trust_remote_code finds them.
    Fixes FileNotFoundError for image_processing_minicpmv.py etc. when cache is incomplete.
    """
    if not model_path or not os.path.isdir(model_path):
        return
    # Cache key is usually the last component of the path or from config _name_or_path
    try:
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        name = cfg.get("_name_or_path", "").split("/")[-1] or os.path.basename(os.path.normpath(model_path))
    except Exception:
        name = os.path.basename(os.path.normpath(model_path))
    cache_dir = os.path.join(
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules"),
        name,
    )
    os.makedirs(cache_dir, exist_ok=True)
    for f in os.listdir(model_path):
        if f.endswith(".py"):
            src = os.path.join(model_path, f)
            dst = os.path.join(cache_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)


def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        video.audio.write_audiofile(temp_audio_file.name, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file.name, sr=16000, mono=True)
    num_units = math.ceil(video.duration)
    
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(0,num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr*i:sr*(i+1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])
    
    return contents

def get_video_chunk_new(video_path, flatten=True):
    video = VideoFileClip(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        video.audio.write_audiofile(temp_audio_file.name, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file.name, sr=16000, mono=True)
    sample_interval = 2 if video.duration > 240 else 1
    num_units = math.ceil(video.duration / sample_interval)
    
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(0, num_units):
        t = (i + 1) * sample_interval  
        t = min(t, video.duration - 0.001)  

        frame = video.get_frame(t)
        image = Image.fromarray((frame).astype(np.uint8))

        audio = audio_np[int(sr * (t - sample_interval)):int(sr * t)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])
    return contents


_sync_trust_remote_code_to_cache(MINICPM_PATH)

model = AutoModel.from_pretrained(MINICPM_PATH, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MINICPM_PATH, trust_remote_code=True)

model.init_tts()


def extract_video_id(video_path):
    if not video_path:
        return "unknown"
    base = os.path.basename(video_path)
    video_id = os.path.splitext(base)[0]
    return video_id

def eval_omni():
    qa_path = os.environ.get(
        "QA_PATH",
        " "
    )
    qa_data = json.load(open(qa_path, "r"))
    qa_dir = os.path.dirname(os.path.abspath(qa_path))

    part_index = int(os.environ.get("PART_INDEX", "0"))
    num_parts = int(os.environ.get("NUM_PARTS", "1"))
    qa_data = [qa for i, qa in enumerate(qa_data) if i % num_parts == part_index]

    answer_root = os.environ.get(
        "ANSWER_DIR",
        " "
    )
    model_name = os.environ.get("MODEL_NAME", "minicpm_av")
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
        sys_msg = model.get_sys_prompt(mode='omni', language='en')

        contents = get_video_chunk_new(video_path)
        if not contents:
            continue
        question = qa['question']
        msg = {"role":"user", "content": contents+[question]}
        msgs = [sys_msg, msg]

        generate_audio = False
        output_audio_path = 'output.wav'

        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.5,
            max_new_tokens=4096,
            omni_input=True, # please set omni_input=True when omni inference
            use_tts_template=True,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path,
            max_slice_nums=1,
            use_image_id=False,
            return_dict=True
        )
        qa["pred"] = res.text
        if num_parts > 1:
            qa["_idx"] = part_index + i * num_parts  # original index for merging
        results.append(qa)

    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def eval_audio():
    qa_path = os.environ.get(
        "QA_PATH",
        " "
    )
    qa_data = json.load(open(qa_path, "r"))

    part_index = int(os.environ.get("PART_INDEX", "0"))
    num_parts = int(os.environ.get("NUM_PARTS", "1"))
    qa_data = [qa for i, qa in enumerate(qa_data) if i % num_parts == part_index]

    answer_root = os.environ.get(
        "ANSWER_DIR",
        " "
    )
    model_name = os.environ.get("MODEL_NAME", "minicpm_av_audio")
    answer_dir = os.path.join(answer_root, model_name)
    os.makedirs(answer_dir, exist_ok=True)
    if num_parts > 1:
        answer_file = os.path.join(answer_dir, f"result_{part_index}.json")
    else:
        answer_file = os.path.join(answer_dir, "result.json")
    results = []

    for i, qa in enumerate(tqdm(qa_data, desc=f"Part {part_index}/{num_parts}")):
        video_path=qa['video_path']
        audio_path=video_path.replace('videos','audios').replace('.mp4','.wav')
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        question = qa['question']
        prompt="Please listen to the audio snippet carefully." + "\n"+ question
        msgs = [{'role': 'user', 'content': [prompt, audio_input]}]
        generate_audio = False
        output_audio_path = 'output.wav'

        res = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.5,
            max_new_tokens=4096,
            omni_input=True, # please set omni_input=True when omni inference
            use_tts_template=True,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path
        )
        qa["pred"] = res
        if num_parts > 1:
            qa["_idx"] = part_index + i * num_parts  # original index for merging
        results.append(qa)

    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # eval_audio()
    eval_omni()
