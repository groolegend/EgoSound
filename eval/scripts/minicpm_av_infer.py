import math
import numpy as np
from PIL import Image
from moviepy import *
import tempfile
import librosa
import soundfile as sf
import torch
import json
from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm

# allow overriding model checkpoint via environment variable
MINICPM_PATH = os.environ.get(
    "MODEL_CKPT",
    " "
)

def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration) 
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
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
    print('video_duration:', video.duration) 
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
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

model = AutoModel.from_pretrained(MINICPM_PATH, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(MINICPM_PATH, trust_remote_code=True)

model.init_tts()

# If you are using an older version of PyTorch, you might encounter this issue "weight_norm_fwd_first_dim_kernel" not implemented for 'BFloat16', Please convert the TTS to float32 type.
# model.tts.float()

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
    qa_data=json.load(open(qa_path, "r"))

    answer_root = os.environ.get(
        "ANSWER_DIR",
        " "
    )
    model_name = os.environ.get("MODEL_NAME", "minicpm_av")
    answer_dir = os.path.join(answer_root, model_name)
    os.makedirs(answer_dir, exist_ok=True)
    answer_file = os.path.join(answer_dir, "result.json")
    results=[]
    #qa_part_1 = qa_data[:len(qa_data)//2]
    qa_part_2 = qa_data[len(qa_data)//2:]
    for i, qa in enumerate(tqdm(qa_part_2), start=1):
        video_path=qa['video_path']
        video_id = int(extract_video_id(video_path))
        if video_id<1307:
            continue
        audio_path=video_path.replace('videos','audios').replace('.mp4','.wav')
        sys_msg = model.get_sys_prompt(mode='omni', language='en')
        # if use voice clone prompt, please set ref_audio
        # ref_audio_path = '/path/to/ref_audio'
        # ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        # sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode='omni', language='en')

        contents = get_video_chunk_new(video_path)
        if not contents:
            continue
        question = qa['question']
        msg = {"role":"user", "content": contents+[question]}
        msgs = [sys_msg, msg]

        # please set generate_audio=True and output_audio_path to save the tts result
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
        #print(res.text)
        qa['pred']= res.text
        results.append(qa)
        if i % 10 == 0 or i == len(qa_part_2):
            print(f"Writing batch {i//10} ({len(results)} items)...")

            if os.path.exists(answer_file) and os.path.getsize(answer_file) > 0:
                with open(answer_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.extend(results)
            with open(answer_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4, ensure_ascii=False)

            results = []

     #with open(answer_file, "w", encoding="utf-8") as f:
         #json.dump(results, f, indent=4, ensure_ascii=False)
        
def eval_audio():
    qa_path = os.environ.get(
        "QA_PATH",
        " "
    )
    qa_data=json.load(open(qa_path, "r"))

    answer_root = os.environ.get(
        "ANSWER_DIR",
        " "
    )
    model_name = os.environ.get("MODEL_NAME", "minicpm_av_audio")
    answer_dir = os.path.join(answer_root, model_name)
    os.makedirs(answer_dir, exist_ok=True)
    answer_file = os.path.join(answer_dir, "result.json")
    results=[]

    for i, qa in enumerate(tqdm(qa_data), start=1):
        video_path=qa['video_path']
        audio_path=video_path.replace('videos','audios').replace('.mp4','.wav')
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True) # load the audio to be mimicked
        #sys_msg = model.get_sys_prompt(mode='omni', language='en')
        # if use voice clone prompt, please set ref_audio
        # ref_audio_path = '/path/to/ref_audio'
        # ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        # sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode='omni', language='en')
        question = qa['question']
        prompt="Please listen to the audio snippet carefully." + "\n"+ question
        msgs = [{'role': 'user', 'content': [prompt, audio_input]}]
        # please set generate_audio=True and output_audio_path to save the tts result
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
        #print(res.text)
        qa['pred']= res
        results.append(qa)
        if i % 50 == 0:
            print(qa)

    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
#eval_audio()
eval_omni()