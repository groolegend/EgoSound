import soundfile as sf

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import os
import json
from tqdm import tqdm
import traceback

# allow overriding checkpoint via environment variable
MODEL_PATH = os.environ.get(
    "MODEL_CKPT",
    " "
)
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.disable_talker()
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
VIDEO_DIR = "/inspire/hdd/global_user/zhubingwen-253108120125/EgoBlind/videos"

def eval_omni(part):
    qa_path=os.environ.get(
        "QA_PATH",
        " ",
    )
    qa_data=json.load(open(qa_path, "r"))
    # patch= len(qa_data)//4
    # if part == 3:
    #     qa_data = qa_data[part*patch:]
    # else:
    #     qa_data = qa_data[part*patch:(part+1)*patch]
    
    # 结果目录：ANSWER_DIR，内部按 question_id 单独落 JSON
    json_root = os.environ.get(
        "ANSWER_DIR",
        " ",
    )
    os.makedirs(json_root, exist_ok=True)
    json_dir = json_root
    pro_ids =[id.split('.')[0] for id in os.listdir(json_dir)]
    pro_ids = set(pro_ids)
    print(len(pro_ids))
    qa_all = []
    
    for qa in tqdm(qa_data):
        if qa['question_id'] in pro_ids:
            continue
        qa_all.append(qa)
        
    patch= len(qa_all)//6
    if part == 5:
        qa_all = qa_all[part*patch:]
    else:
        qa_all = qa_all[part*patch:(part+1)*patch]
    results=[]
    
    # pro_ids =[id.split('.')[0] for id in os.listdir(json_dir)]
    # pro_ids = set(pro_ids)

    for i, qa in enumerate(tqdm(qa_all), start=1):
        video_path=qa['video_path']
#        video_path=os.path.join(VIDEO_DIR,os.path.basename(video_path))
        if video_path.split('/')[-1] in ["00944.mp4","01277.mp4","01306.mp4","01106.mp4","01434.mp4","01435.mp4"]:
            continue

        question=qa['question']
        conversation = [
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

        try:
            # Preparation for inference
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(model.device).to(model.dtype)

            # Inference: Generation of the output text and audio
            text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False, thinker_return_dict_in_generate=True)

            text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            #print(text)
            try:
                qa['pred'] = text[0].split('</think>\n\n')[1]
                results.append(qa)
                json_path = f"/inspire/hdd/global_user/zhubingwen-253108120125/qwen3_omni/result_thinking/{qa['question_id']}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(qa, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"[⚠️ subtask error] saving question {qa.get('question_id')} error: {e}")
                traceback.print_exc()
                continue
            #print(text[0].split('</think>\n\n')[1])
        # qa['pred']= text[0].split('assistant\n')[1]
        # results.append(qa)
        # if i % 50 == 0:
        #     print(qa)
        except Exception as e:
            print(f"[❌ batch error] batch {i//1 + 1} (index {i}~{i+0}) error: {e}")
            traceback.print_exc()
            continue
        
    # with open(answer_file, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)

part=4
eval_omni(part)
