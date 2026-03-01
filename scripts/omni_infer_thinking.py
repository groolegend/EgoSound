import soundfile as sf

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import os
import json
from tqdm import tqdm
import traceback

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

    json_root = os.environ.get("ANSWER_DIR", "./results")
    os.makedirs(json_root, exist_ok=True)
    pro_ids = {id.split('.')[0] for id in os.listdir(json_root)}
    qa_all = []
    
    for qa in tqdm(qa_data):
        if qa['question_id'] in pro_ids:
            continue
        qa_all.append(qa)
    
    results=[]
    

    for i, qa in enumerate(tqdm(qa_all), start=1):
        video_path=qa['video_path']
    
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
            try:
                qa['pred'] = text[0].split('</think>\n\n')[1]
                results.append(qa)
                json_path = os.path.join(json_root, f"{qa['question_id']}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(qa, f, indent=4, ensure_ascii=False)
            except Exception as e:
                # log saving error
                traceback.print_exc()
                continue
            
        
        except Exception as e:
            traceback.print_exc()
            continue
        


if __name__ == '__main__':
    eval_omni()
