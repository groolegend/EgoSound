import os
from openai import OpenAI
from config import OPENAI_API_KEY, GOOGLE_API_KEY
import os
import json
import time
from tqdm import tqdm
import ast
import unicodedata
import argparse

NEW_OPENAI_API_KEY = "OPENAI_API_KEY"

# ========== Configure API ==========
# It is recommended to set OPENAI_API_KEY in the environment variables.
# Otherwise, you can manually set it: api_key="sk-xxx"
client = OpenAI(
    api_key=NEW_OPENAI_API_KEY,
    base_url="http://35.164.11.19:3887/v1"
)

def interaction(client, message_text):
    completion = client.chat.completions.create(
        model="gpt-5",
        messages = message_text,
        max_completion_tokens=800
    )
    return completion

def get_score_videollama(sample):
    question = sample['question']
    answer = sample['answer']
    pred = sample['pred']
    try:
        message=[
                {
                    "role": "system",
                    "content": 
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for video-based question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the correctness and accuracy between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                        "- The predicted answer must be factually accurate and align with the video content.\n"
                        "- The predicted answer and the correct answer may differ in language. Make sure to translate them into the same language and compare their semantic consistency.\n"
                        "- The predicted answer can be considered a valid answer to the question\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
                        "Please generate the response in the form of a Python dictionary string with keys 'binary_pred' and 'score', where value of 'binary_pred' is a string of 'correct' or 'incorrect' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'binary_pred': 'correct', 'score': 4}."
                }
            ]
        # ========== Call OpenAI API ==========
        completion = interaction(client, message)
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content
        content = unicodedata.normalize('NFKC', response_message)
        response_dict = ast.literal_eval(content)
        sample.update(response_dict)
        #print(sample)
        return sample

    except Exception as e:
        print(f"Stage 2 FAILED: {e}")
        # If JSON parsing fails, print the raw text for debugging
        if 'response' in locals():
             print(f"Raw response text (non-JSON): {getattr(completion, 'text', 'N/A')}")
        return {"error": str(e)}

def get_score_videosalmonn(sample):
    question = sample['prompt']['value'].split('\n')[1]
    answer = sample['ref']
    pred = sample['pred']
    try:
        message=[
                {
                    "role": "system",
                    "content": 
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for video-based question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the correctness and accuracy between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                        "- The predicted answer must be factually accurate and align with the video content.\n"
                        "- The predicted answer and the correct answer may differ in language. Make sure to translate them into the same language and compare their semantic consistency.\n"
                        "- The predicted answer can be considered a valid answer to the question\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness."
                        "Please generate the response in the form of a Python dictionary string with keys 'binary_pred' and 'score', where value of 'binary_pred' is a string of 'correct' or 'incorrect' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'binary_pred': 'correct', 'score': 4}."
                }
            ]
        # ========== Call OpenAI API ==========
        completion = interaction(client, message)
        # Convert response to a Python dictionary.
        response_message = completion.choices[0].message.content
        sample['question']=question
        content = unicodedata.normalize('NFKC', response_message)
        response_dict = ast.literal_eval(content)
        # response_dict = ast.literal_eval(response_message)
        sample.update(response_dict)
        #print(sample)
        return sample

    except Exception as e:
        print(f"Stage 2 FAILED: {e}")
        # If JSON parsing fails, print the raw text for debugging
        if 'response' in locals():
             print(f"Raw response text (non-JSON): {getattr(completion, 'text', 'N/A')}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate QA predictions using an LLM-based judge."
    )
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
    args = parser.parse_args()

    # Example paths that were previously used:
    # answer_path = "/inspire/hdd/global_user/zhubingwen-253108120125/qwen3_omni/omni_thinking_result_all.json"
    # answer_path = "/inspire/hdd/project/brainsignals/zhubingwen-253108120125/qwen2.5_omni/result_3b/omni_result_test_1200_2120.json"

    answer_path = args.answer_path
    style = args.style
    results = []
    #qa_eval_path = answer_path.replace('result_test','eval_all')
    qa_eval_path = answer_path.replace('result_test','eval_test')
    # pro_ids = json.load(open(qa_eval_path, "r", encoding="utf-8"))
    # pro_ids = [qa["question_id"] for qa in pro_ids]
    #pro_ids = set(pro_ids)
    pro_ids = set()
    #answer_path = "/inspire/hdd/project/brainsignals/zhubingwen-253108120125/eval/gpt_4o_audio_result_all.json"
    with open(answer_path, "r", encoding="utf-8") as f:
        #data = json.load(f)
        samples = json.load(f)
        print(len(samples))
        for i, sample in enumerate(tqdm(samples), start=1):
            if sample['question_id'] in pro_ids:
                continue 
            if style == "video.llama":
                result = get_score_videollama(sample)
            elif style == "videosalmonn":
                result = get_score_videosalmonn(sample)
            else:
                raise ValueError(f"Unsupported style: {style}")
            if isinstance(result, dict) and "error" in result:
                print(f"---{sample['question_id']}--- --- Eval Failded. ---")
                continue
            results.append(result)
            
            if i % 100 == 0 or i == len(samples):
                print(f"Writing batch {i//100} ({len(results)} items)...")
                print(f"The evaluated model result file is: {answer_path}")

                # Check whether this is the first write
                if os.path.exists(qa_eval_path) and os.path.getsize(qa_eval_path) > 0:
                    # Read existing content
                    with open(qa_eval_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                else:
                    existing = []

                # Merge and write back
                existing.extend(results)
                with open(qa_eval_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=4, ensure_ascii=False)

                # Clear cache
                results = []
            
            time.sleep(0.3)
    
    with open(qa_eval_path, "r", encoding="utf-8") as f:
        #data = json.load(f)
        results = json.load(f)       
        print(f"Total evaluated QA pairs: {len(results)}")
        
    #qa_eval_path = answer_path.replace('answer_all','eval_all')
    # qa_eval_path = answer_path.replace('result_all','eval_all')
    # with open(qa_eval_path, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)
        
        # Calculate average score
        score_sum = 0
        count = 0
        corr = 0
        for sample in results:
            try:
                count += 1
                score_match = sample['score']
                score = int(score_match)
                score_sum += score
                if sample['binary_pred'] == "correct":
                    corr += 1
            except:
                print(sample)
                print("json format error")
                continue
        average_score = score_sum / count
        acc = corr / count
        print(f"The evaluated model result file is: {answer_path}")
        print("Average score for correctness:", average_score)
        print("Accuracy for correctness:", acc)
        
        
        