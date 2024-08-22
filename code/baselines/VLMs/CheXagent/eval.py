import io
import argparse
import os
import json
from tqdm import tqdm

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import math
import shortuuid


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    device = "cuda"
    dtype = torch.float16

    # step 2: Load Processor and Model
    processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
    model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True)
    model.to(device)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    file_path = os.path.expanduser(args.question_file)
    if file_path.endswith('.jsonl'):
        # Handle JSONL files
        with open(file_path, 'r') as file:
            questions = [json.loads(line) for line in file]
    else:
        # Handle JSON files
        with open(file_path, 'r') as file:
            questions = json.load(file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)


    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")
    for line in tqdm(questions):
        idx = line["qid"]
        image_file = line["img_name"]

        question = line["question"]
        if line['question_type'] == "multi-choice":
            if len(line['choices']) >= 4:
                question += f" Please select from the following choices: {line['choices']}"
            else:
                continue
        else:
            question += " Please answer Yes or No."

        # step 3: Fetch the images
        images = [Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")]

        # step 4: Generate the Findings section
        prompt = question
        inputs = processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(
            device=device, dtype=dtype)
        output = model.generate(**inputs, generation_config=generation_config)[0]
        response = processor.tokenizer.decode(output, skip_special_tokens=True)

        ans_id = shortuuid.uuid()
        gt_ans = line["answer"]
        ans_file.write(json.dumps({"question_id": idx,
                                   # "final_prompt":prompt,
                                   # "input_ids":str(inputs.input_ids),
                                   "prompt": question,
                                   "model_answer": response,
                                   "ground_truth": gt_ans,
                                   "image_id": line["img_name"],
                                   "answer_id": ans_id,
                                   "model_id": "CheXagent-8b",
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)