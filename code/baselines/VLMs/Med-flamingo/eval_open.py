from huggingface_hub import hf_hub_download
import torch
import os
import argparse
import os
import json
from tqdm import tqdm
import math
import shortuuid

from open_flamingo import create_model_and_transforms
from accelerate import Accelerator
from einops import repeat
from PIL import Image
import sys

from src.utils import FlamingoProcessor
from demo_utils import image_paths, clean_generation

import random
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    accelerator = Accelerator()  # when using cpu: cpu=True

    device = accelerator.device

    print('Loading model..')

    # >>> add your local path to Llama-7B (v1) model here:
    llama_path = '/path/to/llama_v1'
    if not os.path.exists(llama_path):
        raise ValueError('Llama model not yet set up, please check README for instructions!')

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )
    # load med-flamingo checkpoint:
    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
    print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    processor = FlamingoProcessor(tokenizer, image_processor)

    # go into eval model and prepare:
    model = accelerator.prepare(model)
    is_main_process = accelerator.is_main_process
    model.eval()

    file_path = os.path.expanduser(args.question_file)
    if file_path.endswith('.jsonl'):
        # Handle JSONL files
        with open(file_path, 'r') as file:
            questions = [json.loads(line) for line in file]
    else:
        # Handle JSON files
        with open(file_path, 'r') as file:
            questions = json.load(file)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    need_questions = []
    for i in questions:
        if i["question_type"] == "type4_Knowledge":
            need_questions.append(i)
    questions = need_questions
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "a")
    for line in tqdm(questions):

        if line["question_type"] == "type_2" or line["question_type"] == "type4_Knowledge":
            pass
        else:
            continue

        idx = line["qid"]
        image_file = line["img_name"]

        question = line["question"]
        stru_ans = ""
        if line.__contains__("structured_answer"):
            stru_ans = line["structured_answer"]

        demo_images = [Image.open(os.path.join(args.image_folder, image_file))]
        max_new_tokens = 24
        if line['question_type'] == "type_2" or line['question_type']!="type_1":
            max_new_tokens = 128

        # example few-shot prompt:
        prompt = f"You are a helpful medical assistant. You are being provided with images, a question about the image and an answer. <image>Question: {question} Answer:"
        pixels = processor.preprocess_images(demo_images)
        pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
        tokenized_data = processor.encode_text(prompt)
        generated_text = model.generate(
            vision_x=pixels.to(device),
            lang_x=tokenized_data["input_ids"].to(device),
            attention_mask=tokenized_data["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
        )
        response = processor.tokenizer.decode(generated_text[0])
        response = clean_generation(response)
        response = response.split("Answer:")[-1]
        # _ind = response.index(prompt)
        # response = response[_ind:]

        ans_id = shortuuid.uuid()
        gt_ans = line["answer"]
        ans_file.write(json.dumps({"question_id": idx,
                                   # "final_prompt":prompt,
                                   # "input_ids":str(inputs.input_ids),
                                   "prompt": question,
                                   "model_answer": response,
                                   "ground_truth": gt_ans,
                                   "question_type": line['question_type'],
                                   "structured_answer": stru_ans,
                                   "image_id": line["img_name"],
                                   "answer_id": ans_id,
                                   "model_id": "med-flamingo",
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