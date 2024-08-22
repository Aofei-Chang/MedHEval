import tqdm.auto as tqdm
import torch.nn.functional as F
from typing import Optional, Dict, Sequence
from typing import List, Optional, Tuple, Union
import transformers
from dataclasses import dataclass, field
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from torchvision import transforms
from PIL import Image

import argparse
import os
import json
from tqdm import tqdm
import math
import shortuuid


import random
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to
    '''
    if isinstance(tokenizer_path, str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""

            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image" + str(i * image_num + j) + ">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(
                special_token
            )
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens


def combine_and_preprocess(question, image_list, image_padding_tokens):
    transform = transforms.Compose([
        transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    images = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']

        image = Image.open(img_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1)  # c,w,h,d

        ## pre-process the img first
        target_H = 512
        target_W = 512
        target_D = 4
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images.
        images.append(torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D)))

        ## add img placeholder to text
        new_qestions[position] = "<image>" + image_padding_tokens[padding_index] + "</image>" + new_qestions[position]
        padding_index += 1

    vision_x = torch.cat(images, dim=1).unsqueeze(0)  # cat tensors and expand the batch_size dim
    text = ''.join(new_qestions)
    return text, vision_x,


def eval_model(args):
    print("Setup tokenizer")
    text_tokenizer, image_padding_tokens = get_tokenizer('./Language_files')
    print("Finish loading tokenizer")
    print("Setup Model")
    model = MultiLLaMAForCausalLM(
        lang_model_path='./Language_files',  ### Build up model based on LLaMa-13B config
    )
    ckpt = torch.load('/path/to/RadFM/pytorch_model.bin',
                      map_location='cpu')  # Please dowloud our checkpoint from huggingface and Decompress the original zip file first
    model.load_state_dict(ckpt)
    print("Finish loading model")
    model = model.half()
    model = model.to('cuda')
    model.eval()

    #load test data
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

    # ans_file = open(answers_file, "a")

    for line in tqdm(questions):

        if line["question_type"] == "type_2" or line["question_type"] == "type4_Knowledge":
            pass
        else:
            continue
        idx = line["qid"]
        image_file = line["img_name"]

        question = line["question"] + "Your answer in details:"
        stru_ans = ""
        if line.__contains__("structured_answer"):
            stru_ans = line["structured_answer"]

        image = [
            {
                'img_path': os.path.join(args.image_folder, image_file),
                'position': 0,  # indicate where to put the images in the text string, range from [0,len(question)-1]
            },  # can add abitrary number of imgs
        ]

        text, vision_x = combine_and_preprocess(question, image, image_padding_tokens)
        with torch.no_grad():
            lang_x = text_tokenizer(
                text, max_length=1024, truncation=True, return_tensors="pt"
            )['input_ids'].to('cuda')

            vision_x = vision_x.to('cuda')
            generation = model.generate(lang_x, vision_x)
            generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True)

        ans_id = shortuuid.uuid()
        gt_ans = line["answer"]
        ans_file.write(json.dumps({"question_id": idx,
                                   # "final_prompt":prompt,
                                   # "input_ids":str(inputs.input_ids),
                                   "prompt": question,
                                   "model_answer": generated_texts[0],
                                   "ground_truth": gt_ans,
                                   "question_type": line['question_type'],
                                   "structured_answer": stru_ans,
                                   "image_id": line["img_name"],
                                   "answer_id": ans_id,
                                   "model_id": "RadFM",
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

