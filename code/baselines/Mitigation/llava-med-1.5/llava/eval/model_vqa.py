import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import sys
sys.path.append("/home/avc6555/research/MedH/Mitigation/LVLMs/llava-med-1.5")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from llava.model.moe_llava import LoRA_MOE_FFN, LoRA_MOE_QK, LoRA_MOE_QK_old

from PIL import Image
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    # model_base = os.path.expanduser(args.model_base)
    # model_name = get_model_name_from_path(model_path)
    model_name = "mistral_llava_med_1.5"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, peft_path=args.peft_path)

    if args.baseline == "VCD":
        from VCD_files.vcd_sample import evolve_vcd_sampling
        evolve_vcd_sampling()
    elif args.baseline == "avisc" or args.baseline == "m3id" or args.baseline == "damro":
        print(f"use and set {args.baseline} sampling")
        from avisc_utils.vcd_add_noise import add_diffusion_noise
        from avisc_utils.avisc_sample import evolve_avisc_sampling
        evolve_avisc_sampling()
    

    #LOAD LORA
    peft_path = args.peft_path
    if peft_path is not None and len(str(peft_path))>4:
        from peft import PeftModel
        print(f"Loading LoRA weights from {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)
        print(f"Merging weights")
        model = model.merge_and_unload()
        
        moe_path = os.path.join(peft_path, "non_lora_trainables.bin")
        moe_state_dict = torch.load(moe_path, map_location='cuda')
        if len(moe_state_dict.keys()) > 32:
            print("load MoE parameters!")
            top_layers = []
            num_layers = len(model.base_model.layers)
            for i in range(num_layers):
                if i not in top_layers:
                    original_q = model.base_model.layers[i].self_attn.q_proj
                    model.base_model.layers[i].self_attn.q_proj = \
                        LoRA_MOE_QK(args=None,
                            lora_rank=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            num_experts=args.q_expert_num,
                            original_module=original_q,
                            dense_moe=args.dense_moe).bfloat16()
                    original_k = model.base_model.layers[i].self_attn.k_proj
                    model.base_model.layers[i].self_attn.k_proj = \
                        LoRA_MOE_QK_old(args=None,
                            lora_rank=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            num_experts=args.k_expert_num,
                            top_moe_experts=args.top_moe_num,
                            original_module=original_k).bfloat16()
                else:
                    original_q = model.base_model.layers[i].self_attn.q_proj
                    model.base_model.layers[i].self_attn.q_proj = \
                        LoRA_MOE_QK(args=None,
                            lora_rank=args.lora_r * 4,
                            lora_alpha=args.lora_alpha * 4,
                            num_experts=1,
                            original_module=original_q,
                            dense_moe=args.dense_moe).bfloat16()
                    original_k = model.base_model.layers[i].self_attn.k_proj
                    model.base_model.layers[i].self_attn.k_proj = \
                        LoRA_MOE_QK(args=None,
                            lora_rank=args.lora_r * 4,
                            lora_alpha=args.lora_alpha * 4,
                            num_experts=1,
                            original_module=original_k).bfloat16()
            
                
            new_state_dict = {}
            for key, value in moe_state_dict.items():
                # Replace "base_model.model" with an empty string to remove it
                new_key = key.replace("base_model.model", "")
                if new_key.startswith("."):
                    new_key = new_key[1:]
                new_state_dict[new_key] = value.to("cuda")
                # new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=False)
            model = model.to("cuda")
            for key in new_state_dict.keys():
                # if "mm_projector" in key:
                #     continue
                assert torch.equal(model.state_dict()[key], new_state_dict[key]), f"Mismatch in {key}, {model.state_dict()[key].dtype}, {new_state_dict[key].dtype}"
            print("Subset loaded successfully.")
            print('Convert to FP16...')
            model.to(torch.float16)
        elif len(moe_state_dict.keys()) > 1:
            new_state_dict = {}
            for key, value in moe_state_dict.items():
                # Replace "base_model.model" with an empty string to remove it
                new_key = key.replace("base_model.model", "")
                if new_key.startswith("."):
                    new_key = new_key[1:]
                new_state_dict[new_key] = value.to("cuda")
            model.load_state_dict(new_state_dict, strict=False)
            print("Projector loaded successfully.")
            model = model.to("cuda")
            # for key in new_state_dict.keys():
            #     print(model.state_dict()[key], "cernijvj" ,new_state_dict[key])
            #     assert torch.equal(model.state_dict()[key], new_state_dict[key]), f"Mismatch in {key}"
            model.to(torch.float16)

    if args.baseline == "PAI":
        from transformers.generation.logits_process import LogitsProcessorList
        from PAI_files.model_loader import init_cfg_processor
        from PAI_files.attention import llama_modify
        llama_modify(
            model,
            start_layer=2,
            end_layer=32,
            alpha=0.2,
            use_attn=True,
            use_cfg=True,
            img_start_idx=34,
            img_end_idx=34+576
        )

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        if "conversations" in line:
            question = line["conversations"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversations"][1]['value'] # ['value']        
            qs = question['value']
        else:
            question = line['question']
            gt_ans = line['answer']
            qs = question
            if 'choices' in line and len(line['choices']) > 10:
                qs += " Please choose from the following options: " + line['choices']
            if 'question_type' in line and line['question_type'] == 'binary':
                qs += " Please answer Yes or No."

        if "qid" in line:
            idx = line["qid"]
        else:
            idx = line["id"]
        # question = line["conversations"][0] # ['value'].split('\n')[0]
        # gt_ans = line["conversations"][1] # ['value']      
        # image_file = line["image"]
        # assert gt_ans['from'] == 'gpt'
        if 'image' in line:
            image_file = line["image"]
        elif 'img_name' in line:
            image_file = line["img_name"]

        # qs = question['value']
        qs = qs.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


        if args.baseline == "opera":
            key_position = {
                "image_start": 34, 
                "image_end": 34+576,
                "response_start": input_ids.shape[1] + 576
            }
            # print("use opera")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    num_beams=5,
                    do_sample=False,
                    max_new_tokens=128,
                    output_attentions=True,
                    opera_decoding=True,
                    key_position=key_position,
                    scale_factor=25,
                    threshold=25,
                    num_attn_candidates=5,
                    penalty_weights=1,
                    stopping_criteria=[stopping_criteria])
        elif args.baseline == "PAI":
            logits_processor = (
                    init_cfg_processor(tokenizer=tokenizer, llm_model=model, questions=[cur_prompt], gamma=1.1, beam=1, start_layer=2, end_layer=32)
                )
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    use_cache=True,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=1024,
                    output_attentions=False,
                    output_hidden_states=False,
                    logits_processor=LogitsProcessorList([logits_processor]),
                    stopping_criteria=[stopping_criteria])     
        
        elif args.baseline == "VCD":
            # print("use VCD")
            from VCD_files.vcd_add_noise import add_diffusion_noise
            image_tensor_cd = add_diffusion_noise(image_tensor, 500) #args.noise_step
            # image_tensor_cd = None
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                        cd_alpha = 1, #args.cd_alpha,
                        cd_beta = 0.1, #args.cd_beta,
                        do_sample=True,
                        max_new_tokens=1024,
                        temperature=1,
                        output_attentions=False,
                        output_hidden_states=False,
                        stopping_criteria=[stopping_criteria]
                        )
        elif args.baseline == "avisc":
            use_cd = False
            if use_cd:
                image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=500)
            else:
                image_tensor_cd = None
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        images_cd=(image_tensor_cd.half().cuda() if image_tensor_cd is not None else None),
                        cd_alpha=1.0,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=1024,
                        use_avisc=True,
                        layer_gamma=0.5,
                        masking_scheme="zeros",
                        lamb=1.0,
                        temp=1.0,
                        stopping_criteria=[stopping_criteria]
                    )    
        elif args.baseline == "m3id":
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        images_cd=None,
                        cd_alpha=1.0,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=1024,
                        use_avisc=False,
                        use_m3id=True,
                        layer_gamma=0.5,
                        lamb=1.0,
                        stopping_criteria=[stopping_criteria]
                    )    
        elif args.baseline == "DoLa":
            with torch.no_grad():
                early_exit_layers = [0,2,4,6,8,10,12,14,32]
                mature_layer = early_exit_layers[-1]
                premature_layer = None
                candidate_premature_layers = early_exit_layers[:-1]
                output_ids = model.generate(input_ids, 
                                            images=image_tensor.unsqueeze(0).half().cuda(),
                                            max_new_tokens=1024,
                                            dola_decoding=True,
                                            top_p=0.95, top_k=0, temperature=0.9, 
                                            stopping_criteria=[stopping_criteria], relative_top=0.1, 
                                            mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers,
                                            )
        elif args.baseline == "damro":
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        images_cd=None,
                        cd_alpha=0.5,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=1024,
                        use_damro=True,
                        use_avisc=False,
                        layer_gamma=0.5,
                        masking_scheme="zeros",
                        lamb=1.0,
                        temp=1.0,
                        stopping_criteria=[stopping_criteria]
                    )    
        elif args.baseline == "beam":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=False,
                    num_beams=5,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
        elif args.baseline == "greedy":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
        elif args.baseline == "nucleus":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    top_p=0.9,                 # Cumulative probability threshold
                    top_k=0,                   # Optional; setting to 0 disables top-k filtering
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
        else:

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    # do_sample=False,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # num_beams=5,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    # stopping_criteria=stopping_criteria,
                    use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "gt_ans": gt_ans,
                                #    "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--peft-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--k_expert_num", type=int, default=0)
    parser.add_argument("--q_expert_num", type=int, default=0)
    parser.add_argument("--dense_moe", type=bool, default=False)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--top_moe_num", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)