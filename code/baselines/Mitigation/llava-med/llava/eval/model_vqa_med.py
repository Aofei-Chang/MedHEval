
import sys
sys.path.append("/home/avc6555/research/MedH/Mitigation/LVLMs/llava-med")

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import json
from tqdm import tqdm
import shortuuid


from llava import LlavaLlamaForCausalLM
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.model.moe_llava import LoRA_MOE_FFN, LoRA_MOE_QK, LoRA_MOE_QK_old

from PIL import Image
import random
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"




detail_describe_instructions = [
    "Describe the following image in detail.",
    "Provide a detailed description of the given image.",
    "Give an elaborate explanation of the image you see.",
    "Share a comprehensive rundown of the presented image.",
    "Offer a thorough analysis of the image.",
    "Explain the various aspects of the image before you.",
    "Clarify the contents of the displayed image with great detail.",
    "Characterize the image using a well-detailed description.",
    "Break down the elements of the image in a detailed manner.",
    "Walk through the important details of the image.",
    "Portray the image with a rich, descriptive narrative.",
    "Narrate the contents of the image with precision.",
    "Analyze the image in a comprehensive and detailed manner.",
    "Illustrate the image through a descriptive explanation.",
    "Examine the image closely and share its details.",
    "Write an exhaustive depiction of the given image.",
]

concise_describe_instructions = [
    "Describe the following image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the following image.",
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo below.",
    "Write a terse but informative summary of the following picture.",
    "Create a compact narrative representing the image presented.",
]

prompt_pool = detail_describe_instructions + concise_describe_instructions

prompt_pool = [ "Describe the following image in detail."]


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.baseline == "VCD":
        from VCD_files.vcd_sample import evolve_vcd_sampling
        evolve_vcd_sampling()
    elif args.baseline == "avisc" or args.baseline == "m3id" or args.baseline == "damro":
        print(f"use and set {args.baseline}")
        from avisc_utils.vcd_add_noise import add_diffusion_noise
        from avisc_utils.avisc_sample import evolve_avisc_sampling
        evolve_avisc_sampling()
    

    if args.mm_projector is None:
        patch_config(model_name)
        print("args.mm_projector is None", model_name)
        if "BiomedCLIP" in model_name or "biomed_clip" in model_name:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, use_cache=True).cuda()
            model = model.to(torch.float16)
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            
            openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
            vision_config = openai_vision_tower.config
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)
            setattr(vision_tower, 'config', vision_config)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()
            # print("vision tower", model.config.mm_vision_tower)
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
            vision_tower = model.model.vision_tower[0]
            vision_tower.to(device='cuda', dtype=torch.float16)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        # import pdb; pdb.set_trace()
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        print(image_token_len, "token length")
    else:
        # in case of using a pretrained model with only a MLP projector weights
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.float16).cuda()

        if "BiomedCLIP" in model.config.mm_vision_tower:
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        else:
            image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)


        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        mm_projector = torch.nn.Linear(vision_config.hidden_size, model.config.hidden_size)
        mm_projector_weights = torch.load(args.mm_projector, map_location='cpu')
        mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        model.model.mm_projector = mm_projector.cuda().half()
        model.model.vision_tower = [vision_tower]

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
            # top_layers = [12, 13, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31]
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
                            dense_moe=args.dense_moe)
                    original_k = model.base_model.layers[i].self_attn.k_proj
                    model.base_model.layers[i].self_attn.k_proj = \
                        LoRA_MOE_QK_old(args=None,
                            lora_rank=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            num_experts=args.k_expert_num,
                            top_moe_experts=args.top_moe_num,
                            original_module=original_k)
                else:
                    original_q = model.base_model.layers[i].self_attn.q_proj
                    model.base_model.layers[i].self_attn.q_proj = \
                        LoRA_MOE_QK(args=None,
                            lora_rank=args.lora_r * 4,
                            lora_alpha=args.lora_alpha * 4,
                            num_experts=1,
                            original_module=original_q,
                            dense_moe=args.dense_moe)
                    original_k = model.base_model.layers[i].self_attn.k_proj
                    model.base_model.layers[i].self_attn.k_proj = \
                        LoRA_MOE_QK(args=None,
                            lora_rank=args.lora_r * 4,
                            lora_alpha=args.lora_alpha * 4,
                            num_experts=1,
                            original_module=original_k)
            
                
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
                assert torch.equal(model.state_dict()[key], new_state_dict[key]), f"Mismatch in {key}"
            print("Subset loaded successfully.")
            print('Convert to FP16...')
    print("Model loaded, to cuda and float16")
    model = model.to("cuda")

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
            img_start_idx=39,
            img_end_idx=39+256
        )
        
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # os.makedirs(os.path.join(os.path.dirname(answers_file), "images"), exist_ok=True)
    ans_file = open(answers_file, "w")
    # save_image_folder = os.path.join(os.path.dirname(os.path.expanduser(args.answers_file)), "images")
    print(args.conv_mode, "conv_mode")
    for i, line in enumerate(tqdm(questions)):
        if "qid" in line:
            idx = line["qid"]
        else:
            idx = line["id"]
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

        qs = qs.replace('<image>\n', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
        elif 'img_name' in line:
            image_file = line["img_name"]
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        images = image_tensor.unsqueeze(0).half().cuda()
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n' +  qs
        else:
            qs = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + '\n' + qs
        cur_prompt = '<image>' + '\n' +  cur_prompt
        # else:
        #     images = None

        if args.conv_mode == 'simple_legacy':
            qs += '\n\n### Response:'
        # assert gt_ans['from'] == 'gpt'
        # conv = default_conversation.copy()
        conv = conv_templates[args.conv_mode].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        # print(prompt)
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        if args.baseline == "opera":
            key_position = {
                "image_start": 39, 
                "image_end": 39+256,
                "response_start": input_ids.shape[1]
            }
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    num_beams=5,
                    do_sample=False,
                    max_new_tokens=512,
                    output_attentions=True,
                    opera_decoding=True,
                    key_position=key_position,
                    scale_factor=50,
                    threshold=15,
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
                    images=images,
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
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        images=images,
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
                        images=images,
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
                        images=images,
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
                                            images=images,
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
                        images=images,
                        images_cd=None,
                        cd_alpha=0.5,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=1024,
                        use_damro=True,
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
                    images=images,
                    do_sample=False,
                    # do_sample=args.num_beams == 1,
                    # temperature=0 if args.num_beams == 1 else 0.7,
                    num_beams=5,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
        elif args.baseline == "greedy":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
        elif args.baseline == "nucleus":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    top_p=0.9,                 # Cumulative probability threshold
                    top_k=0,                   # Optional; setting to 0 disables top-k filtering
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
        else:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])

        # TODO: new implementation
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] Sample {i}: {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        if args.conv_mode == 'simple_legacy':
            while True:
                cur_len = len(outputs)
                outputs = outputs.strip()
                for pattern in ['###', 'Assistant:', 'Response:']:
                    if outputs.startswith(pattern):
                        outputs = outputs[len(pattern):].strip()
                if len(outputs) == cur_len:
                    break

        try:
            index = outputs.index(conv.sep)
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep)

        outputs = outputs[:index].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "gt": gt_ans,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--peft-path", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--mm-projector", type=str, default=None)
    parser.add_argument("--vision-tower", type=str, default=None)
    # parser.add_argument("--conv-mode", type=str, default="simple")
    parser.add_argument("--conv-mode", type=str, default="default")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--k_expert_num", type=int, default=0)
    parser.add_argument("--q_expert_num", type=int, default=0)
    parser.add_argument("--dense_moe", type=bool, default=False)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--top_moe_num", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
