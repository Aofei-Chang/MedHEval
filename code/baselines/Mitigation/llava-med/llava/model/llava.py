#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from llava.model.utils import *
import open_clip
import os, json
import copy

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None, use_visual_prompt=False):
        super(LlavaLlamaModel, self).__init__(config)

        self.vision_tower_name = "openai/clip-vit-large-patch14" # microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 # openai/clip-vit-large-patch14
        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            
            if "BiomedCLIP" in config.mm_vision_tower or "biomed_clip" in config.mm_vision_tower:
                model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.vision_tower = [model.visual.trunk] # Please refer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/timm_model.py#LL60C18-L60C18
                self.vision_tower_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            else:
                self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]


        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        self.use_visual_prompt = use_visual_prompt
        # self.visual_prompt = torch.nn.Parameter(torch.zeros(1, 256, 4096))

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):

        if "BiomedCLIP" in vision_tower:
            self.vision_tower_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            return self.initialize_vision_modules_from_biomed_clip(vision_tower, mm_vision_select_layer,
                                pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False)
        else:
            return self.initialize_vision_modules_from_openai_clip(vision_tower, mm_vision_select_layer,
                                pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False)

    def prepare_input_embeds(self, input_ids, inputs_embeds, images):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # image_features_before_projection, image_features = None, None
        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_feature, dummy_image_features = self.extract_visual_features(vision_tower, image.unsqueeze(0))
                        image_features.append(image_feature)
                else:
                    image_features, dummy_image_features = self.extract_visual_features(vision_tower, images)
            # image_features_before_projection = copy.deepcopy(image_features)
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            
            
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            orig_embeds_params = getattr(self, 'orig_embeds_params', None)
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]

                        # import pdb; pdb.set_trace()
                        # print(image_features.size() if image_features is not None else None, "fnjdd")
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                        # print(image_features.size() if image_features is not None else None, "fnj332")
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    # print(image_features.size(), "fnjdd")
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
        return inputs_embeds

    def initialize_vision_modules_from_openai_clip(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def initialize_vision_modules_from_biomed_clip(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower

        

        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        vision_config = openai_vision_tower.config
        del openai_vision_tower
                
        if not hasattr(self, 'vision_tower'):
            model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            vision_tower = model.visual.trunk # Please refer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/timm_model.py#LL60C18-L60C18

            # from huggingface_hub import snapshot_download
            # BiomedCLIP_file_path = "biomed-clip-share"
            # # snapshot_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", local_dir=BiomedCLIP_file_path)
            # with open(os.path.join(BiomedCLIP_file_path, "open_clip_config.json"), 'r') as file:  
            #     config = json.load(file) 


        else:
            vision_tower = self.vision_tower[0]
        
        setattr(vision_tower, 'config', vision_config)
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def get_damro_mask_idx(self, attentions, topK=10):
        damro_mask_idx = None
        attention_last_layer = attentions[-1]
        # Step 1: Average the attention across all heads
        # Shape: [1, 257, 257]
        averaged_attention = attention_last_layer.mean(dim=1)
        # Step 2: Extract attention scores for the [CLS] token (index 0)
        # Shape: [1, 257]
        cls_attention = averaged_attention[:, 0, :]
        # Step 3: Exclude the [CLS] token itself and calculate top-10 visual tokens
        # Shape: [1, 256]
        visual_tokens_attention = cls_attention[:, 1:]
        # Find the indices of the top-10 tokens with highest attention scores
        # Shape: [10]
        top_10_indices = torch.topk(visual_tokens_attention, 10, dim=1).indices
        all_indices = torch.arange(0, 256)
        remaining_indices = torch.tensor([i for i in all_indices if i not in top_10_indices])

        # Final output shape: [1, 246]
        remaining_indices_output = remaining_indices.unsqueeze(0)
        damro_mask_idx = remaining_indices_output
        return damro_mask_idx

    def extract_visual_features(self, vision_tower, images, output_attentions=False):
        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
        damro_mask_idx = None
        
        if "BiomedCLIP" in self.vision_tower_name  or "biomed_clip" in self.vision_tower_name:
            image_forward_outs = vision_tower.get_intermediate_layers(images, n=3) # take last n blocks if n is an int, if in is a sequence, select by matching indices
            image_features = image_forward_outs[select_hidden_state_layer]
            image_features = image_features
            dummy_image_features = torch.zeros(196, 768, device=image_features.device, dtype=image_features.dtype)
        else:
            # print(output_attentions, "out attentions")
            image_forward_outs = vision_tower(images, output_hidden_states=True, output_attentions=output_attentions)
            attentions = image_forward_outs.attentions
            if attentions is not None:
                damro_mask_idx = self.get_damro_mask_idx(attentions)
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            image_features = select_hidden_state[:, 1:]
            dummy_image_features = torch.zeros(256, 1024, device=image_features.device, dtype=image_features.dtype)
        
        return image_features, dummy_image_features, damro_mask_idx

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_prompt: Optional[torch.nn.Parameter] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        img_idx: Optional[Tuple] = None,
        mask_idx: Optional[torch.Tensor] = None,
        masking_scheme=None,
        # we need to get vision attentions here
        use_damro=False,
        out_vit_attention=False,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # print("use_damro", use_damro, masking_scheme)
        damro_mask_idx = None
        # output_vit_attentions = True if (use_damro and not use_cache and past_key_values is not None) else False
        output_vit_attentions = out_vit_attention
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data
        # print(inputs_embeds.size() if inputs_embeds is not None else "no input embed", "input_embeds size")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # image_features_before_projection, image_features = None, None
        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_feature, dummy_image_features, damro_mask_idx = self.extract_visual_features(vision_tower, image.unsqueeze(0), output_attentions=output_vit_attentions)
                        image_features.append(image_feature)
                else:
                    image_features, dummy_image_features, damro_mask_idx = self.extract_visual_features(vision_tower, images, output_attentions=output_vit_attentions)
            # image_features_before_projection = copy.deepcopy(image_features)
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            
            # if self.use_visual_prompt:
            #     if self.visual_prompt is None:
            #         print("initialize visual prompt")
            #         self.visual_prompt = torch.nn.Parameter(torch.zeros_like(image_features[0])) if type(images) is list else torch.nn.Parameter(torch.zeros_like(image_features))
            #         self.visual_prompt.requires_grad = True
            #         print("visual prompt initialized", self.visual_prompt.size())
            #     if type(images) is list:
            #         image_features = [image_feature + self.visual_prompt for image_feature in image_features]
            #     else:
            #         image_features = image_features + self.visual_prompt
            if visual_prompt is not None:
                # print(visual_prompt.size(), "visual prompt size, not None")
                if type(images) is list:
                    image_features = [image_feature + visual_prompt for image_feature in image_features]
                else:
                    image_features = image_features + visual_prompt
                
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]

                        # import pdb; pdb.set_trace()
                        # print(image_features.size() if image_features is not None else None, "fnjdd")
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                        # print(image_features.size() if image_features is not None else None, "fnj332")
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    # print(image_features.size(), "fnjdd")
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
        #for avisc and damro
        if damro_mask_idx is not None:
            # print(damro_mask_idx.size(), "damro mask idx size", damro_mask_idx)
            mask_idx = damro_mask_idx
        if mask_idx is not None and past_key_values is None:
            # top-k masking
            # for att_mask, idx in zip(attention_mask, mask_idx):
            #     att_mask[idx] = 0

            #token noising    
            for input_embed, idx in zip(inputs_embeds, mask_idx):
                # input_embed[idx] = torch.randn(input_embed[idx].size(), dtype=input_embed.dtype).to(input_embed.device) * 0.1
                #input_embed[idx] = add_diffusion_noise(input_embed[idx], noise_step=500)
                if masking_scheme.lower() == "ones":
                    input_embed[idx + 39] = 1.0
                    # print("ones")
                elif masking_scheme.lower() == "zeros":
                    # print("zeros masking")
                    input_embed[idx + 39] = 0.0
                    # print("zeros")
                elif masking_scheme.lower() == "noise":
                    input_embed[idx + 39] = torch.randn(input_embed[idx + 39].size(), dtype=input_embed.dtype).to(input_embed.device)
                    # print("noise")
                else:
                    input_embed[idx + 39] = 0.0

        outputs = super(LlavaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_token_start_index = kwargs['image_token_start_index'],
            question_token_end_index = kwargs['question_token_end_index']
        )
        
        return outputs


class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, enhance_visual=False):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.enhance_visual = enhance_visual
        self.visual_enhance_ratio=0.1
        self.bbox_ratio=0.05
        self.use_moe = False
        self.moe_balance_ratio = 0.05
        self.num_experts = 4
        self.top_heads = 0
        self.use_kl = False
        self.use_visual_prompt = False
        self.visual_prompt = None
        self.init_visual_prompt_flag = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        bboxes: Optional[list] = None,
        image_token_start_index = None,
        question_token_end_index = None,
        # VCD_parameters
        images_cd: Optional[torch.FloatTensor] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        # avisc parameters
        img_idx: Optional[Tuple] = None,
        mask_idx: Optional[torch.Tensor] = None,
        use_avisc: Optional[bool] = None,
        layer_gamma=None,
        masking_scheme=None,
        lamb=None,
        temp=None,
        use_m3id=None,
        use_damro=False,
        out_vit_attention=False,
        # DoLa parameters
        early_exit_layers: Optional[List[int]] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # init visual prompt
        # if not self.init_visual_prompt_flag:
        #     print("initialize visual prompt, outer loop")
        #     self.model.use_visual_prompt = self.use_visual_prompt
        #     self.init_visual_prompt_flag = True
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        global attention_head_counter
        attention_head_counter = 0

        # print(self.visual_prompt.size() if self.visual_prompt is not None else "no visual prompt", "visual prompt size")
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if bboxes is not None or self.enhance_visual:
            output_attentions = True
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_prompt=self.visual_prompt,
            use_cache=use_cache,
            output_attentions=output_attentions,
            # output_attentions=True,
            # output_hidden_states=output_hidden_states,
            # for DoLa baseline
            output_hidden_states=output_hidden_states or early_exit_layers is not None,
            return_dict=return_dict,
            images=images,
            image_token_start_index = image_token_start_index,
            question_token_end_index = question_token_end_index,
            mask_idx=mask_idx,
            masking_scheme=masking_scheme,
            use_damro=use_damro,
            out_vit_attention=out_vit_attention
        )



        #baseline DoLa
        if early_exit_layers is not None:
            logits_dict = {}
            # loss_dict = {}
            for i, early_exit_layer in enumerate(early_exit_layers):
                logits = self.lm_head(outputs.hidden_states[early_exit_layer])
                logits_dict[early_exit_layer] = logits
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                    # loss_dict[early_exit_layer] = loss
                
            final_outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            return logits_dict, final_outputs

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss, loss_visual = None, None
        if bboxes is not None:
            # loss_bbox = calculate_attention_loss(all_attention_maps=outputs[1], bboxes=bboxes)
            if self.top_heads:
                loss_bbox = calculate_top_attention_loss(all_attention_maps=outputs[1], bboxes=bboxes, top_heads=self.top_heads, use_KL=self.use_kl)
            else:
                loss_bbox = calculate_attention_loss_new(all_attention_maps=outputs[1], bboxes=bboxes, use_KL=self.use_kl)
            # loss_bbox = calculate_top_attention_loss(all_attention_maps=outputs[1], bboxes=bboxes)
            # loss_bbox = calculate_top_attention_loss(all_attention_maps=outputs[1], bboxes=bboxes)

        if self.enhance_visual:
            loss_visual = calculate_visual_loss(all_attention_maps=outputs[1], lambda_reg=self.visual_enhance_ratio, top_heads=True)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if bboxes is not None:
                loss += self.bbox_ratio * loss_bbox
            if loss_visual is not None:
                loss += loss_visual
        # moe loss
        
        
        if self.use_moe and self.moe_balance_ratio > 0:
            # print("moe loss", self.moe_balance_ratio)
            batch_size = hidden_states.shape[0]
            routings = outputs[-1]
            # print("routings", len(routings), routings)
            llm_mlp_routing_probs = torch.stack([r[0] for r in routings], dim=0) # [layer, batch, seq_len, num_experts]
            llm_mlp_routing_idxes = torch.stack([r[1] for r in routings], dim=0).detach()

            llm_mlp_expert_balancing_loss = 0.
            for i in range(batch_size):
                probs_i = llm_mlp_routing_probs[:,i].reshape(-1, self.num_experts)
                idxes_i = llm_mlp_routing_idxes[:,i].reshape(-1, self.num_experts)
                # probs_i = llm_mlp_routing_probs[:,i, attention_mask[i].bool()].reshape(-1, self.num_experts)
                # idxes_i = llm_mlp_routing_idxes[:,i, attention_mask[i].bool()].reshape(-1, self.num_experts)

                llm_mlp_expert_balancing_loss += (probs_i.mean(0) * idxes_i.mean(0)).sum()

            loss += llm_mlp_expert_balancing_loss/batch_size * self.moe_balance_ratio
            # print("moe loss", llm_mlp_expert_balancing_loss/batch_size * self.moe_balance_ratio)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        outputs = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values, 
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

        # outputs.additional_outputs = additional_outputs

        return outputs

    def prepare_inputs_for_generation_method(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    #for baseline VCD
    def prepare_inputs_for_generation_cd(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
            }
        )
        return model_inputs

    def prepare_inputs_for_generation_m3id(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids[input_ids != -200].unsqueeze(0)}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask[:,:-1],
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.model.vision_tower[0].config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

# AutoConfig.register("llava", LlavaConfig)
# AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
