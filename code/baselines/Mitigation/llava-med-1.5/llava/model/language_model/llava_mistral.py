from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config, tuple_params=None):
        # print(tuple_params, "tuple_params")
        super(MistralForCausalLM, self).__init__(config, tuple_params=tuple_params)
        self.model = LlavaMistralModel(config)
        self.tuple_params = tuple_params

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
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
        # print(input_ids, "input_ids")
        # print(bboxes, "bbox, first layer")
        damro_mask_idx = None
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                damro_mask_idx
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                out_vit_attention=out_vit_attention
            )
        #     print(f"init embeddings, {damro_mask_idx}, {out_vit_attention}")
        # print(f"existing embeddings!")
        if self.damro_mask_idx is not None and past_key_values is None:
            mask_idx = self.damro_mask_idx
        if mask_idx is not None and past_key_values is None and masking_scheme is not None:
            # top-k masking
            # for att_mask, idx in zip(attention_mask, mask_idx):
            #     att_mask[idx] = 0
            # print(f"use damro masks", mask_idx.size())
            #token noising    
            for input_embed, idx in zip(inputs_embeds, mask_idx):
                # input_embed[idx] = torch.randn(input_embed[idx].size(), dtype=input_embed.dtype).to(input_embed.device) * 0.1
                #input_embed[idx] = add_diffusion_noise(input_embed[idx], noise_step=500)
                if masking_scheme.lower() == "ones":
                    input_embed[idx + 34] = 1.0
                    # print("ones")
                elif masking_scheme.lower() == "zeros":
                    input_embed[idx + 34] = 0.0
                    # print("zeros")
                elif masking_scheme.lower() == "noise":
                    input_embed[idx + 34] = torch.randn(input_embed[idx + 35].size(), dtype=input_embed.dtype).to(input_embed.device)
                    # print("noise")
                else:
                    input_embed[idx + 34] = 0.0
        # print(inputs_embeds.size(), "input_embeds")
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bboxes=bboxes, 
            tuple_params=self.tuple_params,
            image_token_start_index = image_token_start_index,
            question_token_end_index = question_token_end_index,
            early_exit_layers=early_exit_layers,
        )

    # @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # if inputs_embeds is not None:
        #     return super().generate(
        #             position_ids=position_ids,
        #             attention_mask=attention_mask,
        #             inputs_embeds=inputs_embeds,
        #             **kwargs
        #         )
        # print("generate!")
        use_damro = False
        if "use_damro" in kwargs:
            use_damro = kwargs.get("use_damro")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                damro_mask_idx
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                out_vit_attention=use_damro,
            )
            # print(f"generate damro masks, {damro_mask_idx}, {use_damro}")
            self.damro_mask_idx = damro_mask_idx   
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
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

        prepared_inputs_ = kwargs.get("images_cd", None)

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
            
            # print(inputs_embeds.size(), "inputs_embeds")
            # here is a bug, the inputs_embeds should not contain the original image input
            # model_inputs = {"inputs_embeds": inputs_embeds}
            inputs_embeds = torch.cat([inputs_embeds[:,:34,:], inputs_embeds[:,34+576:,:]], dim=1)
            # print(inputs_embeds.size(), "inputs_embeds")
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids[input_ids != -200].unsqueeze(0)}
        # print(model_inputs, "nbij")
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask[:,:-1],
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
