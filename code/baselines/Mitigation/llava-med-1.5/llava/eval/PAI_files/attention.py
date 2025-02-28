import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def my_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, return_attention_weights=False,
                                    use_attn=False, img_start_idx=39, img_end_idx=39+256, use_cfg=False, alpha=0.2) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to("cuda")
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to("cuda")
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            if attn_mask.shape[0] > 1:
                attn_mask = attn_mask[0, ...]
            attn_bias += attn_mask.squeeze(0).squeeze(0)
            # attn_bias += attn_mask.expand_as(attn_bias)
    attn_weights = query @ key.transpose(-2, -1) * scale_factor
    attn_weights += attn_bias

    ### PAI's modification
    # if hasattr(self, "use_attn"):
    #     use_attn = self.use_attn
    #     img_start_idx = self.img_start_idx
    #     img_end_idx = self.img_end_idx
    # else:
    #     use_attn = False

    # if hasattr(self, "use_cfg"):
    #     use_cfg = self.use_cfg
    # else:
    #     use_cfg = False

    if use_attn and not use_cfg:
        attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
            attn_weights[:, :, -1, img_start_idx:img_end_idx].abs() * alpha
            + attn_weights[:, :, -1, img_start_idx:img_end_idx]
        )
    ### PAI's modification

    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
    if return_attention_weights:
        return attn_weights @ value, attn_weights
    return attn_weights @ value, None


def llama_new_forward_caf(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
):
    bsz, q_len, _ = hidden_states.size()
    self.use_moe = False
    if hasattr(self.q_proj, 'num_experts'): # using moe module
        query_out = self.q_proj(hidden_states, **kwargs)
        self.use_moe = True
        if isinstance(query_out, tuple):
            query_states, q_moe_routing = query_out
    else:
        query_states = self.q_proj(hidden_states)
    
    if hasattr(self.k_proj, 'num_experts'): # using moe module
        query_out = self.k_proj(hidden_states)
        self.use_moe = True
        if isinstance(query_out, tuple):
            key_states, k_moe_routing = query_out
    else:
        key_states = self.k_proj(hidden_states)
    
    # key_states = self.k_proj(hidden_states)
    # query_states = self.q_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # attn_output = torch.nn.functional.scaled_dot_product_attention(
    #     query_states,
    #     key_states,
    #     value_states,
    #     attn_mask=attention_mask,
    #     dropout_p=self.attention_dropout if self.training else 0.0,
    #     # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    #     is_causal=self.is_causal and attention_mask is None and q_len > 1,
    # )

    attn_output, attn_weights = my_scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
            return_attention_weights=output_attentions,
            use_attn=self.use_attn,
            use_cfg=self.use_cfg,
            img_start_idx=self.img_start_idx,
            img_end_idx=self.img_end_idx,
            alpha=self.alpha
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if self.use_moe:
        return attn_output, None, past_key_value, k_moe_routing
    return attn_output, None, past_key_value

def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )


    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    ### PAI's modification
    if hasattr(self, "use_attn"):
        use_attn = self.use_attn
        img_start_idx = self.img_start_idx
        img_end_idx = self.img_end_idx
    else:
        use_attn = False

    if hasattr(self, "use_cfg"):
        use_cfg = self.use_cfg
    else:
        use_cfg = False

    if use_attn and not use_cfg:
        attn_weights[:, :, -1, img_start_idx:img_end_idx] = (
            attn_weights[:, :, -1, img_start_idx:img_end_idx].abs() * self.alpha
            + attn_weights[:, :, -1, img_start_idx:img_end_idx]
        )
    ### PAI's modification

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_modify(model, start_layer, end_layer, use_attn, alpha, use_cfg,
                 img_start_idx, img_end_idx):
    for i in range(start_layer, end_layer):
        model.model.layers[i].self_attn.use_attn = use_attn
        model.model.layers[i].self_attn.alpha = alpha
        model.model.layers[i].self_attn.use_cfg = use_cfg
        model.model.layers[i].self_attn.img_start_idx = img_start_idx
        model.model.layers[i].self_attn.img_end_idx = img_end_idx
        model.model.layers[i].self_attn.forward = types.MethodType(llama_new_forward_caf, model.model.layers[i].self_attn)