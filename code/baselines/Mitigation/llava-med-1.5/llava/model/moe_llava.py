from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class LoRALayer(nn.Module):
    def __init__(self, fan_in, fan_out, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        self.lora_A = nn.Parameter(torch.zeros((rank, fan_in)))
        self.lora_B = nn.Parameter(torch.zeros((fan_out, rank)))
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        # initialize B the same way as the default for nn.Linear and A to zero
        # this is different than what is described in the paper but should not affect performance
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, X):             
        result = (self.lora_dropout(X) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        # result += F.linear(x, T(self.weight), bias=self.bias)
        return result

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )


class LoRA_MOE_QK_old(nn.Module): # for Q,K,V,O in attention

    def __init__(
            self,
            args,
            lora_rank: int,
            lora_alpha: int,
            num_experts: int,
            original_module: nn.Module = None,
            image_start_pos: int = 34,
            top_moe_experts: int = 1,
            question_only: bool = False
    ):
        super().__init__()
        self.args = args
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts
        self.image_start_pos = image_start_pos
        self.top_moe_experts = top_moe_experts

        d_model = original_module.in_features
        out_dim = original_module.out_features
        # print(d_model, out_dim, "d_model, out_dim, key")
        # mlp_width = original_module.gate_proj.out_features
        self.original_module = original_module

        self.moe = nn.ModuleList()
        self.original_module.weight.requires_grad = False

        for _ in range(num_experts):
            self.moe.append(LoRALayer.from_linear(
                nn.Linear(d_model, out_dim),
                rank=self.lora_rank,
                lora_dropout_p=args.lora_dropout if args is not None else 0.05,
                lora_alpha=self.lora_alpha
                ))
        
        self.router = nn.Linear(d_model, self.num_experts)

    def forward_lora_moe(self, x, original_proj, routing, moe):
        original_out = original_proj(x)
        lora_out_per_expert = []
        for i in range(self.num_experts):
            lora_out_per_expert.append(moe[i](x))

        lora_out = torch.stack(lora_out_per_expert, 2)

        lora_out = (lora_out * routing[:,:,:,None]).sum(2)

        moe_out = original_out + lora_out
        return moe_out

    def forward_lora_moe_sparse(self, x, original_proj, routing_idx, moe):
        original_out = original_proj(x)

        lora_out = torch.zeros_like(original_out)
        # print(lora_out.size(), "lora_out")
        for i in range(self.num_experts):
            id1, id2, _ = torch.where(routing_idx==i)
            moe_out = moe[i](x[id1, id2])
            # print(moe_out.size(), "moe_our")
            lora_out[id1, id2] = moe_out

        moe_out = original_out + lora_out
        return moe_out
    # def forward_lora_moe_sparse(self, x, original_proj, routing_idx, moe):
    #     original_out = original_proj(x)
    #     lora_out = torch.zeros_like(original_out)

    #     # Process the top-2 experts
    #     for i in range(self.num_experts):
    #         id1, id2, _ = torch.where(routing_idx == i)
    #         if id1.numel() > 0:  # Check if there are any tokens routed to this expert
    #             moe_out = moe[i](x[id1, id2])
    #             lora_out[id1, id2] = moe_out

    #     moe_out = original_out + lora_out
    #     return moe_out

    def forward(self, x):
        routing, expert_choice = None, None
        if x.size(1) > 1:  # Initial forward pass
            image_start_index = self.image_start_pos
            x_non_image_part = x[:, :image_start_index, :]  # Non-question part (before question_start_index)
            x_non_image_part1 = x[:, image_start_index+576:, :]  # Non-question part (before question_start_index)
            x_image_part = x[:, image_start_index:image_start_index+576, :]      # Qu
            non_image_out = self.original_module(x_non_image_part)
            non_image_out1 = self.original_module(x_non_image_part1)
            logits = self.router(x_image_part)
            routing = F.softmax(logits, dim=-1)

            if self.top_moe_experts == 1:
                index = routing.max(-1, keepdim=True)[1]
                y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
                expert_choice = y_hard - routing.detach() + routing
                moe_out = self.forward_lora_moe_sparse(x_image_part, self.original_module, index, self.moe)
            else:
                # Select top-k experts
                topk_indices = torch.topk(routing, k=self.top_moe_experts, dim=-1)[1]  # Get indices of top-2 experts
                y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
                
                # Scatter values for top-k experts
                for i in range(self.top_moe_experts):
                    y_hard.scatter_(-1, topk_indices[:, :, i].unsqueeze(-1), 1.0)
                expert_choice = y_hard - routing.detach() + routing
                moe_out = self.forward_lora_moe_sparse(x_image_part, self.original_module, topk_indices, self.moe)

            # index = routing.max(-1, keepdim=True)[1]
            # y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
            # # # Select top-2 experts
            # # topk_indices = torch.topk(routing, k=2, dim=-1)[1]  # Get indices of top-2 experts
            # # y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
            
            # # # Scatter values for top-2 experts
            # # for i in range(2):
            # #     y_hard.scatter_(-1, topk_indices[:, :, i].unsqueeze(-1), 1.0)

            # expert_choice = y_hard - routing.detach() + routing

            # moe_out = self.forward_lora_moe_sparse(x_image_part, self.original_module, index, self.moe)
            # moe_out = self.forward_lora_moe_sparse(x_image_part, self.original_module, topk_indices, self.moe)
            final_out = torch.cat([non_image_out, moe_out, non_image_out1], dim=1)  # Concatenate along the sequence dimension
        else:
            final_out = self.original_module(x)
        return final_out, (routing, expert_choice)
        # return final_out
    

class LoRA_MOE_QK(nn.Module):  # for Q,K,V,O in attention

    def __init__(
            self,
            args,
            lora_rank: int,
            lora_alpha: int,
            num_experts: int,
            original_module: nn.Module = None,
            image_start_pos: int = 34,
            question_only: bool = False,
            dense_moe: bool = False
    ):
        super().__init__()
        self.args = args
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts
        self.image_start_pos = image_start_pos
        self.dense_moe = dense_moe
        self.all_hidden_forward = True

        d_model = original_module.in_features
        out_dim = original_module.out_features
        self.original_module = original_module
        # print(d_model, out_dim, "d_model, out_dim, query")
        self.moe = nn.ModuleList()
        self.original_module.weight.requires_grad = False
        self.routing_cache = None # cache at inference stage

        for _ in range(num_experts):
            self.moe.append(LoRALayer.from_linear(
                nn.Linear(d_model, out_dim),
                rank=self.lora_rank,
                lora_dropout_p=args.lora_dropout if args is not None else 0.05,
                lora_alpha=self.lora_alpha
                ))
        
        # Router for expert selection, same for the whole sequence
        self.router = nn.Linear(d_model, self.num_experts)

    def forward_lora_moe_dense(self, x, original_proj, routing, moe):
        original_out = original_proj(x)
        lora_out_per_expert = []
        for i in range(self.num_experts):
            lora_out_per_expert.append(moe[i](x))
        # print(f"before stack, {lora_out_per_expert[0].size()}")
        lora_out = torch.stack(lora_out_per_expert, 2)
        # print(f"after stack, {lora_out.size()}, routing {routing.size()}")
        # Apply the expert routing decision (sequence-level)
        lora_out = (lora_out * routing[:, None, :, None]).sum(2)
        # print(f"lora out {lora_out.size()}")
        moe_out = original_out + lora_out
        return moe_out

    def forward_lora_moe_sparse(self, x, original_proj, routing_idx, moe):
        original_out = original_proj(x)

        lora_out = torch.zeros_like(original_out)
        # if top1
        expert_id = routing_idx.cpu().item()
        # print(routing_idx.cpu().item())
        lora_out = lora_out + moe[expert_id](x)
        # for i in range(self.num_experts):
        #     id1, id2 = torch.where(routing_idx == i)
            # lora_out[id1] = moe[i](x[id1])

        moe_out = original_out + lora_out
        return moe_out

    def forward(self, x, **kwargs):
        if self.num_experts == 1:
            lora_out = self.moe[0](x)
            final_out = self.original_module(x) + lora_out
            # return final_out, (None, None)
            return final_out

        question_end_index = -1
        question_start_index = self.image_start_pos + 576 + 1
        if 'question_token_end_index' in kwargs and 'image_token_start_index' in kwargs:
            # print(f"use moeeee kwargs: {kwargs}")
            if kwargs['image_token_start_index'] is not None:
                if isinstance(kwargs['image_token_start_index'], list):
                    question_start_index = kwargs['image_token_start_index'][0] + 576
                    question_end_index = kwargs['question_token_end_index'][0]
                else:
                    question_start_index = kwargs['image_token_start_index'] + 576
                    question_end_index = kwargs['question_token_end_index']

        if x.size(1) > 1:  # Initial forward pass
            # question_start_index = self.image_start_pos + 256 + 1
            # print(question_start_index, x.size(),"inputss > 1", question_end_index)
            x_non_question_part = x[:, :question_start_index, :]
            x_question_part = x[:, question_start_index:question_end_index, :]
            x_non_question_part2 = x[:, question_end_index:, :]
            non_question_out = None
            non_question_out2 = None
            
            if not self.all_hidden_forward:
                non_question_out = self.original_module(x_non_question_part)
                non_question_out2 = self.original_module(x_non_question_part2)

            # Sequence-level routing decision
            x_aggregated = x_question_part.mean(dim=1)
            logits = self.router(x_aggregated)
            routing = F.softmax(logits, dim=-1)
            expert_choice = None
            if self.dense_moe:
                if self.all_hidden_forward:
                    moe_out = self.forward_lora_moe_dense(x, self.original_module, routing, self.moe)
                    self.routing_cache = (routing, None)
                    # return moe_out, (routing, None)
                    return moe_out
                moe_out = self.forward_lora_moe_dense(x_question_part, self.original_module, routing, self.moe)
            else:
                _, top_k_indices = torch.topk(routing, k=1, dim=-1)
                y_hard = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
                expert_choice = y_hard - routing.detach() + routing
                moe_out = self.forward_lora_moe_sparse(x_question_part, self.original_module, top_k_indices, self.moe)

            self.routing_cache = (routing, expert_choice)  # Cache routing for subsequent tokens
            final_out = torch.cat([non_question_out, moe_out, non_question_out2], dim=1)
        else:  # Inference with KV cache
            # use the following code if the answer is considered in the training of MoE
            # print(x.size(),"inputss == 1")
            if self.all_hidden_forward:
                if self.routing_cache is not None:
                    routing, expert_choice = self.routing_cache
                else:
                    raise ValueError("Routing cache is empty during inference stage.")

                # Only process the new token (single token forward)
                if self.dense_moe:
                    final_out = self.forward_lora_moe_dense(x, self.original_module, routing, self.moe)
                else:
                    final_out = self.forward_lora_moe_sparse(x, self.original_module, expert_choice.argmax(dim=-1), self.moe)
            else:
                final_out = self.original_module(x)

        # return final_out, self.routing_cache
        return final_out


    # def forward(self, x):
    #     question_start_index = self.image_start_pos + 256 + 1
    #     print(question_start_index, x.size(),"inputss")
    #     x_non_question_part = x[:, :question_start_index, :]  # Non-question part (before question_start_index)
    #     x_question_part = x[:, question_start_index:, :]      # Question part (after question_start_index)
        
    #     # Non-question part processed normally
    #     non_question_out = None
    #     if not self.all_hidden_forward:
    #         non_question_out = self.original_module(x_non_question_part)
        
    #     # Sequence-level routing decision
    #     # Aggregate token embeddings across the sequence by averaging
    #     x_aggregated = x_question_part.mean(dim=1)  # Shape: (batch_size, d_model)

    #     # Compute expert routing logits using the aggregated sequence embedding
    #     logits = self.router(x_aggregated)  # Shape: (batch_size, num_experts)
    #     routing = F.softmax(logits, dim=-1)  # Softmax to get routing weights over experts

    #     expert_choice = None
    #     if self.dense_moe:
    #         if self.all_hidden_forward:
    #             moe_out = self.forward_lora_moe_dense(x, self.original_module, routing, self.moe)
    #             return moe_out, (routing, None)
    #         else:
    #             moe_out = self.forward_lora_moe_dense(x_question_part, self.original_module, routing, self.moe)
    #     else:
    #         # Top-K gating: for simplicity, we'll use K=2 (modify as needed)
    #         _, top_k_indices = torch.topk(routing, k=1, dim=-1)
            
    #         # Create a routing mask for the top-K experts
    #         y_hard = torch.zeros_like(logits).scatter_(-1, top_k_indices, 1.0)
            
    #         # Expert choice: soft + hard gating (straight-through estimator)
    #         expert_choice = y_hard - routing.detach() + routing
            
    #         # Sparse forward pass using the top-K experts
    #         moe_out = self.forward_lora_moe_sparse(x_question_part, self.original_module, top_k_indices, self.moe)
    #     # Concatenate non-question and MoE outputs
    #     final_out = torch.cat([non_question_out, moe_out], dim=1)  # Concatenate along the sequence dimension
    #     return final_out, (routing, expert_choice)



class LoRA_MOE_FFN(nn.Module): # for llm

    def __init__(
            self,
            args,
            lora_rank: int,
            lora_alpha: int,
            num_experts: int,
            original_module: nn.Module = None,
    ):
        super().__init__()
        self.args = args
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts

        d_model = original_module.gate_proj.in_features
        mlp_width = original_module.gate_proj.out_features
        self.original_module = original_module

        self.moe_gate = nn.ModuleList()
        self.moe_down = nn.ModuleList()
        self.moe_up = nn.ModuleList()
        self.original_module.gate_proj.weight.requires_grad = False
        self.original_module.down_proj.weight.requires_grad = False
        self.original_module.up_proj.weight.requires_grad = False

        for _ in range(num_experts):
            self.moe_gate.append(LoRALayer.from_linear(
                nn.Linear(d_model, mlp_width),
                rank=self.lora_rank,
                lora_dropout_p=0.05,
                lora_alpha=self.lora_alpha
                ))
            self.moe_up.append(LoRALayer.from_linear(
                nn.Linear(d_model, mlp_width),
                rank=self.lora_rank,
                lora_dropout_p=0.05,
                lora_alpha=self.lora_alpha
                ))
            self.moe_down.append(LoRALayer.from_linear(
                nn.Linear(mlp_width, d_model),
                rank=self.lora_rank,
                lora_dropout_p=0.05,
                lora_alpha=self.lora_alpha
                ))
        
        self.router = nn.Linear(d_model, self.num_experts)

    def forward_lora_moe(self, x, original_proj, routing, moe):
        original_out = original_proj(x)
        lora_out_per_expert = []
        for i in range(self.num_experts):
            lora_out_per_expert.append(moe[i](x))

        lora_out = torch.stack(lora_out_per_expert, 2)

        lora_out = (lora_out * routing[:,:,:,None]).sum(2)

        moe_out = original_out + lora_out
        return moe_out

    def forward_lora_moe_sparse(self, x, original_proj, routing_idx, moe):
        original_out = original_proj(x)

        lora_out = torch.zeros_like(original_out)
        for i in range(self.num_experts):
            id1, id2, _ = torch.where(routing_idx==i)
            lora_out[id1, id2] = moe[i](x[id1, id2])

        moe_out = original_out + lora_out
        return moe_out


    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        logits = self.router(x)
        routing = F.softmax(logits, dim=-1)
        index = routing.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        expert_choice = y_hard - routing.detach() + routing

        if self.args.dense_moe:
            gate_moe_out = self.forward_lora_moe(x, self.original_module.gate_proj, routing, self.moe_gate)
            up_moe_out = self.forward_lora_moe(x, self.original_module.up_proj, routing, self.moe_up)
        else:
            gate_moe_out = self.forward_lora_moe_sparse(x, self.original_module.gate_proj, index, self.moe_gate)
            up_moe_out = self.forward_lora_moe_sparse(x, self.original_module.up_proj, index, self.moe_up)

        x = self.original_module.act_fn(gate_moe_out) * up_moe_out
        
        if self.args.dense_moe:
            x = self.forward_lora_moe(x, self.original_module.down_proj, routing, self.moe_down)
        else:
            x = self.forward_lora_moe_sparse(x, self.original_module.down_proj, index, self.moe_down)
        return x, (routing, expert_choice)