from typing import List
from itertools import chain

import numpy as np
from onnx import TensorProto, helper

import math
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn

import onnxruntime

from transformers import PretrainedConfig
from model.configuration_mixformer_sequential import MixFormerSequentialConfig
from model.modeling_mixformer_sequential import InferenceParams, MHA, MLP

torch.manual_seed(0)

def create_block_graph(
    num_heads,
    head_size,
    ln_weight_t,
    ln_bias_t,
    attn_qkv_weight_t,
    attn_qkv_bias_t,
    attn_out_weight_t,
    attn_out_bias_t,
    mlp_fc1_weight_t,
    mlp_fc1_bias_t,
    mlp_fc2_weight_t,
    mlp_fc2_bias_t,
):
    hidden_size = num_heads * head_size

    subgraph_nodes = [
        helper.make_node(
            "LayerNormalization",
            inputs=["i_hidden_states", "ln_weight", "ln_bias"],
            outputs=["ln_out"],
            name= "LayerNormalization",
            epsilon=9.999999747378752e-06,
        ),
        helper.make_node(
            "Attention",
            inputs=[
                "ln_out",
                "attn_qkv_weight",
                "attn_qkv_bias",
                "i_attn_mask",
                "i_kv_cache",
            ],
            outputs=[ "attn_out", "o_kv_cache"],
            name= "Attention",
            domain="com.microsoft",
            num_heads=32,
            unidirectional=1,
            do_rotary=1,
            rotary_embedding=32,
            # past_present_share_buffers=1,
        ),
        helper.make_node(
            "MatMul",
            inputs=[ "attn_out", "attn_out_weight"],
            outputs=[ "matmul_out"],
            name= "OutProj_MatMul",
        ),
        helper.make_node(
            "Add",
            inputs=[ "matmul_out", "attn_out_bias"],
            outputs=[ "add_out"],
            name= "OutProj_Add",
        ),
        helper.make_node(
            "MatMul",
            inputs=[ "ln_out", "mlp_fc1_weight"],
            outputs=[ "fc1_w_out"],
            name= "FC1_MatMul",
        ),
        helper.make_node(
            "Add",
            inputs=[ "fc1_w_out", "mlp_fc1_bias"],
            outputs=[ "fc1_b_out"],
            name= "FC1_Bias",
        ),
        helper.make_node(
            "FastGelu",
            inputs=[ "fc1_b_out"],
            outputs=[ "new_gelu_out"],
            name= "FastGelu",
            domain="com.microsoft",
        ),
        helper.make_node(
            "MatMul",
            inputs=[ "new_gelu_out", "mlp_fc2_weight"],
            outputs=[ "fc2_w_out"],
            name= "FC2_MatMul",
        ),
        helper.make_node(
            "Add",
            inputs=[ "fc2_w_out", "mlp_fc2_bias"],
            outputs=[ "fc2_b_out"],
            name= "FC2_Bias",
        ),
        helper.make_node(
            "Add",
            inputs=["add_out", "fc2_b_out"],
            outputs=["residual_1_out"],
            name= "Residual_Add_1",
        ),
        helper.make_node(
            "Add",
            inputs=["i_hidden_states", "residual_1_out"],
            outputs=["o_hidden_states"],
            name= "Residual_Add_2",
        ),
    ]

    initializers = [
        helper.make_tensor("ln_weight", TensorProto.FLOAT, [hidden_size], ln_weight_t.flatten().tolist()),
        helper.make_tensor("ln_bias", TensorProto.FLOAT, [hidden_size], ln_bias_t.flatten().tolist()),
        helper.make_tensor("attn_qkv_weight", TensorProto.FLOAT, [hidden_size, hidden_size * 3], attn_qkv_weight_t.flatten().tolist()),
        helper.make_tensor("attn_qkv_bias", TensorProto.FLOAT, [hidden_size * 3], attn_qkv_bias_t.flatten().tolist()),
        helper.make_tensor("attn_out_weight", TensorProto.FLOAT, [hidden_size, hidden_size], attn_out_weight_t.flatten().tolist()),
        helper.make_tensor("attn_out_bias", TensorProto.FLOAT, [hidden_size], attn_out_bias_t.flatten().tolist()),
        helper.make_tensor("mlp_fc1_weight", TensorProto.FLOAT, [hidden_size, hidden_size * 4], mlp_fc1_weight_t.flatten().tolist()),
        helper.make_tensor("mlp_fc1_bias", TensorProto.FLOAT, [hidden_size * 4], mlp_fc1_bias_t.flatten().tolist()),
        helper.make_tensor("mlp_fc2_weight", TensorProto.FLOAT, [hidden_size * 4, hidden_size], mlp_fc2_weight_t.flatten().tolist()),
        helper.make_tensor("mlp_fc2_bias", TensorProto.FLOAT, [hidden_size], mlp_fc2_bias_t.flatten().tolist()),
    ]

    graph = helper.make_graph(
        subgraph_nodes,
        "Block_Graph",
        [
            helper.make_tensor_value_info("i_hidden_states", TensorProto.FLOAT, ['batch_size', 'seq_len', hidden_size]),
            helper.make_tensor_value_info("i_attn_mask", TensorProto.INT32, ['batch_size', 'seq_len']),
            helper.make_tensor_value_info("i_kv_cache", TensorProto.FLOAT, [2, 'batch_size', num_heads, 'past_seq_len', head_size]),
        ],
        [
            helper.make_tensor_value_info("o_hidden_states", TensorProto.FLOAT, ['batch_size', 'seq_len', hidden_size]),
            helper.make_tensor_value_info("o_kv_cache", TensorProto.FLOAT, [2, 'batch_size', num_heads, 'total_seq_len', head_size]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()

class ParallelBlock(nn.Module):
    """Parallel block.

    This block applies parallel mixer and MLP layers to the input (used in GPT-J and CodeGen).

    """

    def __init__(
        self,
        config: PretrainedConfig,
        block_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_idx = block_idx

        self.mixer = MHA(config, layer_idx=block_idx)
        self.mlp = MLP(config)

        self.onnx_graph = create_block_graph(
            config.n_head,
            config.head_size,
            self.ln.weight.transpose(0, 1),
            self.ln.bias,
            self.mixer.attn.qkv.weight.reshape(config.n_head, 3, -1).transpose(0, 1).reshape(3 * config.n_embd, -1).transpose(0, 1),
            self.mixer.attn.qkv.bias,
            self.mixer.attn.out_proj.weight.transpose(0, 1),
            self.mixer.attn.out_proj.biasreshape(config.n_head, 3, -1).transpose(0, 1).reshape(-1),
            self.mlp.fc1.weight.transpose(0, 1),
            self.mlp.fc1.bias,
            self.mlp.fc2.weight.transpose(0, 1),
            self.mlp.fc2.bias,
        )

        sess_options = onnxruntime.SessionOptions()
        self.ort_session = onnxruntime.InferenceSession(self.onnx_graph, sess_options, providers=["CUDAExecutionProvider"])


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        attn_outputs = self.mixer(hidden_states, past_key_values=past_key_values, attention_mask=attention_mask)

        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states

    def ort_forward(
        self,
        hidden_states: torch.FloatTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        ort_inputs = {
            "i_hidden_states": hidden_states.cpu().numpy(),
            "i_attn_mask": attention_mask.cpu().numpy(),
            "i_kv_cache": np.zeros([2, 2, 32, 0, 80]).astype(np.float32),
        }

        ort_outs = self.ort_session.run(None, ort_inputs)

        return torch.from_numpy(ort_outs[0])


config = MixFormerSequentialConfig.from_json_file("model/config.json")

block = ParallelBlock(config, block_idx=0)
block.eval()

batch_size = 2
seq_len = 8

kv_mem_dict = {}
kv_mem_dict[0] = torch.zeros([batch_size, 2048, 2, config.n_head, 80]).cuda()
inference_params = InferenceParams(
    max_seqlen=config.n_positions,
    max_batch_size=batch_size,
    seqlen_offset=0,
    batch_size_offset=0,
    key_value_memory_dict=kv_mem_dict,
    lengths_per_sample=None,
)

hidden_states = torch.ones([batch_size, seq_len, config.n_embd]).cuda()
attention_mask = torch.ones([batch_size, seq_len]).cuda()

torch_out = block(
    hidden_states,
    past_key_values=inference_params,
    attention_mask=attention_mask,
)

ort_out = block.ort_forward(
    hidden_states,
    past_key_values=inference_params,
    attention_mask=attention_mask,
)

print(torch_out)
print(ort_out)
