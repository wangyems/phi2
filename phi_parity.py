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

attn_choice = "gqa" # choose from attn, mha, gqa
# attn is all good, mha needs to comment rotary and causal, gqa need to comment rotary

def create_block_graph_attn(
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
            rotary_embedding_dim=32,
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
            helper.make_tensor_value_info("ln_out", TensorProto.FLOAT, ['batch_size', 'seq_len', hidden_size]),
            helper.make_tensor_value_info("attn_out", TensorProto.FLOAT, ['batch_size', 'seq_len', hidden_size]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()

def create_block_graph_mha(
    num_heads,
    head_size,
    ln_weight_t,
    ln_bias_t,
    attn_q_weight_t,
    attn_k_weight_t,
    attn_v_weight_t,
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
            "MatMul",
            inputs=[ "ln_out", "attn_q_weight"],
            outputs=[ "query"],
            name= "Q_MatMul",
        ),
        helper.make_node(
            "MatMul",
            inputs=[ "ln_out", "attn_k_weight"],
            outputs=[ "key"],
            name= "K_MatMul",
        ),
        helper.make_node(
            "MatMul",
            inputs=[ "ln_out", "attn_v_weight"],
            outputs=[ "value"],
            name= "V_MatMul",
        ),
        helper.make_node(
            "MultiHeadAttention",
            inputs=[
                "query",
                "key",
                "value",
                "attn_qkv_bias",
                "i_attn_mask",
                "",
                "past_key",
                "past_value",
            ],
            outputs=[ "attn_out", "present_key", "present_value"],
            name= "MultiHeadAttention_0",
            domain="com.microsoft",
            num_heads=32, 
            # Need to support causal attention if using this op in cpu
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
        helper.make_tensor("attn_q_weight", TensorProto.FLOAT, [hidden_size, hidden_size], attn_q_weight_t.flatten().tolist()),
        helper.make_tensor("attn_k_weight", TensorProto.FLOAT, [hidden_size, hidden_size], attn_k_weight_t.flatten().tolist()),
        helper.make_tensor("attn_v_weight", TensorProto.FLOAT, [hidden_size, hidden_size], attn_v_weight_t.flatten().tolist()),
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
            helper.make_tensor_value_info("past_key", TensorProto.FLOAT, ['batch_size', num_heads, 'past_seq_len', head_size]),
            helper.make_tensor_value_info("past_value", TensorProto.FLOAT, ['batch_size', num_heads, 'past_seq_len', head_size]),
        ],
        [
            helper.make_tensor_value_info("o_hidden_states", TensorProto.FLOAT, ['batch_size', 'seq_len', hidden_size]),
            helper.make_tensor_value_info("present_key", TensorProto.FLOAT, ['batch_size', num_heads, 'total_seq_len', head_size]),
            helper.make_tensor_value_info("present_value", TensorProto.FLOAT, ['batch_size', num_heads, 'total_seq_len', head_size]),
            helper.make_tensor_value_info("ln_out", TensorProto.FLOAT, ['batch_size', 'seq_len', hidden_size]),
            helper.make_tensor_value_info("attn_out", TensorProto.FLOAT, ['batch_size', 'seq_len', hidden_size]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()

def create_block_graph_gqa(
    num_heads,
    head_size,
    ln_weight_t,
    ln_bias_t,
    attn_q_weight_t,
    attn_k_weight_t,
    attn_v_weight_t,
    attn_q_bias_t,
    attn_k_bias_t,
    attn_v_bias_t,
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
            "MatMul",
            inputs=[ "ln_out", "attn_q_weight"],
            outputs=[ "q_matmul_out"],
            name= "Q_MatMul",
        ),
        helper.make_node(
            "MatMul",
            inputs=[ "ln_out", "attn_k_weight"],
            outputs=[ "k_matmul_out"],
            name= "K_MatMul",
        ),
        helper.make_node(
            "MatMul",
            inputs=[ "ln_out", "attn_v_weight"],
            outputs=[ "v_matmul_out"],
            name= "V_MatMul",
        ),
        helper.make_node(
            "Add",
            inputs=[ "q_matmul_out", "attn_q_bias"],
            outputs=[ "query"],
            name= "Q_Bias",
        ),
        helper.make_node(
            "Add",
            inputs=[ "k_matmul_out", "attn_k_bias"],
            outputs=[ "key"],
            name= "K_Bias",
        ),
        helper.make_node(
            "Add",
            inputs=[ "v_matmul_out", "attn_v_bias"],
            outputs=[ "value"],
            name= "V_Bias",
        ),
        helper.make_node(
            "GroupQueryAttention",
            inputs=[
                "query",
                "key",
                "value",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_sequence_length",
            ],
            outputs=[ "attn_out", "present_key", "present_value"],
            name= "GroupQueryAttention_0",
            domain="com.microsoft",
            num_heads=32, 
            kv_num_heads=32,
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

    tensor_type = TensorProto.FLOAT16
    initializers = [
        helper.make_tensor("ln_weight", tensor_type, [hidden_size], ln_weight_t.flatten().tolist()),
        helper.make_tensor("ln_bias", tensor_type, [hidden_size], ln_bias_t.flatten().tolist()),
        helper.make_tensor("attn_q_weight", tensor_type, [hidden_size, hidden_size], attn_q_weight_t.flatten().tolist()),
        helper.make_tensor("attn_k_weight", tensor_type, [hidden_size, hidden_size], attn_k_weight_t.flatten().tolist()),
        helper.make_tensor("attn_v_weight", tensor_type, [hidden_size, hidden_size], attn_v_weight_t.flatten().tolist()),
        helper.make_tensor("attn_q_bias", tensor_type, [hidden_size], attn_q_bias_t.flatten().tolist()),
        helper.make_tensor("attn_k_bias", tensor_type, [hidden_size], attn_k_bias_t.flatten().tolist()),
        helper.make_tensor("attn_v_bias", tensor_type, [hidden_size], attn_v_bias_t.flatten().tolist()),
        helper.make_tensor("attn_out_weight", tensor_type, [hidden_size, hidden_size], attn_out_weight_t.flatten().tolist()),
        helper.make_tensor("attn_out_bias", tensor_type, [hidden_size], attn_out_bias_t.flatten().tolist()),
        helper.make_tensor("mlp_fc1_weight", tensor_type, [hidden_size, hidden_size * 4], mlp_fc1_weight_t.flatten().tolist()),
        helper.make_tensor("mlp_fc1_bias", tensor_type, [hidden_size * 4], mlp_fc1_bias_t.flatten().tolist()),
        helper.make_tensor("mlp_fc2_weight", tensor_type, [hidden_size * 4, hidden_size], mlp_fc2_weight_t.flatten().tolist()),
        helper.make_tensor("mlp_fc2_bias", tensor_type, [hidden_size], mlp_fc2_bias_t.flatten().tolist()),
    ]

    graph = helper.make_graph(
        subgraph_nodes,
        "Block_Graph",
        [
            helper.make_tensor_value_info("i_hidden_states", tensor_type, ['batch_size', 'seq_len', hidden_size]),
            helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, ['batch_size']),
            helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []),
            helper.make_tensor_value_info("past_key", tensor_type, ['batch_size', num_heads, 'past_seq_len', head_size]),
            helper.make_tensor_value_info("past_value", tensor_type, ['batch_size', num_heads, 'past_seq_len', head_size]),
        ],
        [
            helper.make_tensor_value_info("o_hidden_states", tensor_type, ['batch_size', 'seq_len', hidden_size]),
            helper.make_tensor_value_info("present_key", tensor_type, ['batch_size', num_heads, 'total_seq_len', head_size]),
            helper.make_tensor_value_info("present_value", tensor_type, ['batch_size', num_heads, 'total_seq_len', head_size]),
            helper.make_tensor_value_info("ln_out", tensor_type, ['batch_size', 'seq_len', hidden_size]),
            helper.make_tensor_value_info("attn_out", tensor_type, ['batch_size', 'seq_len', hidden_size]),
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

        if attn_choice == "attn":
            self.onnx_graph = create_block_graph_attn(
                config.n_head,
                80, # head_size
                self.ln.weight,
                self.ln.bias,
                #self.mixer.Wqkv.weight.reshape(config.n_head, 3, -1).transpose(0, 1).reshape(3 * config.n_embd, -1).transpose(0, 1),
                #self.mixer.Wqkv.bias.reshape(config.n_head, 3, -1).transpose(0, 1).reshape(-1),
                self.mixer.Wqkv.weight.transpose(0, 1),
                self.mixer.Wqkv.bias,
                self.mixer.out_proj.weight.transpose(0, 1),
                self.mixer.out_proj.bias,
                self.mlp.fc1.weight.transpose(0, 1),
                self.mlp.fc1.bias,
                self.mlp.fc2.weight.transpose(0, 1),
                self.mlp.fc2.bias,
            )
        elif attn_choice == "mha":
            self.onnx_graph = create_block_graph_mha(
                config.n_head,
                80, # head_size
                self.ln.weight,
                self.ln.bias,
                torch.split(self.mixer.Wqkv.weight, 2560)[0].transpose(0, 1),
                torch.split(self.mixer.Wqkv.weight, 2560)[1].transpose(0, 1),
                torch.split(self.mixer.Wqkv.weight, 2560)[2].transpose(0, 1),
                self.mixer.Wqkv.bias,
                self.mixer.out_proj.weight.transpose(0, 1),
                self.mixer.out_proj.bias,
                self.mlp.fc1.weight.transpose(0, 1),
                self.mlp.fc1.bias,
                self.mlp.fc2.weight.transpose(0, 1),
                self.mlp.fc2.bias,
            )
        elif attn_choice == "gqa":
            self.onnx_graph = create_block_graph_gqa(
                config.n_head,
                80, # head_size
                self.ln.weight.to(torch.float16),
                self.ln.bias.to(torch.float16),
                torch.split(self.mixer.Wqkv.weight, 2560)[0].transpose(0, 1).to(torch.float16),
                torch.split(self.mixer.Wqkv.weight, 2560)[1].transpose(0, 1).to(torch.float16),
                torch.split(self.mixer.Wqkv.weight, 2560)[2].transpose(0, 1).to(torch.float16),
                torch.split(self.mixer.Wqkv.bias, 2560)[0].to(torch.float16),
                torch.split(self.mixer.Wqkv.bias, 2560)[1].to(torch.float16),
                torch.split(self.mixer.Wqkv.bias, 2560)[2].to(torch.float16),
                self.mixer.out_proj.weight.transpose(0, 1).to(torch.float16),
                self.mixer.out_proj.bias.to(torch.float16),
                self.mlp.fc1.weight.transpose(0, 1).to(torch.float16),
                self.mlp.fc1.bias.to(torch.float16),
                self.mlp.fc2.weight.transpose(0, 1).to(torch.float16),
                self.mlp.fc2.bias.to(torch.float16),
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
    ):
        batch_size, seq_len, _ = hidden_states.shape

        if attn_choice == "attn":
            ort_inputs = {
                "i_hidden_states": hidden_states.cpu().numpy(),
                "i_attn_mask": np.ones([batch_size, seq_len]).astype(np.int32),
                "i_kv_cache": np.zeros([2, batch_size, 32, 0, 80]).astype(np.float32),
            }
        elif attn_choice == "mha":
            ort_inputs = {
                "i_hidden_states": hidden_states.cpu().numpy(),
                "i_attn_mask": np.ones([batch_size, seq_len]).astype(np.int32),
                "past_key": np.zeros([batch_size, 32, 0, 80]).astype(np.float32),
                "past_value": np.zeros([batch_size, 32, 0, 80]).astype(np.float32),
            }
        elif attn_choice == "gqa":
            ort_inputs = {
                "i_hidden_states": hidden_states.cpu().numpy().astype(np.float16),
                "seqlens_k": seq_len * np.ones([batch_size]).astype(np.int32),
                "total_sequence_length": np.array(seq_len).astype(np.int32),
                "past_key": np.zeros([batch_size, 32, 0, 80]).astype(np.float16),
                "past_value": np.zeros([batch_size, 32, 0, 80]).astype(np.float16),
            }

        ort_outs = self.ort_session.run(None, ort_inputs)

        return ort_outs


config = MixFormerSequentialConfig.from_json_file("model/config.json")

block = ParallelBlock(config, block_idx=0)
block.eval()

batch_size = 1
seq_len = 8

kv_mem_dict = {}
kv_mem_dict[0] = torch.zeros([batch_size, 128, 2, config.n_head, 80])
inference_params = InferenceParams(
    max_seqlen=128, #config.n_positions,
    max_batch_size=batch_size,
    seqlen_offset=0,
    batch_size_offset=0,
    key_value_memory_dict=kv_mem_dict,
    lengths_per_sample=None,
)

hidden_states = torch.randn([batch_size, seq_len, config.n_embd])
attention_mask = torch.ones([batch_size, seq_len])

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

print("torch: output:", torch_out)
# print("ort: ln_out", torch.tensor(ort_out[2]))
# print("ort: attn_out", torch.tensor(ort_out[3]))
print("ort: output", torch.tensor(ort_out[0]))
print("parity:", torch.allclose(torch_out, torch.tensor(ort_out[0]).to(torch.float), atol=3e-3))
