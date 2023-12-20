import torch
import onnx
from typing import List
from itertools import chain
import os

from model.configuration_mixformer_sequential import MixFormerSequentialConfig
from model.modeling_mixformer_sequential import MixFormerSequentialForCausalLM, InferenceParams
from model.modeling_mixformer_sequential import MHA, RotaryEmbedding
from transformers import CodeGenTokenizer
from llama_inputs import get_merged_sample_with_past_kv_inputs

# device_id = 5
# os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
# os.environ["ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION"] = "1"
# os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
# os.environ["ORT_DISABLE_TRT_FLASH_ATTENTION"] = "1"
# os.environ["ORT_DISABLE_FUSED_ATTENTION"] = "1"
# os.environ["ORT_DISABLE_FUSED_CROSS_ATTENTION"] = "1"
os.environ["ORT_ENABLE_FUSED_CAUSAL_ATTENTION"] = "1"

class MyPhi2(MixFormerSequentialForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def post_process(result, num_layer):
        present = []
        for i in range(num_layer):
            present.append(
                result.past_key_values.key_value_memory_dict[i]
            )
        return (result.logits, tuple(present))

    def forward(self, input_ids, attention_mask, *past_key_values):
        kv_mem_dict = {}
        for i, kv in enumerate(past_key_values[0]):
            kv_mem_dict[i] = kv

        inference_params = InferenceParams(
            max_seqlen=self.config.n_positions,
            max_batch_size=input_ids.shape[0],
            seqlen_offset=0,
            batch_size_offset=0,
            key_value_memory_dict=kv_mem_dict,
            lengths_per_sample=None,
        )

        result = super().forward(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=inference_params,
        )

        return MyPhi2.post_process(result, self.config.num_hidden_layers)


def get_merged_model_dynamic_axes(input_names: List[str], output_names: List[str]):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"input_ids", "position_ids"}:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "attention_mask":
            # shape is (batch_size, past_sequence_length + sequence_length) = (batch_size, total_sequence_length)
            # for prompt generation, past_sequence_length = 0
            # for token generation, sequence_length = 1
            dynamic_axes[name] = {0: "batch_size", 1: "total_sequence_length"}
        elif "past" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 1: "max_sequence_length"}
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif "present" in name:
            # shape is (batch_size, num_heads, past_sequence_length + sequence_length, head_size) = (batch_size, num_heads, total_sequence_length, head_size)
            # for prompt generation, past_sequence_length = 0
            # for token generation, sequence_length = 1
            dynamic_axes[name] = {0: "batch_size", 1: "max_sequence_length"}
        else:
            raise Exception("Unknown input or output name found")
    return dynamic_axes

torch.set_default_device("cuda")

config = MixFormerSequentialConfig.from_json_file("model/config.json")
tokenizer = CodeGenTokenizer.from_pretrained("tokenizer")
model = MyPhi2(config)
model.load_state_dict(torch.load("model/pytorch_model.bin"))
model.eval()

# inputs = tokenizer('''```python
# def print_prime(n):
#    """
#    Print all primes between 1 and n
#    """''', return_tensors="pt", return_attention_mask=False)
# print(inputs)
# outputs = model.generate(**inputs, max_length=1)
# text = tokenizer.batch_decode(outputs)[0]
# print(text)

batch_size, sequence_length, past_sequence_length = 2, 8, 0

max_sequence_length = 512

# Export decoder_merged_model.onnx
decoder_merged_inputs = get_merged_sample_with_past_kv_inputs(
    config,
    model.device,
    batch_size,
    sequence_length,
    past_sequence_length,
    max_seq_len=max_sequence_length,
    use_fp16=False,
    world_size=1,
)

# torch pass
decoder_out = model(decoder_merged_inputs[0], decoder_merged_inputs[1], decoder_merged_inputs[2])


# ort run
if False:
    from onnxruntime import InferenceSession, SessionOptions
    import numpy as np
    sess_options = SessionOptions()
    model_path = "/wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp16_opt.onnx"
    #model_path = "/yufeng_data/models/phi2/mlflow_model_folder/data/onnx_models/phi-2_decoder.onnx"
    #model_path = "/wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp16_opt.onnx"
    ep = ("CUDAExecutionProvider")
    ort_session = InferenceSession(model_path, sess_options, providers=[ep])
    # convert decoder_merged_inputs to ort_inputs
    ort_inputs = {}
    for i, name in enumerate(ort_session.get_inputs()):
        if i == 0:
            ort_inputs[name.name] = decoder_merged_inputs[i].cpu().numpy().astype(np.int32)
            #print(ort_inputs[name.name].shape)
        elif i == 1:
            ort_inputs[name.name] = np.ones([batch_size, 8]).astype(np.int32)
            #print(ort_inputs[name.name].shape)
        else:
            #ort_inputs[name.name] = np.transpose(decoder_merged_inputs[2][i - 2].detach().cpu().numpy(), (2, 0, 3, 1, 4))
            ort_inputs[name.name] = np.zeros([2, batch_size, 32, 0, 80]).astype(np.float16)
            #print(ort_inputs[name.name].shape)
    ort_outs = ort_session.run(None, ort_inputs)
    print("torch results")
    print(decoder_out[0].detach().cpu().numpy())
    print("ort results")
    for i in range(len(ort_outs)):
        print("ort output:", i, ort_outs[i].shape, ort_outs[i][0][0][:16])

    #print(ort_outs[0] - decoder_out[0].detach().cpu().numpy())
    #print(np.allclose(decoder_out[0].detach().cpu().numpy(), ort_outs[0], atol=1e-2))

if True:
    use_dynamo = True
    if use_dynamo:
        from torch._dynamo import config
        config.capture_scalar_outputs = True
        temp_path = "phi-2_decoder_fp32.onnx"
        input_ids = decoder_merged_inputs[0]
        input_ids = input_ids.expand(2, -1)
        torch._dynamo.mark_dynamic(input_ids, 0)
        attn_mask = decoder_merged_inputs[1]
        attn_mask = attn_mask.expand(2, -1)
        torch._dynamo.mark_dynamic(attn_mask, 0)
        for i in range(len(decoder_merged_inputs[2])):
            kv_cache = decoder_merged_inputs[2][i]
            decoder_merged_inputs[2][i] = kv_cache.expand(2, -1, -1, -1, -1)
            torch._dynamo.mark_dynamic(decoder_merged_inputs[2][i], 0)

        torch.onnx.dynamo_export(
            model, input_ids, attn_mask, decoder_merged_inputs[2], export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
        ).save(temp_path)
        onnx.checker.check_model(temp_path)
        onnx.shape_inference.infer_shapes_path(temp_path)
    else:
        input_names = [
            "input_ids",
            "attention_mask",
            *list(
                chain.from_iterable(
                    (f"past_key_values.{i}",) for i in range(config.num_hidden_layers)
                )
            ),
        ]
        output_names = [
            "logits",
            *list(
                chain.from_iterable((f"present_key_values.{i}",) for i in range(config.num_hidden_layers))
            ),
        ]
        dynamic_axes = get_merged_model_dynamic_axes(input_names, output_names)
        print(dynamic_axes)
        torch.onnx.export(
            model,
            args=decoder_merged_inputs,
            f="phi2-jit.onnx",
            export_params=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            verbose=True,
        )
