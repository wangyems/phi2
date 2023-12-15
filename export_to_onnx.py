import torch
import onnx
from typing import List
from itertools import chain

from model.configuration_mixformer_sequential import MixFormerSequentialConfig
from model.modeling_mixformer_sequential import MixFormerSequentialForCausalLM, InferenceParams
from model.modeling_mixformer_sequential import MHA, RotaryEmbedding
from transformers import CodeGenTokenizer
from llama_inputs import get_merged_sample_with_past_kv_inputs

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
#model.load_state_dict(torch.load("model/pytorch_model.bin"))
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

batch_size, sequence_length, past_sequence_length = 2, 4096, 0

max_sequence_length = 2048

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
    model_path = "/wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_small_opt.onnx"
    ort_session = InferenceSession(model_path, sess_options, providers=["CUDAExecutionProvider"])
    # convert decoder_merged_inputs to ort_inputs
    ort_inputs = {}
    for i, name in enumerate(ort_session.get_inputs()):
        if i == 0:
            ort_inputs[name.name] = decoder_merged_inputs[i].cpu().numpy()
            print(ort_inputs[name.name].shape)
        elif i == 1:
            ort_inputs[name.name] = np.ones([2, 2048]).astype(np.int64)
            print(ort_inputs[name.name].shape)
        else:
            ort_inputs[name.name] = np.transpose(decoder_merged_inputs[2][i - 2].detach().cpu().numpy(), (2, 0, 3, 1, 4))
            print(ort_inputs[name.name].shape)
    ort_outs = ort_session.run(None, ort_inputs)

use_dynamo = True

if True:
    if use_dynamo:
        from torch._dynamo import config
        config.capture_scalar_outputs = True
        temp_path = "phi-2_decoder_small.onnx"
        torch.onnx.dynamo_export(
            model, decoder_merged_inputs[0], decoder_merged_inputs[1], decoder_merged_inputs[2], export_options=torch.onnx.ExportOptions(dynamic_shapes=True)
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
