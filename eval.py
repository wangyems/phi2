import torch
from typing import List
from itertools import chain

from model.configuration_mixformer_sequential import MixFormerSequentialConfig
from model.modeling_mixformer_sequential import MixFormerSequentialForCausalLM, InferenceParams

from transformers import CodeGenTokenizer
from llama_inputs import get_merged_sample_with_past_kv_inputs

class MyPhi2(MixFormerSequentialForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    # @staticmethod
    # def post_process(result, num_layer):
    #     if isinstance(result[1][0], (tuple, list)):
    #         assert len(result[1]) == num_layer and len(result[1][0]) == 2
    #         # assert len(result[1][0][0].shape) == 4 and result[1][0][0].shape == result[1][0][1].shape
    #         present = []
    #         for i in range(num_layer):
    #             # Since transformers v4.*, past key and values are separated outputs.
    #             # Here we concate them into one tensor to be compatible with Attention operator.
    #             present.append(
    #                 torch.cat(
    #                     (result[1][i][0].unsqueeze(0), result[1][i][1].unsqueeze(0)),
    #                     dim=0,
    #                 )
    #             )
    #         return (result[0], tuple(present))

    def forward(self, input_ids, attention_mask, *past_key_values):
        kv_mem_dict = {}
        for i, kv in enumerate(past_key_values[0]):
            print(i, kv.shape)
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

        return result


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

    # Avoid using system temp dir to avoid overflood on hard disk as 70b model is very large.
    # Use temp folder per rank to avoid race condition here.
torch.onnx.export(
    model,
    args=decoder_merged_inputs,
    f="phi2.onnx",
    export_params=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=13,
    do_constant_folding=True,
    verbose=True,
)





