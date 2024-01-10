import torch

from transformers import CodeGenTokenizer
import numpy as np
import onnxruntime as ort
from typing import List
import time
import os
#os.environ["ORT_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["ORT_ENABLE_FUSED_CAUSAL_ATTENTION"] = "1"

input_ids_name = "input_ids"
attention_mask_name = "attention_mask"
past_seq_len_name = "past_sequence_length"

logits_name = "logits"


def get_initial_inputs_and_outputs(tokenizer: CodeGenTokenizer, prompt: List[str], device: torch.device, use_fp16: bool, use_buffer_share: bool):
    tokenizer.pad_token = "[PAD]"
    encodings_dict = tokenizer.batch_encode_plus(prompt, padding=True)
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    input_ids = torch.tensor(encodings_dict["input_ids"], device=device, dtype=torch.int32)
    attention_mask = torch.tensor(encodings_dict["attention_mask"], device=device, dtype=torch.int32)
    print(input_ids, attention_mask)
    #past_seq_len = torch.tensor(0, device=torch.device("cpu"), dtype=torch.int32)

    inputs = {
        input_ids_name: input_ids.contiguous(),
        attention_mask_name: attention_mask.contiguous(),
        #past_seq_len_name: past_seq_len.contiguous()
    }

    batch_size, sequence_length = input_ids.shape

    max_sequence_length = 128
    num_heads, head_size = 32, 80
    for i in range(32):
        past_key_value = torch.zeros(2, batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        inputs.update({
            f"past_{i}": past_key_value.contiguous(),
        })

    logits = torch.zeros(batch_size, sequence_length, 51200, device=device, dtype=torch_dtype)
    outputs = {
        logits_name: logits.contiguous()
    }
    if not use_buffer_share:
        for i in range(32):
            present_key_value = torch.zeros(2, batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            outputs.update({
                f"present_{i}": present_key_value.contiguous(),
            })

    return inputs, outputs


def get_initial_inputs_and_outputs_for_bench(batch_size, sequence_length, device: torch.device, use_fp16: bool, use_buffer_share: bool):
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    input_ids = torch.randint(0, 51200, (batch_size, sequence_length), device=device, dtype=torch.int32)
    attention_mask = torch.ones(batch_size, sequence_length, device=device, dtype=torch.int32)

    inputs = {
        input_ids_name: input_ids.contiguous(),
        attention_mask_name: attention_mask.contiguous(),
        #past_seq_len_name: past_seq_len.contiguous()
    }

    batch_size, sequence_length = input_ids.shape

    max_sequence_length = 128
    num_heads, head_size = 32, 80
    for i in range(32):
        past_key_value = torch.zeros(2, batch_size, num_heads, max_sequence_length if use_buffer_share else 0, head_size, device=device, dtype=torch_dtype)
        inputs.update({
            f"past_{i}": past_key_value.contiguous(),
        })

    logits = torch.zeros(batch_size, sequence_length, 51200, device=device, dtype=torch_dtype)
    outputs = {
        logits_name: logits.contiguous()
    }
    if not use_buffer_share:
        for i in range(32):
            present_key_value = torch.zeros(2, batch_size, num_heads, sequence_length, head_size, device=device, dtype=torch_dtype)
            outputs.update({
                f"present_{i}": present_key_value.contiguous(),
            })

    return inputs, outputs


def apply_io_binding(model: ort.InferenceSession, inputs: dict, outputs: dict, use_fp16: bool, use_buffer_share: bool):
    # Check that all model inputs will be provided
    model_inputs = set(map(lambda model_input: model_input.name, model.get_inputs()))
    user_inputs = set(inputs.keys())
    missing_inputs = model_inputs - user_inputs
    if len(missing_inputs):
        print(f"The following model inputs are missing: {missing_inputs}")
        raise Exception("There are missing inputs to the model. Please add them and try again.")

    # Remove unnecessary inputs from model inputs
    unnecessary_inputs = user_inputs - model_inputs
    if len(unnecessary_inputs):
        for unnecessary_input in unnecessary_inputs:
            print(f"Removing unnecessary input '{unnecessary_input}' from user provided inputs")
            del inputs[unnecessary_input]

    # Bind inputs/outputs to IO binding
    io_binding = model.io_binding()
    device = None

    for k, v in inputs.items():
        io_binding.bind_input(
            name=k,
            device_type=v.device.type,
            device_id=0 if v.device.type == "cpu" else v.device.index,
            element_type=pt_to_np[repr(v.dtype)],
            shape=tuple(v.shape),
            buffer_ptr=v.data_ptr()
        )
        device = v.device

    for output in model.get_outputs():
        name = output.name
        if use_buffer_share and "present" in name:
            # Bind KV cache outputs to KV cache inputs
            v = inputs[name.replace("present", "past")]
            io_binding.bind_output(
                name=name,
                device_type=v.device.type,
                device_id=v.device.index,
                element_type=np.float16,
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )
        else:
            v = outputs[name]
            io_binding.bind_output(
                name=name,
                device_type=device.type,
                device_id=0 if device.type == "cpu" else device.index,
                element_type=(np.float16 if use_fp16 else np.float32),
                shape=tuple(v.shape),
                buffer_ptr=v.data_ptr()
            )

    return io_binding

def main():
    # User settings
    onnx_model_path, use_fp16, use_buffer_share = "/wy/onnx_models/phi2/mlflow_model_folder/data/phi-2_decoder_fp16_opt.onnx", True, False

    prompt, max_length = ['''```python
    def print_prime(n):
    """
    Print all primes between 1 and n
    """''', "use OnnxRuntime to run model on device"], 128

    # Get information based on user settings
    device_id = 5
    device = torch.device(f"cuda:{device_id}")
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    base_path = "/wy/onnx_models/phi2/mlflow_model_folder/data/"
    #config = MixFormerSequentialConfig.from_json_file(base_path + "model/config.json")
    tokenizer = CodeGenTokenizer.from_pretrained(base_path + "tokenizer")

    sess_options = ort.SessionOptions()
    # sess_options.log_verbosity_level = 1
    # sess_options.log_severity_level = 1
    print("creating session")
    ep = ("CUDAExecutionProvider", {"device_id": device_id, "enable_skip_layer_norm_strict_mode": True}) if device.type == "cuda" else "CPUExecutionProvider"
    model = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=[ep])
    print("done")

    for token_num in [32, 128]:
        for batch_size in [1, 4, 8, 16]:
            for seq_len in [64, 128, 256, 512, 1024, 2048, 4096]:
                if batch_size == 8 and seq_len == 4096:
                    continue
                if batch_size == 16 and seq_len >= 2048:
                    continue
                max_length = seq_len + token_num
                print("batch size:", batch_size, "seq len:", seq_len, "token_num:", token_num)
                # Get model and its initial inputs/outputs
                #inputs, outputs = get_initial_inputs_and_outputs(tokenizer, prompt, device, use_fp16, use_buffer_share)
                inputs, outputs = get_initial_inputs_and_outputs_for_bench(batch_size, seq_len, device, use_fp16, use_buffer_share)

                all_token_ids = inputs[input_ids_name].clone()
                batch_size, sequence_length = all_token_ids.shape

                current_length = sequence_length
                has_eos = torch.zeros(batch_size, device=device, dtype=torch.bool)

                count = 0
                start = time.time()
                while current_length < max_length:
                    # Run inference
                    # print("Input ids:", inputs["input_ids"])
                    # print("Attention mask:", inputs["attention_mask"])
                    # print("Position ids:", inputs["position_ids"])
                    if count == 1:
                        prompt_fence = time.time()

                    io_binding = apply_io_binding(model, inputs, outputs, use_fp16, use_buffer_share)
                    io_binding.synchronize_inputs()
                    model.run_with_iobinding(io_binding)
                    io_binding.synchronize_outputs()

                    # Sample with argmax (greedy search)
                    next_token_logits = outputs[logits_name][:, -1, :]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                    # Check if we previously reached EOS token id or if generated token id is EOS token id
                    has_eos = has_eos | next_tokens == tokenizer.eos_token_id

                    # Determine which new tokens to add to list of all token ids
                    # Add EOS token ids for batch entries that ended early (ragged batching scenario where some batch entries ended early and some haven't)
                    tokens_to_add = next_tokens.masked_fill(has_eos, tokenizer.eos_token_id).reshape([batch_size, 1])
                    all_token_ids = torch.cat([all_token_ids, tokens_to_add], dim=-1)

                    # Return early if all batch entries have reached EOS token id
                    if torch.all(has_eos):
                        final_fence = time.time()
                        print(f"Time for prompt: {1000 * (prompt_fence - start)}ms", f"Time for token: {1000 * (final_fence - prompt_fence) / count}ms")
                        break

                    # Update inputs for next inference run
                    #inputs["past_sequence_length"] = torch.tensor(current_length, device=torch.device("cpu"), dtype=torch.int32)
                    inputs["input_ids"] = tokens_to_add.to(torch.int32)
                    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], (~has_eos).to(torch.int32).reshape(batch_size, 1)], 1)
                    current_length += 1
                    count += 1

                    if current_length == max_length:
                        final_fence = time.time()
                        print(f"Time for prompt: {1000 * (prompt_fence - start)}ms", f"Time for token: {1000 * (final_fence - prompt_fence) / count}ms")
                        break

                    # Set logits to zeros for next inference run and re-use memory buffer
                    if outputs[logits_name].shape[1] != 1:
                        outputs[logits_name] = outputs[logits_name][:, :1, :].contiguous()
                    outputs[logits_name].zero_()

                    if not use_buffer_share:
                        for i in range(32):
                            inputs[f"past_{i}"] = outputs[f"present_{i}"]

                        new_sequence_length = inputs["attention_mask"].shape[1]
                        for i in range(32):
                            present_key_value = torch.zeros(2, batch_size, 32, new_sequence_length, 80, device=device, dtype=torch_dtype)
                            outputs.update({
                                f"present_{i}": present_key_value.contiguous(),
                            })

                # Batch decoding at end of generation (another option instead of iterative decoding)
                #texts = tokenizer.batch_decode(all_token_ids, skip_special_tokens=True)
                # for text in texts:
                #     print(text)
                #     print("-----------------------")

if __name__ == "__main__":
    pt_to_np = {
        "torch.int32": np.int32,
        "torch.float32": np.float32,
        "torch.float16": np.float16
    }
    main()