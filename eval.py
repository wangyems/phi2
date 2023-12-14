import torch
from typing import List
from itertools import chain

from model.configuration_mixformer_sequential import MixFormerSequentialConfig
from model.modeling_mixformer_sequential import MixFormerSequentialForCausalLM, InferenceParams

from transformers import CodeGenTokenizer
from llama_inputs import get_merged_sample_with_past_kv_inputs


# torch.set_default_device("cuda")
# base_path = "/wy/onnx_models/phi2/mlflow_model_folder/data/"
# config = MixFormerSequentialConfig.from_json_file(base_path + "model/config.json")
# tokenizer = CodeGenTokenizer.from_pretrained(base_path + "tokenizer")
# model = MixFormerSequentialForCausalLM(config)
# model.load_state_dict(torch.load(base_path + "model/pytorch_model.bin"))
# model.eval()

# inputs = tokenizer('''```python
# def print_prime(n):
#    """
#    Print all primes between 1 and n
#    """''', return_tensors="pt", return_attention_mask=False)
# outputs = model.generate(**inputs, max_length=1)
# text = tokenizer.batch_decode(outputs)[0]
# print(text)



