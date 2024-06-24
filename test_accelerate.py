import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from accelerate import dispatch_model,load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory, infer_auto_device_map

model_name = "huggyllama/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,device_map="auto")
# print(f"Model local path: {model.config.name_or_path}")
# print(f"✨记录模型包含的模块，及所在设备")
 
# with open("model_no_split.txt", "w") as f:
#     for (name, module), (name, param) in zip(model.named_modules(),model.named_parameters()):
#         f.write(name+" | ")
#         f.write(str(type(module))+" | ")
#         f.write(str(param.device))
#         f.write("\n")

# # no_split_module_classes = [
# #     "Linear",
# #     "LlamaDecoderLayer","LlamaSdpaAttention","LlamaMLP","LlamaRotaryEmbedding",
# #     "LlamaRMSNorm","SiLU"
# # ]
# no_split_module_classes = LlamaForCausalLM._no_split_modules
# max_memory = {1: '12GiB', 2: '12GiB', 3: '12GiB', 4: '12GiB'}
# device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
# print(device_map)
# # model = dispatch_model(model, device_map=device_map, offload_dir="tmp")
# model = load_checkpoint_and_dispatch(model, "/ailab/user/jianghaoyu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16", device_map="auto",no_split_module_classes=no_split_module_classes),

# time.sleep(2300)
# 生成文本
input_text = "def function("
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)
print(model.device)

# 生成文本
outputs = model.generate(input_ids, max_length=100)

# 将生成的文本解码为人类可读的文本
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
print(generated_texts)
