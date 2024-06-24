from transformers import LlamaTokenizer, AutoConfig, AutoModelForCausalLM
import os
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# model_name = 'llama-7b'
model_name = 'huggyllama/llama-13b'

# 首先使用虚拟内存加载模型，这个内存被认为是无限大
# "/{}/".format(model_name)是你自己模型存放的路径，这个不会改的也别来部署大模型了
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b",
                                                 trust_remote_code=True, torch_dtype=torch.float16)

# The model weights are not tied.
# Please use the `tie_weights` method before using the `infer_auto_device` function.
model.tie_weights()
# 人为设置不同的device所分配的内存， devide:0分配得少一些方便后续推理，实际操作中不写也行
max_memory={0: "16GiB", 1: "23GiB", 2: "23GiB", 3:"23GiB", "cpu":"80Gib"}  # cpu辛苦你啦
device_map = infer_auto_device_map(model, max_memory=max_memory)    # 自动推断device_map
#
# print(device_map)  # 可以看看模型不同层被分在了哪里

model = load_checkpoint_and_dispatch(model, checkpoint="/ailab/user/jianghaoyu/.cache/huggingface/hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba",
                    device_map=device_map, offload_folder="offload", no_split_module_classes=["LlamaDecoderLayer"],
                    offload_state_dict=True, dtype=torch.float16).half()
        # half()强制数据转化，看名字应该是一个内存优化的数据类型

print("(*^_^*) model load finished!!!! please play with {}".format(model_name))


while True:
    text_in = input('please input your question: ')
    # 注意，使用Autokenizer会很慢，不知道为什么
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-13b")

    inputs = tokenizer(text_in, return_tensors="pt").to(0)

    outputs = model.generate(inputs["input_ids"], max_new_tokens=20)
    text_out = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)

    print(text_out)
