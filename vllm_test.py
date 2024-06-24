from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-1.3b")
outputs = llm.generate(prompts, sampling_params)


# Print the outputs.
for output in outputs:
    prompt = output.prompt

    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer

# # 初始化模型和tokenizer
# model_name = "facebook/opt-1.3b"
# llm = LLM(model=model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 输入和标签
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
# labels = [
#     "Hello, my name is John.",
#     "The president of the United States is Joe Biden.",
#     "The capital of France is Paris.",
#     "The future of AI is bright.",
# ]

# # 编码输入和标签
# encoded_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
# encoded_labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)

# # 确保输入和标签的长度一致
# input_ids = encoded_inputs["input_ids"]
# label_ids = encoded_labels["input_ids"]

# # 创建SamplingParams对象
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# # 使用模型进行推理并计算损失
# outputs = llm(input_ids=input_ids, labels=label_ids, sampling_params=sampling_params)

# # 打印损失
# print(outputs.loss)
