from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_8B",trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B",trust_remote_code=True)



lm_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
lm_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
