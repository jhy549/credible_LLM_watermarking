from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b")

# lm_tokenizer = AutoTokenizer.from_pretrained('gpt2')
# lm_model = AutoModelForCausalLM.from_pretrained('gpt2')
