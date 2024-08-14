# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaForMaskedLM, RobertaTokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

model = RobertaForMaskedLM.from_pretrained('roberta-large')
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')