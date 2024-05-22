from importlib import reload
import watermarking.watermark_processor

watermarking.watermark_processor = reload(watermarking.watermark_processor)
from watermarking.watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, LogitsProcessor
from watermarking.utils.text_tools import truncate
from watermarking.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
lm_tokenizer = AutoTokenizer.from_pretrained('gpt2')
lm_model = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda:0')

model = model.to('cuda:0')

c4_sliced_and_filted = load_from_disk('./c4-train.00000-of-00512_sliced')
c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(
    range(100))


# select a prompt

sample_idx = 98
input_text = c4_sliced_and_filted[sample_idx]['text']
tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
tokenized_input = truncate(tokenized_input, max_length=300)

min_length_processor = MinLengthLogitsProcessor(min_length=1000,
                                                eos_token_id=tokenizer.eos_token_id)
repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)