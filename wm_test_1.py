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
model = model.to('cuda:3')

c4_sliced_and_filted = load_from_disk('./c4-train.00000-of-00512_sliced')
c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(
    range(100))


sample_idx = 98
input_text = c4_sliced_and_filted[sample_idx]['text']
tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
tokenized_input = truncate(tokenized_input, max_length=300)

min_length_processor = MinLengthLogitsProcessor(min_length=1000,
                                                eos_token_id=tokenizer.eos_token_id)
repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

from importlib import reload
import watermarking.watermark_processors.message_models.lm_message_model
watermarking.watermark_processors.message_models.lm_message_model = reload(watermarking.watermark_processors.message_models.lm_message_model)
import watermarking.watermark_processors.message_model_processor
watermarking.watermark_processors.message_model_processor = reload(watermarking.watermark_processors.message_model_processor)
from watermarking.watermark_processors.message_models.lm_message_model import LMMessageModel
from watermarking.watermark_processors.message_model_processor import WmProcessorMessageModel

lm_message_model = LMMessageModel(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
    delta = 5.5, lm_prefix_len=10, lm_topk=-1, message_code_len = 10,random_permutation_num=50)
wm_precessor_message_model = WmProcessorMessageModel(message_model=lm_message_model,tokenizer=tokenizer,
    encode_ratio=5,max_confidence_lbd=0.5,strategy='max_confidence', message=[1023,1013,213,333,678,994])

start_length = tokenized_input['input_ids'].shape[-1]
wm_precessor_message_model.start_length = start_length
output_tokens = model.generate(**tokenized_input, max_new_tokens=300, num_beams=4,
                               logits_processor=LogitsProcessorList(
                                   [min_length_processor, repetition_processor,
                                    wm_precessor_message_model]))

output_text = tokenizer.decode(output_tokens[0][tokenized_input['input_ids'].shape[-1]:],
                               skip_special_tokens=True)
print(output_text)
prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
log_probs = wm_precessor_message_model.decode(output_text)
print(log_probs)
print(log_probs[0])xxxx