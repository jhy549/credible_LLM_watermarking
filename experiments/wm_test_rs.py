from importlib import reload
import watermarking.watermark_processor
import torch
watermarking.watermark_processor = reload(watermarking.watermark_processor)
from watermarking.watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, LogitsProcessor
from src.utils.text_tools import truncate
from src.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk
from evaluation.tools.test_rs import BinaryCodec

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
# torch.set_printoptions(threshold=10000)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b",torch_dtype=torch.float16)
model = model.to('cuda:0')
# lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# lm_model = AutoModelForCausalLM.from_pretrained("gpt2")
# lm_model = lm_model.to('cuda:0')



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
import watermarking.CredID.message_models.lm_message_model
watermarking.CredID.message_models.lm_message_model = reload(watermarking.CredID.message_models.lm_message_model)
import watermarking.CredID.message_model_processor
watermarking.CredID.message_model_processor = reload(watermarking.CredID.message_model_processor)
from watermarking.CredID.message_models.lm_message_model import LMMessageModel
from watermarking.CredID.message_model_processor import WmProcessorMessageModel

# origin_message = [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0]
origin_message = [1,0,1,0,1,1,1,1,1,1]
codec = BinaryCodec(8) 
# encode_message = origin_message
encode_message = codec.bytearray_to_bin_list(codec.encode(origin_message))
# encode_message = codec.encode(origin_message)
print(encode_message)
encode_ratio = 10
lm_prefix_len=10


lm_message_model = LMMessageModel(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
    delta = 5.5, lm_prefix_len=lm_prefix_len, lm_topk=-1, message_code_len = 1,random_permutation_num=50)
wm_precessor_message_model = WmProcessorMessageModel(message_model=lm_message_model,tokenizer=tokenizer,
    encode_ratio=encode_ratio,max_confidence_lbd=0.5,strategy='max_confidence_updated', message=encode_message)

start_length = tokenized_input['input_ids'].shape[-1]
wm_precessor_message_model.start_length = start_length
output_tokens = model.generate(**tokenized_input, max_new_tokens=len(encode_message)*encode_ratio+lm_prefix_len, num_beams=4,
                               logits_processor=LogitsProcessorList(
                                   [min_length_processor, repetition_processor,
                                    wm_precessor_message_model]))

output_text = tokenizer.decode(output_tokens[0][start_length:],
                               skip_special_tokens=True)
logging.info(output_text)
prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
log_probs = wm_precessor_message_model.decode(output_text)
# print(log_probs) 
print(log_probs[0])
# print(codec.bin_list_to_bytearray(log_probs[0]))
decode_message = codec.decode(codec.bin_list_to_bytearray(log_probs[0]))
print(decode_message)

