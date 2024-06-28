from importlib import reload
import watermarking.watermark_processor
import torch
watermarking.watermark_processor = reload(watermarking.watermark_processor)
from watermarking.watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, LogitsProcessor
from watermarking.utils.text_tools import truncate
from watermarking.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk
from watermarking.watermark_processors.random_message_model_processor import WmProcessorRandomMessageModel
from watermarking.watermark_processors.message_models.random_message_model import RandomMessageModel

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
# torch.set_printoptions(threshold=10000)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b",device_map="auto")#,torch_dtype=torch.float16)
# model = model.to('cuda:1')

# model = model.to(devices)
# model = torch.nn.DataParallel(model)
# lm_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b",trust_remote_code=True)
# lm_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b",trust_remote_code=True)
# lm_model = lm_model.to('cuda:1')


c4_sliced_and_filted = load_from_disk('./c4-train.00000-of-00512_sliced')
c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(
    range(100))


sample_idx = 98
input_text = c4_sliced_and_filted[sample_idx]['text']
# input_text = "For important files, data, information, etc. transmitted and stored in the computer system, it is generally necessary to "
# print(input_text)
tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)
tokenized_input = truncate(tokenized_input, max_length=300)

min_length_processor = MinLengthLogitsProcessor(min_length=10000,
                                                eos_token_id=tokenizer.eos_token_id)
repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)

from importlib import reload
import watermarking.watermark_processors.message_models.lm_message_model
watermarking.watermark_processors.message_models.lm_message_model = reload(watermarking.watermark_processors.message_models.lm_message_model)
import watermarking.watermark_processors.message_model_processor
watermarking.watermark_processors.message_model_processor = reload(watermarking.watermark_processors.message_model_processor)
from watermarking.watermark_processors.message_models.lm_message_model import LMMessageModel
from watermarking.watermark_processors.message_model_processor import WmProcessorMessageModel
from watermarking.watermark_processors.message_models.lm_message_model_update import LMMessageModel_update
from watermarking.watermark_processors.message_model_processor_update import WmProcessorMessageModel_update


lm_message_model = LMMessageModel(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
    delta = 2.5, lm_prefix_len=10, lm_topk=-1, message_code_len = 10,random_permutation_num=50)
wm_precessor_message_model = WmProcessorMessageModel(message_model=lm_message_model,tokenizer=tokenizer,
    encode_ratio=5,max_confidence_lbd=0.5,strategy='max_confidence_updated', message=[681,123,345])

start_length = tokenized_input['input_ids'].shape[-1]
wm_precessor_message_model.start_length = start_length
output_tokens = model.generate(**tokenized_input, max_new_tokens=160, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                               logits_processor=LogitsProcessorList(
                                   [min_length_processor, repetition_processor,
                                    wm_precessor_message_model]))

output_text = tokenizer.decode(output_tokens[0][start_length:],
                               skip_special_tokens=True)
logging.info(output_text)
prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
log_probs = wm_precessor_message_model.decode(output_text)
print("平衡修改：",log_probs[1][1][0][2]) 
print("平衡修改：",log_probs[0])




rm_lm_message_model = RandomMessageModel(tokenizer=tokenizer,
                                          lm_tokenizer=tokenizer,
                                          delta=5.5,
                                          message_code_len=10,
                                          device=model.device,
                                          )

rm_watermark_processor = WmProcessorRandomMessageModel(message_model=rm_lm_message_model,
                                                        tokenizer=tokenizer,
                                                        encode_ratio=5,
                                                        message=[681,123,345],
                                                        top_k=1000,
                                                        )


start_length = tokenized_input['input_ids'].shape[-1]
rm_watermark_processor.start_length = start_length
output_tokens_rm = model.generate(**tokenized_input, max_new_tokens=150, num_beams=4, temperature=1.0,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                               logits_processor=LogitsProcessorList(
                                   [min_length_processor, repetition_processor,
                                    rm_watermark_processor]))

output_text_rm = tokenizer.decode(output_tokens_rm[0][start_length:],
                               skip_special_tokens=True)
logging.info(output_text_rm)
prefix_and_output_text_rm = tokenizer.decode(output_tokens_rm[0], skip_special_tokens=True)
log_probs_rm = rm_watermark_processor.decode(output_text_rm)
print("直接修改：",log_probs_rm[1][1][0][2]) 
print("直接修改：",log_probs_rm[0])




lm_message_model_update = LMMessageModel_update(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
    delta = 5.5, lm_prefix_len=10, lm_topk=-1, message_code_len = 10,random_permutation_num=50)
wm_precessor_message_model_update = WmProcessorMessageModel_update(message_model=lm_message_model_update,tokenizer=tokenizer,
    encode_ratio=5,max_confidence_lbd=0.5,strategy='max_confidence_updated', message=[681,123,345])

start_length = tokenized_input['input_ids'].shape[-1]
wm_precessor_message_model_update.start_length = start_length
output_tokens_update = model.generate(**tokenized_input, max_new_tokens=160, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                               logits_processor=LogitsProcessorList(
                                   [min_length_processor, repetition_processor,
                                    wm_precessor_message_model_update]))

output_text_update = tokenizer.decode(output_tokens_update[0][start_length:],
                               skip_special_tokens=True)
logging.info(output_text_update)
prefix_and_output_text_update = tokenizer.decode(output_tokens_update[0], skip_special_tokens=True)
log_probs_update = wm_precessor_message_model_update.decode(output_text_update)
print("我的修改：",log_probs_update[1][1][0][2]) 
print("我的修改：",log_probs_update[0])









oracle_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
oracle_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b",device_map="auto")#,torch_dtype=torch.float16)

# oracle_tokenizer = load_local_model_or_tokenizer('facebook/opt-2.7b', 'tokenizer')
# oracle_model = load_local_model_or_tokenizer('facebook/opt-2.7b', 'model')
# oracle_model = oracle_model.to('cuda:1')

from watermarking.experiments.watermark import compute_ppl_single

loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text_update,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=output_text_update,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("列表：",loss, ppl)

loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=output_text,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("平衡",loss, ppl)

loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text_rm,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=output_text_rm,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("随机",loss, ppl)




no_wm_output_tokens = model.generate(**tokenized_input, max_new_tokens=160, num_beams=1,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                                     logits_processor=LogitsProcessorList(
                                       [min_length_processor, repetition_processor]))
no_wm_output_text = tokenizer.decode(no_wm_output_tokens[0][tokenized_input['input_ids'].shape[-1]:],
                                     skip_special_tokens=True)
no_wm_prefix_and_output_text = tokenizer.decode(no_wm_output_tokens[0], skip_special_tokens=True)
logging.warning(no_wm_output_text)
loss, ppl = compute_ppl_single(prefix_and_output_text=no_wm_prefix_and_output_text,
                               oracle_model_name='huggyllama/llama-13b',
                               output_text=no_wm_output_text,
                               oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
print("无水印",loss, ppl)
