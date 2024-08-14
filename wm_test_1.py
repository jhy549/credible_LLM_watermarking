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
import numpy as np
import json

import logging
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
torch.set_printoptions(threshold=np.inf)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/ailab/user/jianghaoyu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549")
model = AutoModelForCausalLM.from_pretrained("/ailab/user/jianghaoyu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/4782ad278652c7c71b72204d462d6d01eaaf7549",device_map="auto")#,torch_dtype=torch.float16)
# model = model.to('cuda:1')

# model = model.to(devices)
# model = torch.nn.DataParallel(model)
# lm_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b",trust_remote_code=True)
# lm_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b",trust_remote_code=True)
# lm_model = lm_model.to('cuda:1')


c4_sliced_and_filted = load_from_disk('./c4-train.00000-of-00512_sliced')
c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(
    range(100))


sample_idx = 77
# # input_text = "We are pleased to announce the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25), which will be held in Philadelphia, Pennsylvania at the Pennsylvania Convention Center from February 25 to March 4, 2025."
# input_text = c4_sliced_and_filted[sample_idx]['text']
input_text ="Lawn sweepers are more efficient and less time consuming than raking leaves.\n3 What Kind of Rake to Use for Grass Seed?\nLawn sweepers operate with brushes underneath that pick up lawn debris and flip it into a hopper. The process is the same as a vacuum cleaner, but the brushes turn manually as you push a walk-behind sweeper. You may use a lawn sweeper to clean the yard after mowing it, or to clean fallen leaves and debris. An accumulation of dead grass and leaves on your lawn blocks sunlight and air from the grass and can kill the lawn. The process of using a lawn sweeper takes little time and your lawn will thank you for it.\nPut on leather work gloves to protect your hands from thorns or sharp objects. Pick up all branches, rocks, pinecones and other large debris in the area. Lawn sweepers will remove small twigs, grass clippings and leaves, but not large items.\nMow your lawn as you normally would. Lawn sweepers work better on freshly mowed grass that is all the same height.\nAdjust the brush height on a lawn sweeper to the same height as the grass. Some models have a dial to turn or a handle to move"
# print(input_text)

# prefixes = []

# with open('open-generation-data/gpt3_davinci_003_300_len.jsonl_pp', 'r', encoding='utf-8') as file:
#     for line in file:
#         data = json.loads(line.strip())
#         prefixes.append(data['prefix'])


# # 选择样本
# sample_idx = 77
# input_text = prefixes[sample_idx]

# ds = load_dataset("ChristophSchuhmann/essays-with-instructions")
# essays_sliced_and_filted = ds['train'].shuffle(seed=42).select(range(100))
# sample_idx = 77
# input_text = essays_sliced_and_filted[sample_idx]['essays']


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
from watermarking.watermark_processors.message_models.lm_message_model_update_1 import LMMessageModel_update
from watermarking.watermark_processors.message_model_processor_update import WmProcessorMessageModel_update


# lm_message_model = LMMessageModel(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
#     delta = 1.5, lm_prefix_len=10, lm_topk=-1, message_code_len = 10,random_permutation_num=30)
# wm_precessor_message_model = WmProcessorMessageModel(message_model=lm_message_model,tokenizer=tokenizer,
#     encode_ratio=5,max_confidence_lbd=0.5,strategy='max_confidence_updated', message=[241])

# start_length = tokenized_input['input_ids'].shape[-1]
# wm_precessor_message_model.start_length = start_length
# output_tokens = model.generate(**tokenized_input, max_new_tokens=60, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
#                                logits_processor=LogitsProcessorList(
#                                    [min_length_processor, repetition_processor,
#                                     wm_precessor_message_model]))

# output_text = tokenizer.decode(output_tokens[0][start_length:],
#                                skip_special_tokens=True)
# logging.info(output_text)
# prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
# log_probs = wm_precessor_message_model.decode(output_text)
# print("平衡修改：",log_probs[1][1][0][2]) 
# print("平衡修改：",log_probs[0])




# rm_lm_message_model = RandomMessageModel(tokenizer=tokenizer,
#                                           lm_tokenizer=tokenizer,
#                                           delta=1.5,
#                                           message_code_len=10,
#                                           device=model.device,
#                                           )

# rm_watermark_processor = WmProcessorRandomMessageModel(message_model=rm_lm_message_model,
#                                                         tokenizer=tokenizer,
#                                                         encode_ratio=5,
#                                                         message=[102],
#                                                         top_k=1000,
#                                                         )


# start_length = tokenized_input['input_ids'].shape[-1]
# rm_watermark_processor.start_length = start_length
# output_tokens_rm = model.generate(**tokenized_input, max_new_tokens=50, num_beams=4, temperature=1.0,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
#                                logits_processor=LogitsProcessorList(
#                                    [min_length_processor, repetition_processor,
#                                     rm_watermark_processor]))

# output_text_rm = tokenizer.decode(output_tokens_rm[0][start_length:],
#                                skip_special_tokens=True)
# logging.info(output_text_rm)
# prefix_and_output_text_rm = tokenizer.decode(output_tokens_rm[0], skip_special_tokens=True)
# log_probs_rm = rm_watermark_processor.decode(output_text_rm)
# print("直接修改：",log_probs_rm[1][1][0][2]) 
# print("直接修改：",log_probs_rm[0])




# lm_message_model_update = LMMessageModel_update(tokenizer=tokenizer,lm_model=model,lm_tokenizer=tokenizer,
#     delta = 1.5, lm_prefix_len=10, lm_topk=-1, message_code_len = 20,random_permutation_num=200)
# wm_precessor_message_model_update = WmProcessorMessageModel_update(message_model=lm_message_model_update,tokenizer=tokenizer,
#     encode_ratio=4,max_confidence_lbd=0.5,strategy='max_confidence_updated', message=[618])

# start_length = tokenized_input['input_ids'].shape[-1]
# wm_precessor_message_model_update.start_length = start_length
# output_tokens_update = model.generate(**tokenized_input, max_new_tokens=90, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
#                                logits_processor=LogitsProcessorList(
#                                    [min_length_processor, repetition_processor,
#                                     wm_precessor_message_model_update]))

# output_text_update = tokenizer.decode(output_tokens_update[0][start_length:],
#                                skip_special_tokens=True)
# # logging.info(output_text_update)
# # print(output_text_update)
# # output_text_update_1 = """for everyone.”
# # Miry said she hopes the Hardin can become the centerpiece of a rejuvenated downtown corridor.
# # “I think it’s going to be a very vibrant part of the community,” he said. “We’re trying to create a walkable, bike-friendly neighborhood, where you can get your errands done without having to get in your car.”
# # Read or Share this story: https://www.knoxnews.com/story/money/business/2019/07/18/hardin-apartment-complex-opens-downtown-knoxville/1759387001/
# # Knoxville entrepreneur launches clothing company for kids with special needs
# # Pigeon Forge tops TripAdvisor list of best C.K. summer destinations
# # W"""

# prefix_and_output_text_update = tokenizer.decode(output_tokens_update[0], skip_special_tokens=True)
# log_probs_update = wm_precessor_message_model_update.decode(output_text_update[:len(output_text_update)*2//3])
# print("我的修改：",log_probs_update[1][1][0][2]) 
# print("我的修改：",log_probs_update[0])









# oracle_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
# oracle_model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-13b",device_map="auto")#,torch_dtype=torch.float16)


# from watermarking.experiments.watermark import compute_ppl_single

# loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text_update,
#                                oracle_model_name='huggyllama/llama-13b',
#                                output_text=output_text_update,
#                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
# print("我的：",loss, ppl)

# pre = "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: How can we reduce air pollution?\n\n### Assistant:Here are some ways to reduce air pollution:\n\\begin{itemize}\n\\item Encouraging the use of public transportation, cycling, and walking instead of driving personal cars. This will reduce the number of cars on the road, which in turn will lower emissions from vehicles.\n\\item Planning cities and communities in a more sustainable way, with the aim of reducing the need for long commutes and promoting the use of green spaces and public transportation.\n\\item Encouraging the use of renewable energy sources like wind and solar power instead of fossil fuels. This will reduce the emissions from power plants and help the transition to a greener future.\n\\item Encouraging recycling and composting of waste to reduce the amount that ends up in landfills, where it releases methane, a potent greenhouse gas.\n\\item Encouraging the use of green building practices like energy-efficient windows, insulation, and solar panels to reduce the energy consumption of buildings.\n\\end{itemize}\n\nThese are some of the ways we can reduce air pollution. It is important to note that it takes time and effort"
# out = "A chat between a curious human and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions.\n\n### Human: How can we reduce air pollution?\n\n### Assistant:"
# loss, ppl = compute_ppl_single(prefix_and_output_text=pre,
#                                oracle_model_name='huggyllama/llama-13b',
#                                output_text=out,
#                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
# print("测试：",loss, ppl)


# loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
#                                oracle_model_name='huggyllama/llama-13b',
#                                output_text=output_text,
#                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
# print("平衡",loss, ppl)

# loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text_rm,
#                                oracle_model_name='huggyllama/llama-13b',
#                                output_text=output_text_rm,
#                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
# print("随机",loss, ppl)




no_wm_output_tokens = model.generate(**tokenized_input, max_new_tokens=400, num_beams=4,#top_k=50,top_p=0.95,do_sample=True,#num_beams=4,
                                     logits_processor=LogitsProcessorList(
                                       [min_length_processor, repetition_processor]))
no_wm_output_text = tokenizer.decode(no_wm_output_tokens[0][tokenized_input['input_ids'].shape[-1]:],
                                     skip_special_tokens=True)
no_wm_prefix_and_output_text = tokenizer.decode(no_wm_output_tokens[0], skip_special_tokens=True)
print(no_wm_output_text )
# # logging.warning(no_wm_output_text)
# # loss, ppl = compute_ppl_single(prefix_and_output_text=no_wm_prefix_and_output_text,
# #                                oracle_model_name='huggyllama/llama-13b',
# #                                output_text=no_wm_output_text,
# #                                oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
# # print("无水印",loss, ppl)


# output_text_update_no = tokenizer.decode(no_wm_output_tokens[0][start_length:],
#                                skip_special_tokens=True)
# output_text_update_no = "We are pleased to announce the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25), which will be held in Philadelphia, Pennsylvania at the Pennsylvania Convention Center from February 25 to March 4, 2025"
# # logging.info(output_text_update)
# for i in range(100):
#     output_text_update_no = c4_sliced_and_filted[i]['text'][:60]

#     log_probs_update = wm_precessor_message_model_update.decode(output_text_update_no)
#     print("二型错误：",log_probs_update[1][1][0][2]) 
#     print("二型错误：",log_probs_update[0])