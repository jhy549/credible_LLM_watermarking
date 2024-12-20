import json
import os.path
from dataclasses import dataclass
import random

import torch
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from datasets import load_dataset, load_from_disk
from watermarking.watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from src.utils.load_local import load_local_model_or_tokenizer
from watermarking.arg_classes.wm_arg_class import WmLMArgs
from watermarking.CredID.message_models.lm_message_model import LMMessageModel
from watermarking.CredID.message_model_processor import WmProcessorMessageModel

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d


def main(args: WmLMArgs):
    if os.path.exists(args.complete_save_file_path):
        print(f'{args.complete_save_file_path} already exists, skip.')
        return

    if args.model_name in ['facebook/opt-1.3b', 'facebook/opt-2.7b', 'alpaca-native','huggyllama/llama-7b','tiiuae/falcon-7b','google/gemma-7b','meta-llama/Meta-Llama-3-8B']:
        # tokenizer = load_local_model_or_tokenizer(args.model_name.split('/')[-1], 'tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # model = load_local_model_or_tokenizer(args.model_name.split('/')[-1], 'model')
        model = AutoModelForCausalLM.from_pretrained(args.model_name,device_map="auto")
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(args.model_name,device_map="auto")
    else:
        raise NotImplementedError(f'model_name: {args.model_name}')

    # model = model.to(args.device)

    # if args.lm_model_name == 'same':
    #     lm_model = model
    #     lm_tokenizer = tokenizer
    # else:
    #     # lm_model = load_local_model_or_tokenizer(args.lm_model_name.split('/')[-1], 'model')
    #     # lm_tokenizer = load_local_model_or_tokenizer(args.lm_model_name.split('/')[-1], 'tokenizer')
    #     lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_name,device_map="auto")
    #     lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name)
    #     # lm_model = lm_model.to(args.device)

    c4_sliced_and_filted = load_from_disk(
        os.path.join(ROOT_PATH, 'c4-train.00000-of-00512_sliced'))
    c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=args.sample_seed).select(
        range(args.sample_num))
    


    lm_message_model = LMMessageModel(tokenizer=tokenizer, lm_model=model,
                                      lm_tokenizer=tokenizer,
                                      delta=args.delta, lm_prefix_len=args.lm_prefix_len,
                                      lm_topk=args.lm_top_k, message_code_len=args.message_code_len,
                                      random_permutation_num=args.random_permutation_num,shifts=args.shifts)

    watermark_processor = WmProcessorMessageModel(message_model=lm_message_model,
                                                  tokenizer=tokenizer,
                                                  encode_ratio=args.encode_ratio,
                                                  max_confidence_lbd=args.max_confidence_lbd,
                                                  strategy=args.message_model_strategy,
                                                  message=args.message,
                                                  top_k=args.top_k,
                                                  )  

    min_length_processor = MinLengthLogitsProcessor(min_length=10000,
                                                    # just to make sure there's no EOS
                                                    eos_token_id=tokenizer.eos_token_id)
    rep_processor = RepetitionPenaltyLogitsProcessor(penalty=args.repeat_penalty)

    logit_processor = LogitsProcessorList(
        [min_length_processor, rep_processor, watermark_processor])

    results = {'text': [],
               'prefix_and_output_text': [],
               'output_text': [],
               'decoded_message': [],
               'acc': [],
               'confidence': []}

    try:
        text_list = c4_sliced_and_filted['text']
        total_length = len(text_list)

        # 计算一半的长度
        half_length = total_length // 4

        # 获取前一半的数据
        half_text_list = text_list[:half_length]
        
        # essays_sliced_and_filted['essays']
        for text in tqdm(c4_sliced_and_filted['text']):
            tokenized_input = tokenizer(text, return_tensors='pt').to(model.device)
            tokenized_input = truncate(tokenized_input, max_length=args.prompt_length)

            watermark_processor.start_length = tokenized_input['input_ids'].shape[-1]
            output_tokens = model.generate(**tokenized_input,
                                           temperature=args.temperature,
                                           max_new_tokens=args.generated_length,
                                        #    num_beams=args.num_beams,
                                        top_k=5,top_p=0.95,do_sample=True,
                                           logits_processor=logit_processor)

            output_text = \
                tokenizer.batch_decode(
                    output_tokens[:, tokenized_input["input_ids"].shape[-1]:],
                    skip_special_tokens=True)[0]

            prefix_and_output_text = tokenizer.batch_decode(output_tokens,
                                                            skip_special_tokens=True)[0]

            results['text'].append(text)
            results['output_text'].append(output_text)
            results['prefix_and_output_text'].append(prefix_and_output_text)

            decoded_message, other_information = watermark_processor.decode(output_text, disable_tqdm=True)
            print(decoded_message)
            confidences = other_information[1]

            available_message_num = args.generated_length // (
                int(args.message_code_len * args.encode_ratio))
            acc = decoded_message[:available_message_num] == args.message[:available_message_num]

            results['confidence'].append(confidences)

            results['decoded_message'].append(decoded_message)
            results['acc'].append(acc)

    except KeyboardInterrupt:
        pass

    os.makedirs(os.path.dirname(args.complete_save_file_path), exist_ok=True)

    args_dict = vars(args)
    results['args'] = args_dict
    with open(args.complete_save_file_path, 'w') as f:
        json.dump(results, f, indent=4)
