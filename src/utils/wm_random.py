import json
import os.path
from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
from datasets import load_dataset, load_from_disk
from .watermark_processor import RepetitionPenaltyLogitsProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from .utils.load_local import load_local_model_or_tokenizer
from .arg_classes.wm_arg_class import WmRandomArgs
from .watermark_processors.random_message_model_processor import WmProcessorRandomMessageModel
from .watermark_processors.message_models.random_message_model import RandomMessageModel

ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def truncate(d, max_length=200):
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and len(v.shape) == 2:
            d[k] = v[:, :max_length]
    return d


def main(args: WmRandomArgs):
    if os.path.exists(args.complete_save_file_path):
        print(f'{args.complete_save_file_path} already exists, skip.')
        return

    if args.model_name in ['huggyllama/llama-7b','facebook/opt-1.3b', 'facebook/opt-2.7b', 'alpaca-native']:
        # tokenizer = load_local_model_or_tokenizer(args.model_name.split('/')[-1], 'tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # model = load_local_model_or_tokenizer(args.model_name.split('/')[-1], 'model')
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(args.model_name,device_map="auto")
    else:
        raise NotImplementedError(f'model_name: {args.model_name}')

    # model = model.to(args.device)

    # if args.lm_model_name == 'same':
    #     lm_tokenizer = tokenizer
    # else:
    #     lm_tokenizer = load_local_model_or_tokenizer(args.lm_model_name.split('/')[-1], 'tokenizer')

    c4_sliced_and_filted = load_from_disk(
        os.path.join(ROOT_PATH, 'c4-train.00000-of-00512_sliced'))
    c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=args.sample_seed).select(
        range(args.sample_num))

    lm_message_model = RandomMessageModel(tokenizer=tokenizer,
                                          lm_tokenizer=tokenizer,
                                          delta=args.delta,
                                          message_code_len=args.message_code_len,
                                          device=model.device,
                                          )

    watermark_processor = WmProcessorRandomMessageModel(message_model=lm_message_model,
                                                        tokenizer=tokenizer,
                                                        encode_ratio=args.encode_ratio,
                                                        message=args.message,
                                                        top_k=args.top_k,
                                                        )

    min_length_processor = MinLengthLogitsProcessor(min_length=1000,
                                                    # just to make sure there's no EOS
                                                    eos_token_id=tokenizer.eos_token_id)
    rep_processor = RepetitionPenaltyLogitsProcessor(penalty=args.repeat_penalty)

    logit_processor = LogitsProcessorList(
        [min_length_processor, rep_processor, watermark_processor])

    results = {'text': [],
               'prefix_and_output_text': [],
               'output_text': [],
               'decoded_message': [],
               'acc': []}

    try:
        for text in tqdm(c4_sliced_and_filted['text']):
            tokenized_input = tokenizer(text, return_tensors='pt').to(model.device)
            tokenized_input = truncate(tokenized_input, max_length=args.prompt_length)

            watermark_processor.start_length = tokenized_input['input_ids'].shape[-1]
            output_tokens = model.generate(**tokenized_input,
                                           temperature=args.temperature,
                                           max_new_tokens=args.generated_length,
                                           num_beams=args.num_beams,
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

            decoded_message = watermark_processor.decode(output_text, disable_tqdm=True)[0]
            available_message_num = args.generated_length // (
                int(args.message_code_len * args.encode_ratio))
            acc = decoded_message[:available_message_num] == args.message[:available_message_num]

            results['decoded_message'].append(decoded_message)
            results['acc'].append(acc)

    except KeyboardInterrupt:
        pass

    os.makedirs(os.path.dirname(args.complete_save_file_path), exist_ok=True)

    args_dict = vars(args)
    results['args'] = args_dict
    with open(args.complete_save_file_path, 'w') as f:
        json.dump(results, f, indent=4)
