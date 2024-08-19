# CredID: Credible LLM Watermarking Framework and Toolkit

## Overview

This repository contains the open-source toolkit and research materials for the "Credibility-Driven Multi-bit Watermark for Large Language Models Identification" project. The project introduces a novel multi-party credible watermarking framework that enhances the identification, privacy, and security of Large Language Models (LLMs). The watermarking scheme is designed to embed and extract multi-bit information without compromising the quality of the generated text. This repository contains code for the paper [Towards Codable Watermarking for Large Language Models](https://github.com/lancopku/codable-watermarking-for-llm) by Lean Wang, Wenkai Yang, Deli Chen, Hao Zhou, Yankai Lin, Fandong Meng, Jie Zhou and Xu Sun, and the paper [MarkLLM: An Open-source toolkit for LLM Watermarking](https://github.com/THU-BPM/MarkLLM) by Leyi Pan, Aiwei Liu, Zhiwei He, Zitian Gao, Xuandong Zhao, Yijian Lu, Binglin Zhou, Shuliang Liu, Xuming Hu, Lijie Wen, Irwin King and Philip S. Yu.

## Key Features

- **Multi-party Participation**: A collaborative watermarking framework involving a Trusted Third Party (TTP) and multiple LLM vendors.
- **Privacy Protection**: Ensures that user prompts and model outputs remain confidential during the watermarking process.
- **Flexibility and Scalability**: Easily accommodates different vendors' private watermarking technologies and allows for the upgrading of authentication and encryption algorithms.
- **Robust Watermarking Algorithm**: A multi-bit watermarking method that improves success rates and information capacity, with a focus on maintaining text quality.
- **Open-source Toolkit**: A user-friendly toolkit that includes watermarking frameworks, experiment configurations, datasets, and baseline algorithms.

## Repository Structure

```
llm-watermarking/
│
├── paper/                  # Research paper and related documents
│   ├── Credibility-Driven_Multi-bit_Watermark.pdf
│   └── Reproducibility_Checklist.md
│
├── config/                 # Configuration files for various watermark algorithms          
│   ├── CredID.json       
│   ├── EXP.json           
│   ├── KGW.json
│   ├── CTWL.json            
│   ├── SIR.json            
│   ├── SWEET.json         
│   ├── Unigram.json        
│   ├── MPAC.json           
│   └── Cyclic-shift.json    
│       
├── src/                    # Source code for the watermarking framework
│   ├── embedding/          # Embedding watermark into LLM responses
│   ├── extraction/         # Extracting watermark from text
│   └── utils/              # Utility functions and modules
│
├── datasets/               # Sample datasets used in the experiments
│   ├── multi_vendors_data/
│   ├── OpenGen/
│   ├── C4_News/
│   └── Essays/
│
├── watermarking/          #Implementation framework for watermark algorithms
│   ├── watermark_processor.py
│   ├── base.py
│   ├── KGW/
│   ├── EXP/
│   ├── SIR/
│   ├── SWEET/
│   ├── Unigram/
│   ├── CredID/
│   ├── CTWL/
│   ├── MPAC/
│   └── Cyclic-shift/
│
├── attacks/                #Implementation framework for watermark algorithms
│   ├── get_datasets.py
│   ├── cp_attack.py
│   ├── Deletion_attack.py/
│   ├── Substitution_attack.py
│   └── Homoglyph_attack.py
│
├── evaluation/                 # Evaluation module, including tools and pipelines
│   ├── dataset.py              # Script for handling dataset operations within evaluations
│   ├── examples/               # Scripts for automated evaluations using pipelines
│   │   ├── assess_success_rate.py  
│   │   ├── assess_quality.py
│   │   ├── assess_speed.py    
│   │   └── assess_robustness.py   
│   ├── pipelines/              # Pipelines for structured evaluation processes
│   │   ├── speed_analysis.py    
│   │   ├── success_rate_analysis.py    
│   │   ├── multi_voting_analysis.py  
│   │   └── quality_analysis.py 
│   └── tools/                  # Evaluation tools
│       ├── ppl.py
│       ├── metrics/ 
│       └── BLEU.py   
│
├── demo/                   # Credible watermark framework cases and examples for user testing
│   ├── test_method_single_party.py      
│   ├── test_pipeline_multi_party.py    
│   └── test_real_word.py
│
├── experiments/            # Scripts and configurations for running experiments
│
├── README.md               # Main project documentation
└── requirements.txt        # Dependencies required for the project
```

## Installation

To install the required dependencies and set up the environment, follow these steps:

1. Clone the repository:

   ```
   git clone https://anonymous.4open.science/r/credible-LLM-watermarking-7D62.git
   ```

2. Navigate to the repository:

   ```
   cd credible-LLM-watermarking
   ```

3. Create a virtual environment and activate it (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

The toolkit provides functionalities for both watermark embedding and extraction. Here are the basic steps to use the toolkit:

### Embedding Watermarks

1. Prepare your LLM and the text prompt, and converts model version information, timestamps, and extra information into a binary sequence of a specific length as a secret message.

   ```python
   # generate secret message
    message = secret_message_gen(model_version, timestamps, extra_info, length)
   # message:[0011110001] → 241
   
   #./config/CredID.json
    {
       "method": "CredID",
       "prompt": "A new study has found that the universe is made of",
       "model_name": "huggyllama/llama-7b",
       "oracle_model_name": "huggyllama/llama-13b",
       "dataset_path": "./datasets/c4-train.00000-of-00512_sliced",
       "log_file": "example.log",
       "sample_idx": 77,
       "max_new_tokens": 110,
       "num_beams": 4,
       "min_length": 10000,
       "penalty": 1.5,
       "lm_message_model_params": {
           "delta": 1.5,
           "lm_prefix_len": 10,
           "lm_topk": -1,
           "message_code_len": 10,
           "random_permutation_num": 50
       },
       "wm_processor_message_model_params": {
           "encode_ratio": 10,
           "max_confidence_lbd": 0.5,
           "strategy": "max_confidence_updated",
           "message": [241]
       }
   }
   ```



2. Use the embedding module to integrate the watermark into the LLM's response.

   ``````python
   output_tokens = model.generate(
            **tokenized_input, 
            max_new_tokens=config['max_new_tokens'], 
            num_beams=config['num_beams'],
            logits_processor=LogitsProcessorList([min_length_processor, repetition_processor, wm_precessor_message_model])
         )
   print("watermarked_text:",output_tokens)
   ``````

### Extracting Watermarks

1. Obtain the watermarked text.  
2. Utilize the extraction module to decode and verify the watermark.

   ```python
   log_probs = wm_precessor_message_model.decode(output_text)
   print("my watermark confidence:", log_probs[1][1][0][2])
   #watermark confidence:0.99872
   print("extracted message:", log_probs[0])
   #extracted message:241
   ```

### Watermark Pipeline User Case

```python
import json
import os
import torch
import sys
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, MinLengthLogitsProcessor
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from watermarking.watermark_processor import RepetitionPenaltyLogitsProcessor
from src.utils.text_tools import truncate
from src.utils.load_local import load_local_model_or_tokenizer
from datasets import load_dataset, load_from_disk
from watermarking.CredID.message_models.lm_message_model import LMMessageModel
from watermarking.CredID.message_model_processor import WmProcessorMessageModel
from src.utils.watermark import compute_ppl_single
from src.embedding.load import load_config, prepare_input_from_prompts, load_model_and_tokenizer

   def main():
      # Define the path of the configuration file
      config_path = os.path.join(project_root, 'config', 'CTWL.json')
      
      # Load the configuration
      config = load_config(config_path)
      
      # Configure logging
      logging.basicConfig(filename=config['log_file'], level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
      torch.set_printoptions(threshold=np.inf)
      
      #Load the model and tokenizer using the new function
      tokenizer, model = load_model_and_tokenizer(config) 

      tokenized_input = prepare_input_from_prompts(model.device,tokenizer, config, project_root)

      min_length_processor = MinLengthLogitsProcessor(min_length=config['min_length'], eos_token_id=tokenizer.eos_token_id)
      repetition_processor = RepetitionPenaltyLogitsProcessor(penalty=config['penalty'])

      # Read the parameters of lm_message_model from the configuration file
      lm_message_model_params = config['lm_message_model_params']
      lm_message_model = LMMessageModel(
         tokenizer=tokenizer, 
         lm_model=model, 
         lm_tokenizer=tokenizer,
         delta=lm_message_model_params['delta'], 
         lm_prefix_len=lm_message_model_params['lm_prefix_len'], 
         lm_topk=lm_message_model_params['lm_topk'], 
         message_code_len=lm_message_model_params['message_code_len'], 
         random_permutation_num=lm_message_model_params['random_permutation_num']
      )

      # Read the parameters of wm_processor_message_model from the configuration file
      wm_processor_message_model_params = config['wm_processor_message_model_params']
      wm_precessor_message_model = WmProcessorMessageModel(
         message_model=lm_message_model, 
         tokenizer=tokenizer,
         encode_ratio=wm_processor_message_model_params['encode_ratio'], 
         max_confidence_lbd=wm_processor_message_model_params['max_confidence_lbd'], 
         strategy=wm_processor_message_model_params['strategy'], 
         message=wm_processor_message_model_params['message']
      )
      
      start_length = tokenized_input['input_ids'].shape[-1]
      wm_precessor_message_model.start_length = start_length
      output_tokens = model.generate(
         **tokenized_input, 
         max_new_tokens=config['max_new_tokens'], 
         num_beams=config['num_beams'],
         logits_processor=LogitsProcessorList([min_length_processor, repetition_processor, wm_precessor_message_model])
      )
      print("watermarked_text:",output_tokens)
      output_text = tokenizer.decode(output_tokens[0][start_length:], skip_special_tokens=True)
      logging.info(output_text)
      prefix_and_output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
      log_probs = wm_precessor_message_model.decode(output_text)
      print("my watermark confidence:", log_probs[1][1][0][2])
      print("decoded message:", log_probs[0])
      
      
      oracle_tokenizer = AutoTokenizer.from_pretrained(config['oracle_model_name'])
      oracle_model = AutoModelForCausalLM.from_pretrained(config['oracle_model_name'],device_map="auto")#,torch_dtype=torch.float16)
      

      loss, ppl = compute_ppl_single(prefix_and_output_text=prefix_and_output_text,
                                 oracle_model_name='huggyllama/llama-13b',
                                 output_text=output_text,
                                 oracle_model=oracle_model, oracle_tokenizer=oracle_tokenizer)
      print("loss,ppl:",loss, ppl)

   if __name__ == '__main__':
      main()
```

### Running Experiments

1. Configure the experiment settings in the `experiments/` directory.
2. Run the experiment scripts to evaluate the watermarking performance in the `evaluation/` directory.

### Running Demo

```
cd ./demo
python test_method_single_party.py  #for single party watermark embedding and extraction

python test_method_multi_party.py  #for multiple parties watermark embedding and joint-voting extraction

# data from ./datasets/multi_vendors_data
python test_real_word.py  #for realistic scenario simulation emulation: mix of non-watermarked text and multi-vendor watermarked text
```



## Contributing

We welcome contributions to the project. Please follow the guidelines in `CONTRIBUTING.md` for submitting issues and pull requests.

## License

This project is licensed under the [LICENSE_NAME](LICENSE). Please refer to the license file for more information.

## Citation

If you use this toolkit in your research, please cite our paper:

@misc{anonymous2024watermarking,
  author = {Anonymous},
  title = {CredID: Credible Multi-bit Watermark for Large Language Models Identification},
  year = {2024},
  howpublished = {https://anonymous.4open.science/r/credible-LLM-watermarking-7D62},
}

## Acknowledgements

We would like to acknowledge the contributions of the LLM research community and the developers of the open-source models used in our experiments.

---

Please replace `LICENSE_NAME` with the actual name of the license you are using for this project. If you need any further customization or have specific requirements, feel free to ask.