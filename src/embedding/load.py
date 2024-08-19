import os
import json
from datasets import load_dataset, load_from_disk
from src.utils.text_tools import truncate
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as file:
        prompts = json.load(file)
    return prompts

def prepare_input_from_prompts(device,tokenizer, config, project_root):
    # Define the path for the prompts JSON file
    prompts_path = os.path.join(project_root, 'config', 'prompts.json')

    try:
        # Try to load prompts from the JSON file
        prompts = load_prompts(prompts_path)
        
        # Choose a prompt based on the sample index
        sample_idx = config['sample_idx']  # Ensure this index is valid
        input_text = prompts['sample_idx']  
        # Tokenize and truncate the input text
        tokenized_input = tokenizer(input_text, return_tensors='pt').to(device)
        tokenized_input = truncate(tokenized_input, max_length=300)
    except (FileNotFoundError, IndexError):
        # If the prompts file does not exist or the index is invalid, load from dataset
        c4_sliced_and_filted = load_from_disk(config['dataset_path'])
        c4_sliced_and_filted = c4_sliced_and_filted['train'].shuffle(seed=42).select(range(100))
        
        sample_idx = config['sample_idx']
        input_text = c4_sliced_and_filted[sample_idx]['text']
        # Tokenize and truncate the input text
        tokenized_input = tokenizer(input_text, return_tensors='pt').to(device)
        tokenized_input = truncate(tokenized_input, max_length=300)

    return tokenized_input



def load_model_and_tokenizer(config):
    """
    Load the tokenizer and model based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing model details.

    Returns:
        tokenizer: Loaded tokenizer.
        model: Loaded model.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(config['model_name'], device_map="auto")
    
    return tokenizer, model
