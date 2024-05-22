from watermarking.utils.experiment_utils import Sh_run

sample_seeds = [42]

# sample_seeds = [42, 43]
encode_ratios = [10., 5.]
repeat_panalities = [1.5]
deltas = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]
new_deltas = [1.0, 1.2, 1.5, 2.0, 3.0]

# new_deltas = [1.5, 1.8, 2.5]
new_encode_ratios = [10.]
# deltas = [0.9, 1.0, 1.2, 1.5, 2.0]
sample_num = 500


deltas = [1.5]
new_deltas = [1.5]
encode_ratios = [10.]
sample_num = 500

hypers = [
    # # {
    # #     'expr': 'python run_wm_lm.py',
    # #     'model_name': ['facebook/opt-1.3b'],
    # #     'repeat_penalty': repeat_panalities,
    # #     'temperature': [1.0],
    # #     'sample_seed': sample_seeds,
    # #     'num_beams': [4],
    # #     'generated_length': [200],
    # #     'sample_num': [sample_num],
    # #     'lm_prefix_len': [10],
    # #     'lm_top_k': [-1],
    # #     'message_code_len': [20],
    # #     'random_permutation_num': [100],
    # #     'max_confidence_lbd': [0.5],
    # #     'encode_ratio': new_encode_ratios,
    # #     'delta': new_deltas,
    # #     'message_model_strategy': ['vanilla'],
    # #     'lm_model_name': ['gpt2'],
    # #     'sub_replacement_proportion': [0.05,0.1],
    # #     'sub_normalize': [True],
    # # },
    # # {
    # #     'expr': 'python run_wm_lm_no_hash.py',
    # #     'model_name': ['facebook/opt-1.3b'],
    # #     'repeat_penalty': repeat_panalities,
    # #     'temperature': [1.0],
    # #     'sample_seed': sample_seeds,
    # #     'num_beams': [4],
    # #     'generated_length': [200],
    # #     'sample_num': [sample_num],
    # #     'lm_prefix_len': [10],
    # #     'lm_top_k': [-1],
    # #     'message_code_len': [20],
    # #     'random_permutation_num': [100],
    # #     'max_confidence_lbd': [0.5],
    # #     'encode_ratio': new_encode_ratios,
    # #     'delta': new_deltas,
    # #     'message_model_strategy': ['vanilla'],
    # #     'lm_model_name': ['gpt2']
    # # },
    # {
    #     'expr': 'python run_wm_random.py',
    #     'model_name': ['facebook/opt-1.3b'],
    #     'repeat_penalty': repeat_panalities,
    #     'temperature': [1.0],
    #     'sample_seed': sample_seeds,
    #     'num_beams': [4],
    #     'generated_length': [200],
    #     'sample_num': [sample_num],
    #     'top_k': [1000],
    #     'message_code_len': [20],
    #     'encode_ratio': new_encode_ratios,
    #     'delta': deltas,
    #     'lm_model_name': ['gpt2'],
    #     # 'sub_replacement_proportion': [0.05,0.1]
    # },
    # # {
    # #     'expr': 'python run_wm_random_no_hash.py',
    # #     'model_name': ['facebook/opt-1.3b'],
    # #     'repeat_penalty': repeat_panalities,
    # #     'temperature': [1.0],
    # #     'sample_seed': sample_seeds,
    # #     'num_beams': [4],
    # #     'generated_length': [200],
    # #     'sample_num': [sample_num],
    # #     'top_k': [1000],
    # #     'message_code_len': [20],
    # #     'encode_ratio': new_encode_ratios,
    # #     'delta': new_deltas,
    # #     'lm_model_name': ['gpt2'],
    # # },
    # # {
    # #     'expr': 'python run_wm_none.py',
    # #     'model_name': ['facebook/opt-1.3b'],
    # #     'repeat_penalty': repeat_panalities,
    # #     'temperature': [1.0],
    # #     'sample_seed': sample_seeds,
    # #     'num_beams': [4],
    # #     'generated_length': [200],
    # #     'sample_num': [sample_num],
    # # },
    # # {
    # #     'expr': 'python run_wm_lm_sub_attack.py',
    # #     'model_name': ['facebook/opt-1.3b'],
    # #     'repeat_penalty': repeat_panalities,
    # #     'temperature': [1.0],
    # #     'sample_seed': sample_seeds,
    # #     'num_beams': [4],
    # #     'generated_length': [200],
    # #     'sample_num': [sample_num],
    # #     # 'sample_num': [50],
    # #     'lm_prefix_len': [10],
    # #     'lm_top_k': [-1],
    # #     'message_code_len': [20],
    # #     'random_permutation_num': [100],
    # #     'max_confidence_lbd': [0.5],
    # #     'encode_ratio': [10.],
    # #     'delta': new_deltas,
    # #     'message_model_strategy': ['vanilla'],
    # #     'lm_model_name': ['gpt2']
    # # },
    # # {
    # #     'expr': 'python run_wm_lm_cp_attack.py',
    # #     'model_name': ['facebook/opt-1.3b'],
    # #     'repeat_penalty': repeat_panalities,
    # #     'temperature': [1.0],
    # #     'sample_seed': sample_seeds,
    # #     'num_beams': [4],
    # #     'generated_length': [200],
    # #     'sample_num': [sample_num],
    # #     # 'sample_num': [50],
    # #     'lm_prefix_len': [10],
    # #     'lm_top_k': [-1],
    # #     'message_code_len': [20],
    # #     'random_permutation_num': [100],
    # #     'max_confidence_lbd': [0.5],
    # #     'encode_ratio': [10.],
    # #     'delta': list(set(deltas) - set(new_deltas)),
    # #     'message_model_strategy': ['vanilla'],
    # #     'lm_model_name': ['gpt2'],
    # #     'cover_old': [True]
    # # },
    # # {
    # #     'expr': 'python run_wm_lm_sub_attack.py',
    # #     'model_name': ['facebook/opt-1.3b'],
    # #     'repeat_penalty': repeat_panalities,
    # #     'temperature': [1.0],
    # #     'sample_seed': sample_seeds,
    # #     'num_beams': [4],
    # #     'generated_length': [200],
    # #     'sample_num': [sample_num],
    # #     # 'sample_num': [50],
    # #     'lm_prefix_len': [10],
    # #     'lm_top_k': [-1],
    # #     'message_code_len': [20],
    # #     'random_permutation_num': [100],
    # #     'max_confidence_lbd': [0.5],
    # #     'encode_ratio': [10.],
    # #     'delta': deltas,
    # #     'message_model_strategy': ['vanilla'],
    # #     'lm_model_name': ['gpt2'],
    # # },
    # {
    #     'expr': 'python run_wm_lm.py',
    #     'model_name': ['facebook/opt-1.3b'],
    #     'repeat_penalty': repeat_panalities,
    #     'temperature': [1.0],
    #     'sample_seed': sample_seeds,
    #     'num_beams': [4],
    #     'generated_length': [200],
    #     'sample_num': [sample_num],
    #     'lm_prefix_len': [10],
    #     'lm_top_k': [-1],
    #     'message_code_len': [20],
    #     'random_permutation_num': [50,150,250,25],
    #     'max_confidence_lbd': [0.5],
    #     'encode_ratio': new_encode_ratios,
    #     'delta': deltas,
    #     'message_model_strategy': ['vanilla'],
    #     'lm_model_name': ['gpt2']
    # },
    # {
    #     'expr': 'python run_wm_lm.py',
    #     'model_name': ['facebook/opt-1.3b'],
    #     'repeat_penalty': repeat_panalities,
    #     'temperature': [1.0],
    #     'sample_seed': sample_seeds,
    #     'num_beams': [4],
    #     'generated_length': [210],
    #     'sample_num': [sample_num],
    #     'lm_prefix_len': [10],
    #     'lm_top_k': [-1],
    #     'message_code_len': [20, 10, 5, 7],
    #     # 'message_code_len': [5],
    #     'random_permutation_num': [100],
    #     'max_confidence_lbd': [0.5],
    #     'encode_ratio': new_encode_ratios,
    #     'delta': deltas,
    #     # 'delta': [1.0,1.2],
    #     'message_model_strategy': ['vanilla'],
    #     'lm_model_name': ['gpt2']
    # },
    # {
    #     'expr': 'python run_wm_lm.py',
    #     'model_name': ['facebook/opt-1.3b'],
    #     'repeat_penalty': repeat_panalities,
    #     'temperature': [1.0],
    #     'sample_seed': sample_seeds,
    #     'num_beams': [4],
    #     'generated_length': [200],
    #     'sample_num': [sample_num],
    #     'lm_prefix_len': [5, 7, 15, 20],
    #     'lm_top_k': [-1],
    #     'message_code_len': [20],
    #     'random_permutation_num': [100],
    #     'max_confidence_lbd': [0.5],
    #     'encode_ratio': new_encode_ratios,
    #     'delta': new_deltas,
    #     'message_model_strategy': ['vanilla'],
    #     'lm_model_name': ['gpt2']
    # },
    {
        'expr': 'python run_wm_lm.py',
        'model_name': ['facebook/opt-1.3b'],
        'repeat_penalty': repeat_panalities,
        'temperature': [1.0],
        'sample_seed': sample_seeds,
        'num_beams': [4],
        'generated_length': [200],
        'sample_num': [sample_num],
        'lm_prefix_len': [10],
        'lm_top_k': [-1],
        'message_code_len': [20],
        'random_permutation_num': [100],
        'max_confidence_lbd': [0.5],
        'encode_ratio': encode_ratios,
        'delta': deltas,
        'message_model_strategy': ['vanilla'],
        'lm_model_name': ['gpt2-large', 'gpt2-medium_', 'gpt2-xl']
    },
]
run = Sh_run(hyperparameter_lists=hypers,
             gpu_list=[0,1,2,3,4,5])

run.run()

print(f'assigned tasks on gpus {run.gpu_list}')

print(f'running tasks {run.experiment_strs}')
