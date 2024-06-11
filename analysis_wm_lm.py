from watermarking.wm_lm import WmLMArgs
from watermarking.arg_classes.wm_arg_class import WmAnalysisArgs
from transformers import HfArgumentParser
from watermarking.wm_analysis import main

oracle_model_name = 'facebook/opt-2.7b'
parser = HfArgumentParser((WmLMArgs,))
args: WmLMArgs
args, = parser.parse_args_into_dataclasses()
# save_file_path = args.complete_save_file_path
save_file_path = "my_watermark_result/lm_new_7_10/facebook-opt-1.3b_1.5_1.5_300_200_100_42_42_20_10.0_4_1.0_10_-1_100_vanilla_0.5_gpt2_1000.json"

analysis_args = WmAnalysisArgs(analysis_file_name=save_file_path, device=args.device,
                      oracle_model_name=oracle_model_name, args=args)
main(analysis_args)
