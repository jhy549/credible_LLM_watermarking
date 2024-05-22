from watermarking.wm_random_no_hash import main
from watermarking.arg_classes.wm_arg_class import WmRandomNoHashArgs
from transformers import HfArgumentParser

parser = HfArgumentParser((WmRandomNoHashArgs,))
args: WmRandomNoHashArgs
args, = parser.parse_args_into_dataclasses()
main(args)