## README

### Preparation

Requirement: transformers==4.28, torch==2.3.1, numpy



##### datasets

Please download c4-train.00000-of-00512_sliced to ./c4-train.00000-of-00512_sliced



##### models

Download facebook/opt-1.3b(2.7b) to ./llm-ckpts/opt-1.3b or ./llm-ckpts/opt-1.3b

Download gpt2 to ./llm-ckpts/gpt2

(You can ignore this if your network is fine and transformers.AutoModel/AutoTokenier can successfully download from internet)



### Usage

See demo at wm.ipynb



To run experiments, use experiment.py to generate text with watermark and get watermark successfully decoded rate in json format

To cal ppl, use append_analysis.py



For example:

```shell
python experiment.py
python append_analysis.py
sh gpu_sh.sh
```



(See details in watermarking/wm_*.py)

'expr': 'python run_wm_lm.py': watermark with LM

'expr': 'python run_wm_random.py': watermark extended from white list

'expr': 'python run_wm_none.py': no watermark



hypers in python experiment.py:

```
[
	{
		a: [a_1,a_2]
		b: [b_1,b_2]
	}
	{
		a: [a_3, b_3]
	}
]

will run experiment on arguments (a_1,b_1), (a_1,b_2), (a_2,b_1), (a_2,b_2), (a_3,b_3)

do not forget to set gpu_list in 'run = Sh_run(hyperparameter_lists=hypers,gpu_list=[0])''
```



to read results, you can initialize an arg from ./watermarking/arg_classes/wm_arg_class.py and use `load_result` method:

```python
from watermarking.arg_classes.wm_arg_class import WmLMArgs
args = WmLMArgs()
results = args.load_result()
```



### Add changes

See ./watermarking/watermark_processors/message_models/lm_message_model.py for details

You can add a new .py file, inherit this class and make changes on how seeds is generated or how A is chosen.



use `LMMessageModel.cal_log_Ps`  to cal $P(x_{:t},x_{t+1},m)$:

```python
log_Ps = cal_log_Ps(x_prefix, x_cur,messages,lm_predictions)
# log_Ps[i,j,k] = P(x_prefix[i],x_cur[i,j],messages[k])
# lm_predictions[i,j] = P_LM(x_prefix[i],vocab[j])
```

### Results

See current results in cur_new_results_7_10.ipynb

current LM watermark decoding speed 3s/item (100 A sets); generate speed about 10s/item

$(1-\lambda)P(x_{:t},x_t,m) + \lambda (P(x_{:t},x_t,m) - argmax_{m\not=m_0} P(x_{:t},x_t,m_0)$ 
is now avalible with `message_model_strategy`='max_confidence' and 
`max_confidence_lambda`=$\lambda$ 
but I do not check its performance yet.

public lm can be changed to 'gpt2' rather than the same generating model, but the performance 
will be a little worse, also, set random_permutation_num to 50 rather than 100 will slightly hurt 
the performance but speed up the decoding (2x speed)

