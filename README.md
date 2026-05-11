# Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models

![overview](./assets/overview_figure.png)

> We propose a 

## Setup
```bash
$ conda create -n dlm_steering python=3.10
$ conda activate dlm_steering
$ pip install -r requirements.txt
$ mkdir outputs
```
## Usage
### Making Contrastive Safety Direction
```bash
$ python make_csd_llada.py
$ python make_csd_dream.py
```

### Inference with LLaDA
```bash
$ sh scripts/llada_steer.sh
```
### Inference with Dream
```bash
$ sh scripts/dream_steer.sh
```
### Validation
```bash
$ sh scripts/llama_guard.sh             #llama guard (JailBreakBench, AdvBench)
$ sh scripts/test_rouge_score.sh        # rouge score (TruthfulQA)
$ sh scripts/mmlu_eval.sh               # accuracy (MMLU)
$ sh scripts/math-500_eval.sh           # accuracy (MATH-500)
```


---
### Additional Information
Our code is based on the code from [LLaDA](https://github.com/guanghanwang/ReMDM-LLaDA), [Dream](https://github.com/DreamLM/Dream), and [ReMDM-LLaDA](https://github.com/guanghanwang/ReMDM-LLaDA).
