# Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models


> **Adaptive Steering and Remasking** proposes a training-free safety framework that prevents jailbreak attacks in diffusion language models by steering harmful generation trajectories during the denoising process.

![overview](./assets/overview_figure.png)

We proposes a training-free safety framework for diffusion language models that combines adaptive semantic steering and harmful token remasking during the denoising process.  

The method first constructs a **Contrastive Safety Direction (CSD)** to distinguish harmful and safe semantic representations, and applies **adaptive steering** in the early denoising stages to guide generation toward safer trajectories.  

It then performs **selective token remasking** to regenerate potentially harmful tokens, effectively reducing jailbreak attacks while preserving the fluency and overall quality of generated responses.

## 📌 Performance of  

## 🛠️ Setup
```bash
$ conda create -n dlm_steering python=3.10
$ conda activate dlm_steering
$ pip install -r requirements.txt
$ mkdir outputs
```
## 🚀 Usage
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
$ sh scripts/llama_guard.sh             # llama_guard (JailBreakBench, AdvBench)
$ sh scripts/test_rouge_score.sh        # rouge_score (TruthfulQA)
$ sh scripts/mmlu_eval.sh               # accuracy (MMLU)
$ sh scripts/math-500_eval.sh           # accuracy (MATH-500)
```


---
### Additional Information
Our code is based on the code from [LLaDA](https://github.com/guanghanwang/ReMDM-LLaDA), [Dream](https://github.com/DreamLM/Dream), and [ReMDM-LLaDA](https://github.com/guanghanwang/ReMDM-LLaDA).
