python eval_llada_steering.py \
    --csv_path JBB \
    --attack_method zeroshot \
    --self_reminder False \
    --model_path GSAI-ML/LLaDA-8B-Instruct \
    --generated_samples_path ./outputs/LLaDA_zeroshot \
    --steering_vector_path steering_vectors2.pt \
    --target_layer 31 \
    --sampling_steps 128 \
    --mask_length 128 \
    --block_size 128 \
    --sampler steering \
    --steering_overshoot 1.0 \
    --initial_steering_ratio 0.1 \
    --max_refinement_iters 5 \
    --gen_batch_size 2 \
    --device cuda:0
# gen_batch_size 说明（2026-07-03 GPU 验收实测，docs/batch_inference_gpu_acceptance_log.md）：
# 本机 V100(bf16 无 tensor core) 上 bs=2 是唯一正收益点(1.11x)，bs=4 1.06x、bs=8 反而更慢；
# 需要与历史产物逐位可比时用 1；A100/H100 或 fp16 场景可开 4–8。
