# DAPO Open-Source Implementation

> Open-Source Algorithm Implementation & Expriements Running: [Yuxuan Tong](https://tongyx361.github.io/), [Guangming Sheng](https://hk.linkedin.com/in/guangming-sheng-b50640211)

🏠 [Homepage](https://dapo-sia.github.io/) | 📝 [Paper](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) | 🤗 [Datasets&Models@HF](https://huggingface.co/collections/BytedTsinghua-SIA/dapo-67d7f1517ee33c8aed059da0) | 🐱 [Code@GitHub](https://github.com/volcengine/verl/tree/gm-tyx/puffin/main/recipe/dapo)

> We propose the **D**ecoupled Clip and Dynamic s**A**mpling **P**olicy **O**ptimization (DAPO) algorithm. By making our work publicly available, we provide the broader research community and society with practical access to scalable reinforcement learning, enabling all to benefit from these advancements. Our system is based on the awesome [verl](https://github.com/volcengine/verl) framework. Thanks for their great work! Applying DAPO training to Qwen2.5-32B base model proves to outperform the previous state-of-the-art DeepSeek-R1-Zero-Qwen-32B on AIME 2024, achieving **50%** accuracy with **50%** less training steps.
>
> ![dapo-main-result](https://dapo-sia.github.io/static/images/score.png)

## Quickstart

1. Prepare the datasets **on the Ray cluster**:

```bash
bash prepare_dapo_data.sh # This downloads the datasets to ${HOME}/verl/data by default
```

2. Submit the job to the Ray cluster **from any machine**:

```bash
cd verl # Repo root
export RAY_ADDRESS="http://${RAY_IP:-localhost}:8265" # The Ray cluster address to connect to
export WORKING_DIR="${PWD}" # The local directory to package to the Ray cluster
# Set the runtime environment like env vars and pip packages for the Ray cluster in yaml
export RUNTIME_ENV="./verl/trainer/runtime_env.yaml"
bash recipe/dapo/run_dapo_qwen2.5_32b.sh
```

## Reproduction Runs

### DAPO w/o Token-level PG loss & Dynamic Sampling -> 44% on AIME 2024

We achieved 44% accuracy on AIME 2024 with a early version of DAPO w/o Token-level PG Loss & Dynamic Sampling.

The training record will be available on WandB soon.

The corresponding training script is [run_dapo_early_qwen2.5_32b.sh](./run_dapo_early_qwen2.5_32b.sh).

### DAPO -> 50% on AIME 2024

The training record will be available on WandB soon.

The corresponding training script is [run_dapo_qwen2.5_32b.sh](./run_dapo_qwen2.5_32b.sh).

## Configuration

### Separated Clip Episilons (-> Clip-Higher)

An example configuration:

```yaml
actor_rollout_ref:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
```

`clip_ratio_low` and `clip_ratio_high` specify the $\varepsilon_{\text {low }}$ and $\varepsilon_{\text {high }}$ in the DAPO objective.

Core relevant code:

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

### Dynamic Sampling (with Group Filtering)

An example configuration:

```yaml
data:
  gen_batch_size: 1536
  train_batch_size: 512
algorithm:
  filter_groups:
    enable: True
    metric: acc # score / seq_reward / seq_final_reward / ...
    max_num_gen_batches: 10 # Non-positive values mean no upper limit
```

Setting `filter_groups.enable` to `True` will filter out groups whose outputs' `metric` are all the same, e.g., for `acc`, groups whose outputs' accuracies are all 1 or 0.

The trainer will repeat sampling with `gen_batch_size` until there are enough qualified groups for `train_batch_size` or reaching the upper limit specified by `max_num_gen_batches`.

Core relevant code:

```python
prompt_bsz = self.config.data.train_batch_size
if num_prompt_in_batch < prompt_bsz:
    print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
    num_gen_batches += 1
    max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        print(f'{num_gen_batches=} < {max_num_gen_batches=}. Keep generating...')
        continue
    else:
        raise ValueError(
            f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
        )
else:
    # Align the batch
    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    batch = batch[:traj_bsz]
```

### Token-level Policy Gradient Loss

An example configuration:

```yaml
actor_rollout_ref:
  actor:
    use_token_level_loss: True
```

Setting `use_token_level_loss` to `True` will mean the policy gradient loss across all the tokens in all the sequences in a batch.

Core relevant code:

```python
if use_token_level_loss:
    pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
else:
    pg_loss = torch.sum(pg_losses * eos_mask, dim=1) / seq_len_per_sample
    pg_loss = torch.mean(pg_loss)
```

### Overlong Reward Shaping

An example configuration:

```yaml
data:
  max_response_length: 20480 # 16384 + 4096
custom_reward_function:
  overlong_buffer: 
    enable: True
    len: 4096
    penalty_factor: 1.0
```

Setting `overlong_buffer.enable` to `True` will penalize the outputs whose length entering the last `overlong_buffer.len` tokens before the `max_response_length`.

The penalty increases linearly from 0 to `overlong_buffer.penalty_factor`.

Core relevant code:

```python
if self.overlong_buffer_cfg.enable:
    overlong_buffer_len = self.overlong_buffer_cfg.len
    expected_len = self.max_resp_len - overlong_buffer_len
    exceed_len = valid_response_length - expected_len
    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    final_reward += overlong_reward
```