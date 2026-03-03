set -x
ENGINE=${1:-vllm}
n_gpu=8


train_data_size=16
val_data_size=128
group_size=8


num_attempts=3
val_only=False
val_opponent_policy="cfr"
val_play_mode="second"
mcts_sim_num=100
play_mode="mixed"
second_player_ratio=0.5
use_length_penalty=True
length_penalty_coef=2.0
max_response_length=8192


val_before_train=False
version="mage"
mode="mean_norm" # "mean_norm" or "mean_std_norm"
refletion_type="reflection_only" # "reflection_only" or "reflection_and_history" or "history_only"
model_path=/storage/openpsi/models/Qwen__Qwen3-4B
project_name='mage'
experiment_name=kuhnpoker_${version}_gigpo_qwen3_4b
log_name=$experiment_name


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=/storage/openpsi/users/yl/LaMer/data/verl-agent/text/train.parquet \
    data.val_files=/storage/openpsi/users/yl/LaMer/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.model.enable_thinking=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.seed=20 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=2.0 \
    +actor_rollout_ref.actor.use_length_penalty=$use_length_penalty \
    +actor_rollout_ref.actor.length_penalty_coef=$length_penalty_coef \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    reward_model.reward_manager=episode \
    env.env_name="KuhnPoker" \
    env.seed=0 \
    env.rollout.n=$group_size \
    env.num_attempts=$num_attempts \
    env.max_steps=20 \
    env.max_turns=6 \
    env.kuhnpoker.mcts_sim_num=$mcts_sim_num \
    env.kuhnpoker.play_mode=$play_mode \
    +env.kuhnpoker.val_play_mode=$val_play_mode \
    +env.kuhnpoker.val_opponent_policy=$val_opponent_policy \
    env.kuhnpoker.second_player_ratio=$second_player_ratio \
    +env.reflection_type=$refletion_type \
    +env.multipolicy=True \
    trainer.val_only=$val_only \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=$val_before_train \
    trainer.log_val_generations=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    2>&1 | tee -a ./logs/${log_name}.log

