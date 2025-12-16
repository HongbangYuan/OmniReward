FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 PYTHONPATH=./ WANDB_DISABLED=true \
python -m torch.distributed.launch --nproc_per_node=4 --use-env src/train.py \
--model_name_or_path jinzhuoran/OmniRewardModel --trust_remote_code \
--freeze_vision_tower --freeze_multi_modal_projector --train_mm_proj_only False \
--flash_attn fa2 --stage rm --finetuning_type full \
--deepspeed examples/deepspeed/ds_z2_config.json \
--dataset skywork_reward \
--template minicpm_o --cutoff_len 18000 --overwrite_cache true --preprocessing_num_workers 32 \
--output_dir saves/minicpm_o/eval --logging_steps 5 \
--save_steps 400 --plot_loss --overwrite_output_dir --save_total_limit 20 \
--per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2.0e-6 \
--num_train_epochs 2.0 --lr_scheduler_type cosine --warmup_ratio 0.05 --weight_decay 1.0e-3 \
--bf16 --ddp_timeout 180000000 --per_device_eval_batch_size 1 --eval_strategy steps \
--eval_steps 100 --do_predict --do_eval --eval_dataset omni_t2t --max_samples 1000 \
--video_maxlen 8 --save_only_model --eval_tie