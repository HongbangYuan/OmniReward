FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 PYTHONPATH=./ WANDB_DISABLED=true \
python -m torch.distributed.launch --nproc_per_node=4 --use-env src/train.py \
--model_name_or_path openbmb/MiniCPM-o-2_6 --trust_remote_code \
--freeze_vision_tower --freeze_multi_modal_projector --train_mm_proj_only False \
--flash_attn fa2 --stage rm --do_train --finetuning_type full \
--deepspeed examples/deepspeed/ds_z2_config.json \
--dataset hpdv2,evalmuse,omni_t2i_hpd,omni_t2i_oip,videodpo,visionreward \
--template minicpm_o --cutoff_len 18000 --overwrite_cache true --preprocessing_num_workers 32 \
--output_dir saves/minicpm_o/omni_t2iv --logging_steps 5 \
--save_steps 400 --plot_loss --overwrite_output_dir --save_total_limit 20 \
--per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 2.0e-6 \
--num_train_epochs 2.0 --lr_scheduler_type cosine --warmup_ratio 0.05 --weight_decay 1.0e-3 \
--bf16 --ddp_timeout 180000000 --per_device_eval_batch_size 1 --eval_strategy steps \
--eval_steps 100 --do_predict --do_eval --eval_dataset vl_rewardbench --max_samples 50000 \
--video_maxlen 4 --save_only_model