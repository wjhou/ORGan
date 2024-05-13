#!/bin/sh
export TOKENIZERS_PARALLELISM=true
warmup_ratio=0.0
max_tgt_length=16
num_train_epochs=15
overwrite_output_dir=false
evaluation_strategy=epoch
per_device_train_batch_size=32
per_device_eval_batch_size=64
gradient_accumulation_steps=1
debug_model=false
seed=42
weight_decay=0.0
num_beams=4
slow_lr=5e-5
fast_lr=1e-4
alpha=0.5
log_level="info"
output_dir="./tmp/iu_xray_planner_epoch${num_train_epochs}"
checkpoint_name=$2

if [ "$1" -ne 1 ];
then
    echo "********** debug **********"
    echo "********** debug **********"
    echo "********** debug **********"
    num_train_epochs=1
    output_dir="./tmp/bert_doc_baseline_debug"
    overwrite_output_dir=true
    debug_model=true
fi

python3 -u ./src_plan/run_ende.py \
    --chexpert_model_name_or_path $checkpoint_name \
    --annotation_file "./data/iu_xray/annotation.json" \
    --image_path "./data/iu_xray/images/" \
    --id2tagpos_path "./data/iu_xray_id2tagpos.json" \
    --tag_path "./data/iu_xray_id2tag.csv" \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --alpha $alpha \
    --max_tgt_length $max_tgt_length \
    --output_dir $output_dir \
    --warmup_ratio $warmup_ratio \
    --num_train_epochs $num_train_epochs \
    --learning_rate $slow_lr \
    --weight_decay $weight_decay \
    --fast_lr $fast_lr \
    --evaluation_strategy $evaluation_strategy \
    --save_strategy $evaluation_strategy \
    --save_total_limit 1 \
    --seed $seed \
    --logging_steps 100 \
    --fp16 \
    --fp16_opt_level O2 \
    --fp16_full_eval \
    --dataloader_num_workers 8 \
    --load_best_model_at_end true \
    --overwrite_output_dir $overwrite_output_dir \
    --eval_on_gen \
    --greater_is_better true \
    --metric_for_best_model eval_micro_abn_prf \
    --debug_model $debug_model \
    --num_beams $num_beams
