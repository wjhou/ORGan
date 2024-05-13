#!/bin/sh
export TOKENIZERS_PARALLELISM=true
warmup_ratio=0.0
max_tgt_length=64
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
topk=30
beta=4
trr=1
outline_level=3
log_level="info"
output_dir="./tmp/mimic_cxr_oragn_epoch${num_train_epoch}_ngram${topk}_bs${per_device_train_batch}_tgt${max_tgt_length}_$seed"
chexpert_model_name_or_path=$2
plan_model_name_or_path=$3
plan_eval_file=$4

if [ "$1" -ne 1 ];
then
    echo "********** debug **********"
    echo "********** debug **********"
    echo "********** debug **********"
    suffix="_debug"
    num_train_epochs=1
    output_dir="./tmp/bert_doc_baseline_debug"
    overwrite_output_dir=true
    debug_model=true
fi

python3 -u ./src/run_ende.py \
    --plan_model_name_or_path $plan_model_name_or_path \
    --plan_eval_file $plan_eval_file \
    --annotation_file "./data/mimic_cxr/annotation.json" \
    --node_file "./data/mimic_cxr_filter_pmi.json" \
    --image_path "./data/mimic_cxr/images/" \
    --tag_path "./data/mimic_cxr_id2tag.csv" \
    --do_train \
    --do_eval \
    --do_predict \
    --trr $trr \
    --topk $topk \
    --outline_level $outline_level \
    --beta $beta \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
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
    --metric_for_best_model eval_BLEU_4 \
    --debug_model $debug_model \
    --num_beams $num_beams
