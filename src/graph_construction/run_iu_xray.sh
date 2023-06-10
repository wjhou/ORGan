dataset="iu_xray"
ngram_dir="./data/iu_xray_ngram.txt"
chexbert_dir="../CheXbert/src/data/iu_xray/id2tag.csv"
pmi_dir="./data/iu_xray_pmi.json"
filter_pmi_dir="./data/iu_xray_filter_pmi_test.json"
id2ngram_dir="./data/iu_xray_id2ngram.json"
pycx_dir="./data/iu_xray_pycx.json"
checkpoint_path="../CheXbert/chexbert.pth"

python src/graph_construction/pmi_ngram.py \
    --dataset $dataset \
    --output_dir $ngram_dir \
    --ngram 4 \
    --min_count 3 \
    --ngram_freq_threshold 50

python src/graph_construction/pmi_mention_ngram.py \
    --dataset $dataset \
    --ngram_dir $ngram_dir \
    --chexbert_dir $chexbert_dir \
    --output_dir $pmi_dir \
    --pmi_threshold 0.0 \
    --pycx_dir $filter_pmi_dir

python src/graph_construction/filter_ngram.py \
    --pmi_dir $pmi_dir \
    --filter_pmi_dir $filter_pmi_dir \
    --ngram_dir $ngram_dir \
    --topk 10000

python src/graph_construction/generate_ngram_label.py \
    --dataset $dataset \
    --filter_pmi_dir $filter_pmi_dir \
    --id2ngram_dir $id2ngram_dir
