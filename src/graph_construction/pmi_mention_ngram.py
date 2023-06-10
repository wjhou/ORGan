import argparse
import json
import math
from collections import defaultdict
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from tqdm import tqdm

from tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="the name of dataset")
parser.add_argument("--ngram_dir", type=str, required=True, help="the output path")
parser.add_argument("--chexbert_dir", type=str, required=True, help="the output path")
parser.add_argument("--output_dir", type=str, required=True, help="the output path")
parser.add_argument(
    "--pmi_threshold", type=float, required=True, help="the output path"
)
parser.add_argument("--pycx_dir", type=str, required=True, help="the output path")
config = parser.parse_args()

print("dataset: ", config.dataset)
if config.dataset == "mimic_cxr" or config.dataset == "mimic_abn":
    clean_fn = Tokenizer.clean_report_mimic_cxr
else:
    clean_fn = Tokenizer.clean_report_iu_xray
print(clean_fn)

with open("../%s/annotation.json" % config.dataset, "r", encoding="utf-8") as f:
    f_read = json.load(f)["train"]

reports = {}
report_collections = set()
for report in tqdm(f_read, desc="Loading"):
    text = clean_fn(report["report"])
    # if text in report_collections:
    #     continue
    reports[report["id"]] = text
    # report_collections.add(text)

mention_ngram_stat = defaultdict(int)
mention_stat = defaultdict(int)

ngram_stat = pd.read_csv(config.ngram_dir)
ngram_stat = {
    a: b for a, b in zip(ngram_stat["ngram"], ngram_stat["count"])
}  # if len(a.split()) > 1}
id2tags, headers = Tokenizer.load_tag2ids(
    config.chexbert_dir,
    need_header=True,
)


def sublist_count(l, sl):
    x = 0
    for i in range(len(l)):
        if l[i : i + len(sl)] == sl:
            x += 1
    return x


for idx in tqdm(reports, desc="Counting"):
    tags = id2tags[idx]
    text = reports[idx]
    all_units = text.split()
    for tag in tags:
        observation = tag + ("_True" if tags[tag] else "_False")
        mention_stat[observation] += 1
        for ngram in ngram_stat:
            units = ngram.split()
            if ngram in text:
                mention_ngram_stat["@".join((observation, ngram))] += sublist_count(
                    l=all_units, sl=units
                )

p_x_norm = sum([x[1] for x in ngram_stat.items()])
p_y_norm = len(reports)

norm = 0
p_xy_norm = defaultdict(int)
for mention_ngram in mention_ngram_stat:
    mention, ngram = mention_ngram.split("@")
    p_xy_norm[mention] += mention_ngram_stat[mention_ngram]
    norm += mention_ngram_stat[mention_ngram]

p_xy = {x[0]: x[1] / p_xy_norm[x[0].split("@")[0]] for x in mention_ngram_stat.items()}
p_x = {x[0]: x[1] / p_x_norm for x in ngram_stat.items()}
p_y = {x[0]: x[1] / p_y_norm for x in mention_stat.items()}
pmi = {}
k = 1
for xy in p_xy:
    y, x = xy.split("@")
    try:
        pmi_xy = math.log(p_xy[xy] ** k / (p_x[x] * p_y[y]), 2)
        if pmi_xy < config.pmi_threshold:
            continue
        pmi[xy] = pmi_xy
    except Exception:
        print("Error", xy, p_xy[xy] ** k, p_x[x], p_y[y])

with open(config.output_dir, "w", encoding="utf-8") as f:
    json.dump(pmi, f, ensure_ascii=False, indent=4)
