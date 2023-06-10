import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import json

from tokenizer import Tokenizer
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="the output path")
parser.add_argument("--filter_pmi_dir", type=str, required=True, help="the output path")
parser.add_argument("--id2ngram_dir", type=str, required=True, help="the output path")
config = parser.parse_args()

with open(config.filter_pmi_dir, "r", encoding="utf-8") as f:
    mention2ngram = json.load(f)
ngram_collection = set()
for m in mention2ngram:
    ngram_collection.update(mention2ngram[m])

print("dataset: ", config.dataset)
if config.dataset == "mimic_cxr" or config.dataset == "mimic_abn":
    clean_fn = Tokenizer.clean_report_mimic_cxr
else:
    clean_fn = Tokenizer.clean_report_iu_xray
print(clean_fn)
with open("../%s/annotation.json" % config.dataset, "r", encoding="utf-8") as f:
    f_read = json.load(f)  # ["train"]

reports = {}
for split in f_read:
    for report in tqdm(f_read[split], desc="Loading Reports"):
        text = clean_fn(report["report"])
        if len(text) == 0:
            continue
        reports[report["id"]] = text


def sublist_count(l, sl):
    x = 0
    for i in range(len(l)):
        if l[i : i + len(sl)] == sl:
            x += 1
    return x


id2ngram = {}
for idx in tqdm(reports, desc="Labeling Reports"):
    text = reports[idx].split()
    id2ngram[idx] = set()
    for ngram in ngram_collection:
        units = ngram.split()
        if sublist_count(text, units) > 0:
            id2ngram[idx].add(ngram)

for idx in id2ngram:
    id2ngram[idx] = list(id2ngram[idx])

with open(config.id2ngram_dir, "w", encoding="utf-8") as f:
    json.dump(id2ngram, f, ensure_ascii=False, indent=4)
