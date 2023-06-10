# from https://github.com/shizhediao/T-DNA/blob/main/TDNA/pmi_ngram.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tokenizer import Tokenizer
import argparse
import json
import math
import re
from collections import defaultdict

import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm


class FindNgrams:
    def __init__(self, min_count=0, min_pmi=0, language="en"):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.words = defaultdict(int)
        self.ngrams, self.pairs = defaultdict(int), defaultdict(int)
        self.total = 0.0
        self.language = language

    def text_filter(self, sentence):
        cleaned_text = []
        index = 0
        for i, w in enumerate(sentence):
            if re.match("[^\u0600-\u06FF\u0750-\u077F\u4e00-\u9fa50-9a-zA-Z]+", w):
                if i > index:
                    cleaned_text.append([w.lower() for w in sentence[index:i]])
                index = 1 + i
        if index < len(sentence):
            cleaned_text.append([w.lower() for w in sentence[index:]])
        return cleaned_text

    def count_ngram(self, texts, n):
        self.ngrams = defaultdict(int)
        for sentence in texts:
            sub_sentence = sentence.split()
            for i in range(n):
                n_len = i + 1
                for j in range(len(sub_sentence) - i):
                    ngram = tuple([w for w in sub_sentence[j : j + n_len]])
                    self.ngrams[ngram] += 1
        self.ngrams = {i: j for i, j in self.ngrams.items() if j > self.min_count}

    def find_ngrams_pmi(self, texts, n, freq_threshold, freq_threshold2):
        for sentence in tqdm(texts, desc="Processing"):
            sub_sentence = sentence.split()
            self.words[sub_sentence[0]] += 1
            for i in range(len(sub_sentence) - 1):
                self.words[sub_sentence[i + 1]] += 1
                self.pairs[(sub_sentence[i], sub_sentence[i + 1])] += 1
                self.total += 1
        self.words = {i: j for i, j in self.words.items() if j > self.min_count}
        self.pairs = {i: j for i, j in self.pairs.items() if j > self.min_count}

        min_mi = math.inf
        max_mi = -math.inf

        self.strong_segments = set()
        for i, j in self.pairs.items():
            if i[0] in self.words and i[1] in self.words:
                mi = math.log(self.total * j / (self.words[i[0]] * self.words[i[1]]))
                if mi > max_mi:
                    max_mi = mi
                if mi < min_mi:
                    min_mi = mi
                if mi >= self.min_pmi:
                    self.strong_segments.add(i)

        self.ngrams = defaultdict(int)
        for sentence in texts:
            sub_sentence = sentence.split()
            s = [sub_sentence[0]]
            for i in range(len(sub_sentence) - 1):
                if (sub_sentence[i], sub_sentence[i + 1]) in self.strong_segments:
                    s.append(sub_sentence[i + 1])
                else:
                    self.ngrams[tuple(s)] += 1
                    s = [sub_sentence[i + 1]]

        self.ngrams = {
            i: j for i, j in self.ngrams.items() if j > self.min_count and len(i) <= n
        }
        self.renew_ngram_by_freq(texts, freq_threshold, freq_threshold2, n)

    def renew_ngram_by_freq(self, all_sentences, min_feq, max_feq, ngram_len=10):
        new_ngram2count = {}
        new_all_sentences = []

        for sentence in all_sentences:
            sentence = sentence.split()
            sen = sentence
            for i in range(len(sen)):
                for n in range(1, ngram_len + 1):
                    if i + n > len(sentence):
                        break
                    n_gram = tuple(sentence[i : i + n])
                    if n_gram not in self.ngrams:
                        continue
                    if n_gram not in new_ngram2count:
                        new_ngram2count[n_gram] = 1
                    else:
                        new_ngram2count[n_gram] += 1
        print(min_feq, max_feq)
        self.ngrams = {
            gram: c
            # for gram, c in new_ngram2count.items() if c > min_feq and c < max_feq
            for gram, c in new_ngram2count.items()
            if min_feq < c < max_feq
        }


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="the name of dataset")
parser.add_argument("--output_dir", type=str, required=True, help="the output path")
parser.add_argument("--ngram", type=int, default=5, help="n")
parser.add_argument("--min_count", type=int, default=5, help="min_count")
parser.add_argument("--min_pmi", type=int, default=1, help="min_pmi")
parser.add_argument(
    "--ngram_freq_threshold", type=int, default=5, help="ngram_freq_threshold"
)
parser.add_argument(
    "--ngram_freq_threshold2", type=int, default=10000000, help="ngram_freq_threshold"
)
parser.add_argument(
    "--delete_special_symbol",
    action="store_false",
    help="Whether to remove special symbols",
)
config = parser.parse_args()

ngram_list = []
dataset = config.dataset
ngram = config.ngram
min_count = config.min_count
min_pmi = config.min_pmi
ngram_freq_threshold = config.ngram_freq_threshold
ngram_freq_threshold2 = config.ngram_freq_threshold2

print("dataset: ", dataset)
if config.dataset == "mimic_cxr" or config.dataset == "mimic_abn":
    clean_fn = Tokenizer.clean_report_mimic_cxr
else:
    clean_fn = Tokenizer.clean_report_iu_xray
print(clean_fn)
with open("../%s/annotation.json" % config.dataset, "r", encoding="utf-8") as f:
    f_read = json.load(f)["train"]

sentence_list = []
reports = []
words = []
for report in tqdm(f_read, desc="Loading"):
    report = clean_fn(report["report"])
    if len(report) == 0:
        continue
    reports.append(report)
    sentence_list.extend([z.strip() for z in report.split(".") if len(z.strip()) > 0])
    words.append(report.split())
    # sentence_list.extend(
    #     [sent.strip() for sent in report.split(".") if len(sent.strip()) > 0])

ngram_finder = FindNgrams(min_count=min_count, min_pmi=min_pmi)
ngram_finder.find_ngrams_pmi(
    sentence_list, ngram, ngram_freq_threshold, ngram_freq_threshold2
)

ngram_type_count = [0 for _ in range(ngram)]
ngram_finder.ngrams = dict(
    sorted(
        ngram_finder.ngrams.items(),
        key=lambda kv: (kv[1], kv[0]),
        reverse=True,
    )
)  # sort

count = 0
for w, c in ngram_finder.ngrams.items():
    count += 1
    s = ""
    for word_index in range(len(w)):
        s += w[word_index] + " "
    s = s.strip()
    i = len(s)
    if config.delete_special_symbol:
        while i > 0:
            if s[i - 1].isalnum():
                break
            i -= 1
    s = s[0:i]
    if s not in ngram_list and len(s) > 0:
        if s not in ngram_list:
            ngram_list.append(s)


def find_punc(units, punc):
    for tok in units:
        if tok in punc:
            return True
    return False


def find_stopwords(units, stopwords):
    count = 0
    for unit in units:
        if unit in stopwords:
            count += 1
    return count / len(units) >= 0.5


def strip_stopwords(units, stopwords):
    start = 0
    while start < len(units):
        if units[start] not in stopwords:
            break
        start += 1
    end = len(units) - 1
    while end > start:
        if units[end] not in stopwords:
            break
        end -= 1
    return units[start : end + 1]


ngram_count = 0
stopwords = stopwords.words("english")
stopwords.append("xx")
punc = set(".,?;*!%^&_+():-\[\]\{\}")
# punc.update(stopwords)
numbers = set("0123456789")
ngram_stat = {}

for ngram_phrase in tqdm(ngram_list, desc="Filtering"):
    ngram_count += 1
    units = ngram_phrase.split()
    units = strip_stopwords(units, stopwords)
    if (
        len(units) <= 0
        or find_stopwords(units, stopwords)
        or len(numbers & set(ngram_phrase)) > 0
        or find_punc(units, punc)
    ):
        continue
    ngram_phrase = " ".join(units)
    if ngram_phrase in ngram_stat:
        continue
    print(ngram_phrase)
    query = tuple(ngram_phrase.split())
    if query not in ngram_finder.ngrams:
        continue
    ngram_stat[ngram_phrase] = ngram_finder.ngrams[query]
    ngram_type_count[len(list(ngram_phrase.split())) - 1] += 1

sorted_ngrams = sorted(ngram_stat.items(), key=lambda x: -x[1])
df = pd.DataFrame(
    data={
        "ngram": [x[0] for x in sorted_ngrams],
        "count": [x[1] for x in sorted_ngrams],
    }
)
df.to_csv(config.output_dir, index=False)