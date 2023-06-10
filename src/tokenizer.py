import copy
import json
import re
from collections import Counter, defaultdict

import pandas as pd
from transformers.tokenization_utils import PreTrainedTokenizer
import os
import pickle


class Tokenizer:
    def __init__(self, config):
        self.model_input_names = ["image_path"]
        self.padding_side = "right"
        self.ann_path = config.annotation_file
        self.threshold = config.threshold
        self.dataset = config.dataset
        if self.dataset == "iu_xray":
            self.clean_report = Tokenizer.clean_report_iu_xray
        else:
            self.clean_report = Tokenizer.clean_report_mimic_cxr
        print(self.clean_report)
        self.ann = json.loads(open(self.ann_path, "r").read())
        self.token2idx, self.idx2token, self.special_tokens = self.create_vocabulary()
        self.bos_token_id = self.eos_token_id = self.decoder_start_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2

    def create_vocabulary(self):
        total_tokens = []
        for example in self.ann["train"]:
            tokens = self.clean_report(example["report"]).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold]
        vocab.sort()
        special_tokens = ["[CLS]", "[PAD]", "[UNK]"]
        vocab = special_tokens + vocab
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        return token2idx, idx2token, special_tokens[:-1]

    @staticmethod
    def clean_report_iu_xray(report):
        report_cleaner = (
            lambda t: t.replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("1. ", "")
            .replace(". 2. ", ". ")
            .replace(". 3. ", ". ")
            .replace(". 4. ", ". ")
            .replace(". 5. ", ". ")
            .replace(" 2. ", ". ")
            .replace(" 3. ", ". ")
            .replace(" 4. ", ". ")
            .replace(" 5. ", ". ")
            .strip()
            .lower()
            .split(". ")
        )
        sent_cleaner = lambda t: re.sub(
            "[.,?;*!%^&_+():-\[\]{}]",
            "",
            t.replace('"', "")
            .replace("/", "")
            .replace("\\", "")
            .replace("'", "")
            .strip()
            .lower(),
        )
        tokens = [
            sent_cleaner(sent).strip() + " ."
            for sent in report_cleaner(report)
            if len(sent_cleaner(sent).strip()) > 0
        ]
        report = " ".join(tokens)
        return report

    @staticmethod
    def clean_report_mimic_cxr(report):
        report_cleaner = (
            lambda t: t.replace("\n", " ")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("__", "_")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("..", ".")
            .replace("1. ", "")
            .replace(". 2. ", ". ")
            .replace(". 3. ", ". ")
            .replace(". 4. ", ". ")
            .replace(". 5. ", ". ")
            .replace(" 2. ", ". ")
            .replace(" 3. ", ". ")
            .replace(" 4. ", ". ")
            .replace(" 5. ", ". ")
            .strip()
            .lower()
            .split(". ")
        )
        sent_cleaner = lambda t: re.sub(
            "[.,?;*!%^&_+():-\[\]{}]",
            "",
            t.replace('"', "")
            .replace("/", "")
            .replace("\\", "")
            .replace("'", "")
            .strip()
            .lower(),
        )
        tokens = [
            sent_cleaner(sent).strip() + " ."
            for sent in report_cleaner(report)
            if len(sent_cleaner(sent).strip()) > 0
        ]
        report = " ".join(tokens)
        return report

    @staticmethod
    def load_tag2ids(
        tag_path,
        train_idxs=None,
        need_header=False,
    ):
        cached_path = tag_path + ".pkl"
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                tags = pickle.load(f)
        else:
            tags = pd.read_csv(tag_path)
            with open(cached_path, "wb") as f:
                pickle.dump(tags, file=f)
        tags = tags.fillna(2).replace(-1, 1)
        diseases = list(tags)[2:]
        id2tags = defaultdict(dict)
        count = 0
        given_count = 0
        conflict = 0
        for i in range(len(tags)):
            tag = tags.iloc[i]
            idx = tag[1]
            no_finding = True
            given_no_finding = False
            id2tags[idx] = {}
            for key, disease in zip(tag[2:], diseases):
                if key == 2:
                    continue
                id2tags[idx][disease] = True if key == 1 else False
                if disease == "No Finding":
                    given_no_finding = id2tags[idx][disease]
                if id2tags[idx][disease] and disease not in {
                    "Support Devices",
                    "No Finding",
                }:
                    no_finding = False
            if given_no_finding != no_finding:
                conflict += 1
                # print(idx)
            if given_no_finding:
                given_count += 1
            # id2tags[idx]["No Finding"] = no_finding #and given_no_finding
            if not id2tags[idx]["No Finding"]:
                count += 1
        print(
            "Abnormal Report %d, Total Report %d, Ratio %0.3f"
            % (count, len(tags), count / len(tags))
        )
        print(
            "Conflict No Finding Label %d, Total No Finding Label %d, Ratio %0.3f"
            % (conflict, len(tags), conflict / len(tags))
        )
        print("Given No Finding", given_count)

        if need_header:
            return id2tags, diseases
        return id2tags

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx["[UNK]"]
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [self.decoder_start_token_id] + ids + [self.eos_token_id]
        return ids

    def encode(
        self,
        report,
        add_special_tokens=True,
    ):
        ids = []
        tokens = self.clean_report(report).split()
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        if add_special_tokens:
            ids = [self.decoder_start_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True, separator=" "):
        txt = []
        for i, idx in enumerate(ids):
            if idx not in self.idx2token:
                idx = self.unk_token_id
            token = self.idx2token[idx]
            if skip_special_tokens and token in self.special_tokens:
                continue
            txt.append(token)
        return separator.join(txt)

    def batch_decode(self, ids_batch, skip_special_tokens=True, separator=" "):
        out = []
        for ids in ids_batch:
            out.append(
                self.decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    separator=separator,
                )
            )
        return out

    def save_pretrained(self, save_directory):
        return ""
