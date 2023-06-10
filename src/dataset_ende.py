import imp
import json
import os
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from data_arguments import DataTrainingArguments
from tokenizer import Tokenizer
from tqdm import tqdm


def process_examples(examples, text_tokenizer: Tokenizer, max_tgt_length):
    progress = tqdm(
        range(len(examples["id"])),
        desc="Processing Samples",
    )
    pixel_values = []
    labels = []
    idxs = []
    image_paths = []
    for index in progress:
        image_path = examples["image_path"][index]
        report = examples["report"][index]
        pixel_value = None
        label = text_tokenizer.encode(
            report,
            add_special_tokens=False,
        )
        if len(label) > max_tgt_length - 1:
            label = label[: max_tgt_length - 1]
        label = label + [text_tokenizer.eos_token_id]

        pixel_values.append(pixel_value)
        labels.append(label)
        idxs.append(examples["id"][index])
        image_paths.append(image_path)
    return idxs, image_paths, pixel_values, labels


class DatasetCustom(Dataset):
    def __init__(
        self,
        data_args: DataTrainingArguments,
        config,
        annotation,
        split: str,
        image_tokenizer: transforms.Compose,
        text_tokenizer: Tokenizer,
        id2tags: Tuple[dict, list] = None,
        nodes: Tuple[dict, list] = None,
        keep_columns={
            "id",
            "report",
            "image_path",
        },
    ) -> None:
        super().__init__()
        self.config = config
        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        self.data_args = data_args
        self.split = split
        self.id2tags = id2tags[1]
        self.id2tagpos = id2tags[0]
        self.obs2key = id2tags[-1]
        self.offset = 0
        if self.split != "train":
            root_path = data_args.plan_model_name_or_path
            file_name = data_args.plan_eval_file
            if self.split == "valid":
                path = os.path.join(
                    root_path,
                    file_name,
                )
            else:
                path = os.path.join(root_path, "predictions.json")
            self.id2tagpos = json.load(
                open(
                    path,
                    "r",
                    encoding="utf-8",
                )
            )
        self.dataset = (
            "iu_xray" if "iu_xray" in data_args.annotation_file else "mimic_cxr"
        )
        self.header = {h: i for i, h in enumerate(id2tags[2])}
        (
            key_vocab,
            tok_vocab,
            self.node2id,
            self.id2node,
            self.key2tok,
            self.node2tok,
        ) = nodes
        print("*********************")
        print("*********************")
        print("Number of N grams", len(self.node2id) + 2 * len(self.header))
        print("*********************")
        print("*********************")
        # if self.split == "train":
        self.id2ngram = json.load(
            open(
                "./data/%s_id2ngram.json" % self.dataset,
                "r",
                encoding="utf-8",
            )
        )
        # convert ngram to node_id
        for idx in self.id2ngram:
            l = set()
            for ngram in self.id2ngram[idx]:
                tokens = ngram.split()
                if len(tokens) == 1 and (ngram + "_NODE") in self.node2id:
                    l.add(self.node2id[ngram + "_NODE"])
                elif ngram in self.node2id:
                    l.add(self.node2id[ngram])
                for token in tokens:
                    if token in self.node2id:
                        l.add(self.node2id[token])
            self.id2ngram[idx] = l

        self.instance_topk = data_args.topk
        examples = {kc: [] for kc in keep_columns}
        samples = annotation[split.replace("valid", "val")]
        for sample in samples:
            for key in sample:
                if key not in examples:
                    continue
                examples[key].append(sample[key])
        for key in examples:
            print(key, examples[key][:3])

        idxs, image_paths, pixel_values, labels = process_examples(
            examples=examples,
            text_tokenizer=text_tokenizer,
            max_tgt_length=data_args.max_tgt_length,
        )
        self.data = [
            {
                "id": c,
                "pixel_values": a,
                "labels": b,
                "image_path": d,
            }
            for a, b, c, d in zip(
                pixel_values,
                labels,
                idxs,
                image_paths,
            )
        ]

        self.all_index = list(range(len(self.data)))

    def __getitem__(self, index):
        pixel_values = self.data[index]["pixel_values"]
        labels = self.data[index]["labels"]
        idx = self.data[index]["id"]
        tags = self.id2tags[idx]
        if self.split == "train":
            plan_index = 0
            tag_pos = self.id2tagpos[idx]
        else:
            plan_index = 0
            tag_pos = self.id2tagpos[idx]
            tag_pos = tag_pos[plan_index]

        if idx in self.id2ngram:
            gt_ngrams = self.id2ngram[idx]
        else:
            gt_ngrams = []

        node_ids = []
        outline_ids = []
        pos = []
        mention2ngram = defaultdict(set)
        ngram2token = defaultdict(set)
        if len(tag_pos) == 0:
            if self.split != "train":
                tag_pos["No Finding_True"] = 0
            else:
                tag_pos["No Finding"] = 0
                tags["No Finding"] = True

        for tag in tag_pos:
            if self.split != "train":
                p = tag_pos[tag]
                mention = tag
                tag, obs = mention.split("_")
            else:
                p = tag_pos[tag]
                mention = tag
                if not tags[tag]:
                    mention = mention + "_False"
                else:
                    mention = mention + "_True"
            if mention not in self.node2id:
                continue
            cur_tag_id = self.node2id[mention]
            if cur_tag_id in node_ids:
                continue
            pos.append(p)
            outline_ids.append(cur_tag_id)
            node_ids.append(cur_tag_id)
            if mention not in self.obs2key:
                continue
            for key in self.obs2key[mention][: self.config.topk_ngram]:
                # for mention level (2nd-level)
                # cur_tag_id: mention id
                # key: ngram id
                mention2ngram[cur_tag_id].add(self.node2id[key])
                for tok in self.key2tok[key]:
                    # for token level (3rd-level)
                    # tok: token id
                    ngram2token[self.node2id[key]].add(self.node2id[tok])
        gt_tokens = set()
        for ngram in gt_ngrams:
            gt_tokens.update(ngram2token[ngram])

        outline_ids = list(
            map(
                lambda x: x[0],
                sorted(
                    zip(outline_ids, pos),
                    key=lambda x: x[1],
                ),
            )
        )
        node_ids = list(
            map(
                lambda x: x[0],
                sorted(
                    zip(node_ids, pos),
                    key=lambda x: x[1],
                ),
            )
        )

        node_mask = [1] * len(node_ids)
        # gather all ngram ids
        ngram_labels = [-1] * len(node_ids)
        if self.config.outline_level > 1:
            ngram_ids = set()
            for mention in mention2ngram:
                if mention not in node_ids:
                    continue
                ngram_ids.update(mention2ngram[mention])
            ngram_ids = sorted(ngram_ids, key=lambda x: x)
            node_ids.extend(ngram_ids)
            node_mask = node_mask + [2] * len(ngram_ids)
            ngram_labels_ = [-1] * len(ngram_ids)
            ngram_labels = ngram_labels + ngram_labels_
        # gather all token ids
        if self.config.outline_level > 2:
            token_ids = set()
            for ngram in ngram2token:
                token_ids.update(ngram2token[ngram])
            token_ids = sorted(token_ids, key=lambda x: x)
            node_ids.extend(token_ids)
            node_mask = node_mask + [3] * len(token_ids)
            token_labels = [0] * len(token_ids)
            for ti, token_id in enumerate(token_ids):
                if token_id in gt_ngrams:
                    token_labels[ti] = 1
            ngram_labels = ngram_labels + token_labels
        node2index = {node: index for index, node in enumerate(node_ids)}
        # mention-ngram
        l2_pairs = []
        if self.config.outline_level > 1:
            for mention in mention2ngram:
                for ngram in mention2ngram[mention]:
                    l2_pairs.append((node2index[mention], node2index[ngram]))
        # ngram-token
        l3_pairs = []
        if self.config.outline_level > 2:
            for ngram in ngram2token:
                for token in ngram2token[ngram]:
                    l3_pairs.append((node2index[ngram], node2index[token]))

        matrix = np.zeros(
            (len(node_ids), len(node_ids)),
            dtype=int,
        )
        for z in range(len(outline_ids)):
            matrix[z, z] = 1
            if z + 1 < len(outline_ids):
                matrix[z, z + 1] = 1
                matrix[z + 1, z] = 1
        for l2_pair in l2_pairs:
            x, y = l2_pair
            matrix[y, y] = -2
            matrix[y, x] = 2
        for l3_pair in l3_pairs:
            x, y = l3_pair
            matrix[y, y] = -3
            matrix[y, x] = 3

        # TODO
        item = {
            "image_path": [
                os.path.join(self.data_args.image_path, a)
                for a in self.data[index]["image_path"]
            ],
            "labels": labels,
            "split": self.split,
            "node_ids": node_ids,
            "node_mask": node_mask,
            "matrix": matrix,
            "ngram_labels": ngram_labels,
        }
        if self.split != "train":
            item["report_ids"] = idx
        if pixel_values is not None:
            item["input_pixels"] = pixel_values
        return item

    def __len__(self):
        return len(self.data)
