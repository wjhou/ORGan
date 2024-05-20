import os
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from data_arguments import DataTrainingArguments
from tokenizer import Tokenizer
from tqdm import tqdm
from PIL import Image


def process_examples(examples):
    progress = tqdm(
        range(len(examples["id"])),
        desc="Processing Samples",
    )
    idxs = []
    image_paths = []
    for index in progress:
        image_path = examples["image_path"][index]
        idxs.append(examples["id"][index])
        image_paths.append(image_path)
    return idxs, image_paths


class DatasetCustom(Dataset):
    def __init__(
        self,
        data_args: DataTrainingArguments,
        annotation,
        split: str,
        image_tokenizer: transforms.Compose,
        text_tokenizer: Tokenizer,
        status: str = None,
        id2tags: Tuple[dict, list] = None,
        keep_columns={
            "id",
            "report",
            "image_path",
        },
    ) -> None:
        super().__init__()
        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        self.data_args = data_args
        self.split = split
        self.status = status if split == "train" else split
        self.id2tags = id2tags[1]
        self.id2tagpos = id2tags[0]
        self.headers = {h: i for i, h in enumerate(id2tags[-1])}
        examples = {kc: [] for kc in keep_columns}
        samples = annotation[split.replace("valid", "val")]
        for sample in samples:
            for key in sample:
                if key not in examples:
                    continue
                examples[key].append(sample[key])
        for key in examples:
            print(key, examples[key][:3])
        self.dataset = (
            "iu_xray" if "iu_xray" in data_args.annotation_file else "mimic_cxr"
        )
        idxs, image_paths = process_examples(examples=examples)

        self.data = [
            {
                "id": a,
                "image_path": b,
            }
            for a, b in zip(
                idxs,
                image_paths,
            )
        ]
        self.all_index = list(range(len(self.data)))

    def __getitem__(self, index):
        idx = self.data[index]["id"]
        tag_pos = self.id2tagpos[idx]
        tag_pre = self.id2tags[idx]
        labels = []
        pos = []
        for tag in tag_pos:
            p = tag_pos[tag]
            token = tag + "_" + str(tag_pre[tag])
            tag_idx = self.text_tokenizer.token2idx[token]
            if tag_idx in labels:
                continue
            pos.append(p)
            labels.append(tag_idx)
        # for observation planning
        labels = list(
            map(
                lambda x: x[0],
                sorted(
                    zip(labels, pos),
                    key=lambda x: x[1],
                ),
            )
        )
        labels = labels + [self.text_tokenizer.eos_token_id]

        image_paths = [
            os.path.join(self.data_args.image_path, a)
            for a in self.data[index]["image_path"]
        ]
        pixel_values = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        pixel_values = torch.stack(
            [self.image_tokenizer(img) for img in pixel_values], dim=0
        )

        item = {
            "image_path": image_paths,
            "pixel_values": pixel_values,
            "labels": labels,
            "split": self.split,
        }
        if self.status != "train":
            item["report_ids"] = idx
        return item

    def __len__(self):
        return len(self.data)
