import random
import warnings
from dataclasses import dataclass
from optparse import Option
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from PIL import Image
from regex import E
from transformers import DataCollatorForSeq2Seq

# from transformers.utils import PaddingStrategy
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


@dataclass
class DataCollatorForEnDe(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    text_tokenizer: Optional[Any] = None
    train_image_tokenizer: Optional[Any] = None
    eval_image_tokenizer: Optional[Any] = None
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        input_ids = (
            [feature["input_ids"] for feature in features]
            if "input_ids" in features[0].keys()
            else None
        )
        pixel_values = (
            [feature["pixel_values"] for feature in features]
            if "pixel_values" in features[0].keys()
            else None
        )
        split = (
            [feature["split"] for feature in features][0]
            if "split" in features[0].keys()
            else "eval"
        )
        report_ids = (
            [feature["report_ids"] for feature in features]
            if "report_ids" in features[0].keys()
            else None
        )

        if pixel_values is None:
            image_paths = [feature["image_path"] for feature in features]

        batch_outputs = {}

        if labels is not None:
            batch_outputs["labels"] = []
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                feature["labels"] = feature["labels"] + remainder
                batch_outputs["labels"].append(feature["labels"])

        if input_ids is not None:
            batch_outputs["input_ids"] = []
            max_label_length = max(max(len(z) for z in l) for l in input_ids)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                feature["input_ids"] = [
                    input_ids_
                    + [self.text_tokenizer.pad_token_id]
                    * (max_label_length - len(input_ids_))
                    for input_ids_ in feature["input_ids"]
                ]
                batch_outputs["input_ids"].append(feature["input_ids"])

        image_tokenizer = (
            self.train_image_tokenizer
            if split == "train"
            else self.eval_image_tokenizer
        )
        features = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        if pixel_values is None:
            pixel_values = load_images(
                image_paths=image_paths,
                image_tokenizer=image_tokenizer,
            )

        features["pixel_values"] = torch.stack(pixel_values, dim=0)
        features["input_ids"] = (
            features["labels"]
            .clone()
            .masked_fill(
                (features["labels"] == -100)
                | (features["labels"] == self.text_tokenizer.eos_token_id),
                self.text_tokenizer.pad_token_id,
            )
        )
        features["attention_mask"] = torch.ones_like(features["input_ids"]).masked_fill(
            features["input_ids"] == self.text_tokenizer.pad_token_id, 0
        )
        if report_ids is not None:
            features["report_ids"] = report_ids

        # if (labels is not None and self.model is not None and hasattr(
        #         self.model, "prepare_decoder_input_ids_from_labels")):
        # decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
        #     labels=features["labels"])
        # features["decoder_input_ids"] = decoder_input_ids
        return features

    def pad_sequence(self, seqs, padding_idx, max_len):
        new_seqs = []
        for seq in seqs:
            seq_len = len(seq)
            diff = max_len - seq_len
            new_seqs.append(seq + [padding_idx] * diff)
        return new_seqs


def load_images(image_paths, image_tokenizer):
    pixel_values = []
    for image_path in image_paths:
        pixel_value = []
        for img_path in image_path:
            image = Image.open(img_path).convert("RGB")
            pixel_val = image_tokenizer(image)
            pixel_value.append(pixel_val)
        pixel_value = torch.stack(pixel_value, dim=0)
        pixel_values.append(pixel_value)
    return pixel_values
