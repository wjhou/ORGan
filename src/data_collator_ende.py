from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from PIL import Image
from transformers import DataCollatorForSeq2Seq
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
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        input_pixels = (
            [feature["input_pixels"] for feature in features]
            if "input_pixels" in features[0].keys()
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
        node_ids = (
            [feature["node_ids"] for feature in features]
            if "node_ids" in features[0].keys()
            else None
        )
        matrix = (
            [feature["matrix"] for feature in features]
            if "matrix" in features[0].keys()
            else None
        )
        if input_pixels is None:
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

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)
                batch_outputs["labels"].append(feature["labels"])

        if node_ids is not None:
            batch_outputs["node_ids"] = []
            batch_outputs["node_mask"] = []
            batch_outputs["ngram_labels"] = []
            max_len = max(len(l) for l in node_ids)
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_len - len(feature["node_ids"])
                )
                feature["node_ids"] = feature["node_ids"] + remainder
                remainder = [0] * (max_len - len(feature["node_mask"]))
                feature["node_mask"] = feature["node_mask"] + remainder
                remainder = [-1] * (max_len - len(feature["ngram_labels"]))
                feature["ngram_labels"] = feature["ngram_labels"] + remainder
                batch_outputs["node_ids"].append(feature["node_ids"])
                batch_outputs["node_mask"].append(feature["node_mask"])
                batch_outputs["ngram_labels"].append(feature["ngram_labels"])

        if matrix is not None:
            batch_outputs["matrix"] = []
            max_len = max(len(l) for l in matrix)
            for feature in features:
                diff = max_len - len(feature["matrix"])
                feature["matrix"] = np.pad(
                    feature["matrix"],
                    ((0, diff), (0, diff)),
                    "constant",
                    constant_values=(0, 0),
                )
                batch_outputs["matrix"].append(feature["matrix"].tolist())

        image_tokenizer = (
            self.train_image_tokenizer
            if split == "train"
            else self.eval_image_tokenizer
        )
        features = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        # image lazy loading
        if input_pixels is None:
            input_pixels = []
            for image_path in image_paths:
                pixel_value = []
                for img_path in image_path:
                    image = Image.open(img_path).convert("RGB")
                    pixel_val = image_tokenizer(image)
                    pixel_value.append(pixel_val)
                pixel_value = torch.stack(pixel_value, dim=0)
                input_pixels.append(pixel_value)
        features["input_pixels"] = torch.stack(input_pixels, dim=0)

        if report_ids is not None:
            features["report_ids"] = report_ids

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids
        return features

    def pad_sequence(self, seqs, padding_idx, max_len):
        new_seqs = []
        for seq in seqs:
            seq_len = len(seq)
            diff = max_len - seq_len
            new_seqs.append(seq + [padding_idx] * diff)
        return new_seqs
