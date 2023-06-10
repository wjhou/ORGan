import os

import torch
from tqdm import tqdm

from metrics import compute_scores
import json


def train(training_args, data_args, last_checkpoint, trainer, train_dataset):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def eval_text(
    max_tgt_length: int,
    model,
    tokenizer,
    test_dataset,
    num_beams=None,
):
    model.eval()

    max_length = max_tgt_length
    print("******************")
    print("Text generation max length", max_length)
    print("******************")

    predictions = []
    references = []
    test_progress = tqdm(
        test_dataset,
        desc="Evaluating Model (Report Generation)",
    )
    if num_beams is None:
        num_beams = 1

    print("******************")
    print("Beam Size", num_beams)
    print("******************")

    with torch.no_grad():
        for i, batch in enumerate(test_progress):
            max_length = max_tgt_length
            min_length = 2
            model_inputs = {
                "input_pixels": batch["input_pixels"].cuda(),
                "node_ids": batch["node_ids"].cuda(),
                "node_mask": batch["node_mask"].cuda(),
                "matrix": batch["matrix"].cuda(),
                "num_beams": num_beams,
                "max_length": max_length,
                "min_length": min_length,
                "decoder_start_token_id": model.config.decoder_start_token_id,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,
                "early_stopping": True,
            }
            outputs = model.generate(**model_inputs)
            output_sequences = outputs
            prediction = tokenizer.batch_decode(
                output_sequences.tolist(),
                skip_special_tokens=True,
            )
            labels = batch["labels"].masked_fill(
                batch["labels"] == -100,
                tokenizer.pad_token_id,
            )
            reference = tokenizer.batch_decode(
                labels.tolist(),
                skip_special_tokens=True,
            )
            predictions.extend(prediction)
            references.extend(reference)

    assert len(references) == len(predictions), "Prediction Num != Reference Num"
    bleu_scores = compute_scores(
        gts={index: [gt] for index, gt in enumerate(references)},
        res={index: [re] for index, re in enumerate(predictions)},
    )
    for score in bleu_scores:
        print("%s\t%0.4f" % (score, bleu_scores[score]))
    return bleu_scores
