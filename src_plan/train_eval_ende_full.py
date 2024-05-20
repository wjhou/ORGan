import json
import os
from collections import defaultdict
from tokenizer import Tokenizer
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from metrics import compute_scores


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


def prf(ref, hyp):
    ref = set(ref)
    hyp = set(hyp)
    tp = len(ref & hyp)
    p = tp / max(len(hyp), 1)
    r = tp / max(len(ref), 1)
    if p == 0 or r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)
    return [p, r, f1]


def clean_outputs(outputs, special_tokens, offset):
    # remove duplicated observations
    def clean(seq):
        new_seq = []
        obs = set()
        for tok in seq[1:]:
            if tok == special_tokens[0]:
                break
            if tok % offset in obs or tok == special_tokens[1]:
                continue
            new_seq.append(tok)
            obs.add(tok % offset)
        return new_seq

    new_outputs = []
    for output in outputs:
        new_output = [clean(out) for out in output]
        new_outputs.append(new_output)
    return new_outputs


def eval_text(
    max_tgt_length: int,
    model,
    tokenizer: Tokenizer,
    test_dataset,
    output_path: str,
    prediction_file_name: str = "predictions.txt",
    num_beams=None,
    return_result=False,
):
    prediction_file_name = prediction_file_name.replace(".txt", ".json")
    model.eval()
    max_length = max_tgt_length
    print("******************")
    print("Text generation max length", max_length)
    print("******************")

    predictions = []
    references = []
    report_ids = []
    all_output_sequences = []
    test_progress = tqdm(
        test_dataset,
        desc="Evaluating Model (Observation Planning)",
    )
    if num_beams is None:
        num_beams = 1
    with torch.no_grad():
        for i, batch in enumerate(test_progress):
            max_length = max_tgt_length
            min_length = 2
            pixel_values = batch["pixel_values"].cuda()
            model_inputs = {
                "pixel_values": pixel_values,
                "num_beams": num_beams,
                "max_length": max_length,
                "min_length": min_length,
                "decoder_start_token_id": model.config.decoder_start_token_id,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,
                "no_repeat_ngram_size": 2,  # remove duplicated observations
                "return_dict_in_generate": True,
                "num_return_sequences": num_beams,
            }
            outputs = model.generate(**model_inputs)
            output_sequences = (
                outputs["sequences"]
                .view(-1, num_beams, outputs["sequences"].size(-1))
                .tolist()
            )
            reference = (
                batch["labels"]
                .masked_fill(
                    batch["labels"] == -100,
                    model.config.pad_token_id,
                )
                .tolist()
            )
            reference = [
                [
                    r
                    for r in ref
                    if r != model.config.pad_token_id and r != model.config.bos_token_id
                ]
                for ref in reference
            ]
            prediction = []

            output_sequences = clean_outputs(
                outputs=output_sequences,
                special_tokens=[model.config.eos_token_id, model.config.pad_token_id],
                offset=(len(tokenizer.idx2token) - 2) // 2,
            )
            prediction = [o_s[0] for o_s in output_sequences]
            predictions.extend(prediction)
            references.extend(reference)
            report_ids.extend(batch["report_ids"])
            all_output_sequences.extend(output_sequences)

    acc = 0
    gts = {}
    res = {}
    index = 0
    ref_len = 0
    hyp_len = 0
    nor_preds = []
    nor_trues = []
    abn_preds = []
    abn_trues = []
    dist = 0
    for a, b in zip(predictions, references):
        ref_len += len(b)
        hyp_len += len(a)
        astr = "".join([chr(tok + 97) for tok in a])
        bstr = "".join([chr(tok + 97) for tok in b])
        res[index] = [" ".join([chr(tok + 97) for tok in a])]
        gts[index] = [" ".join([chr(tok + 97) for tok in b])]

        index += 1
        if astr == bstr:
            acc += 1
        a = set(a)
        b = set(b)
        abn_pred = [0] * ((len(tokenizer.idx2token) - 2) // 2)
        abn_true = [0] * ((len(tokenizer.idx2token) - 2) // 2)
        nor_pred = [0] * ((len(tokenizer.idx2token) - 2))
        nor_true = [0] * ((len(tokenizer.idx2token) - 2))
        for ele in a:
            if ele < len(abn_pred):
                abn_pred[ele] = 1
            nor_pred[ele] = 1
        for ele in b:
            if ele < len(abn_true):
                abn_true[ele] = 1
            nor_true[ele] = 1
        abn_preds.append(abn_pred)
        abn_trues.append(abn_true)
        nor_trues.append(nor_true)
        nor_preds.append(nor_pred)
    acc /= len(predictions)
    ref_len /= len(references)
    hyp_len /= len(predictions)
    dist /= len(predictions)

    import warnings

    warnings.filterwarnings("ignore")
    micro_nor_prf = precision_recall_fscore_support(
        y_true=nor_trues,
        y_pred=nor_preds,
        average="micro",
    )[:-1]
    macro_nor_prf = precision_recall_fscore_support(
        y_true=nor_trues,
        y_pred=nor_preds,
        average="macro",
    )[:-1]
    micro_abn_prf = precision_recall_fscore_support(
        y_true=abn_trues,
        y_pred=abn_preds,
        average="micro",
    )[:-1]
    macro_abn_prf = precision_recall_fscore_support(
        y_true=abn_trues,
        y_pred=abn_preds,
        average="macro",
    )[:-1]
    nor_trues = np.array(nor_trues)
    nor_preds = np.array(nor_preds)
    for i in range(nor_trues.shape[1]):
        zzzz = precision_recall_fscore_support(
            y_true=nor_trues[:, i],
            y_pred=nor_preds[:, i],
            average="binary",
        )
        print(
            i,
            tokenizer.idx2token[i],
            zzzz[0],
            zzzz[1],
            zzzz[2],
            sum(nor_trues[:, i]),
            sum(nor_preds[:, i]),
        )

    bleu_scores = compute_scores(gts=gts, res=res)

    print("===========================")
    print("Micro Normal Precision %0.4f Recall %0.4f F1 %0.4f" % micro_nor_prf[:3])
    print("Macro Normal Precision %0.4f Recall %0.4f F1 %0.4f" % macro_nor_prf[:3])
    print("===========================")
    print("Micro ANormal Precision %0.4f Recall %0.4f F1 %0.4f" % micro_abn_prf[:3])
    print("Macro ANormal Precision %0.4f Recall %0.4f F1 %0.4f" % macro_abn_prf[:3])
    print("===========================")
    print("Accuracy %0.4f" % acc)
    print("Ref Len %0.4f\tHyp Len %0.4f" % (ref_len, hyp_len))
    for score in bleu_scores:
        print("%s\t%0.4f" % (score, bleu_scores[score]))
    print("===========================")
    print()

    if output_path:
        with open(
            os.path.join(output_path, prediction_file_name),
            "w",
            encoding="utf-8",
        ) as f:
            dist_ref = defaultdict(int)
            dist_pre = defaultdict(int)
            for idx, rid, pre, ref in zip(
                range(len(predictions)),
                report_ids,
                predictions,
                references,
            ):
                ref = "-".join([str(k) for k in ref])
                pre = "-".join([str(k) for k in pre])
                dist_ref[ref] += 1
                dist_pre[pre] += 1
            predictions = {
                report_id: [
                    {
                        tokenizer.idx2token[pred]: idx
                        for idx, pred in enumerate(prediction)
                    }
                    for prediction in output_sequences
                ]
                for report_id, output_sequences in zip(
                    report_ids,
                    all_output_sequences,
                )
            }
            json.dump(predictions, f, ensure_ascii=False, indent=4)
            if return_result:
                return predictions
    print("-------------------------------")
    print("DISTINCT Generated Plan %d" % len(dist_pre))
    print("DISTINCT Reference Plan %d" % len(dist_ref))
    print("-------------------------------")
    return {"micro_abn_prf": micro_abn_prf[2]}
