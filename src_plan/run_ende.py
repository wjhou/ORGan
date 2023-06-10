#!/usr/bin/env python
# coding=utf-8
import json
import logging
import os
import sys

import datasets
import torch
import transformers
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    BartConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer_utils import get_last_checkpoint

from data_arguments import DataTrainingArguments
from data_collator_ende import DataCollatorForEnDe as DataCollatorForSeq2Seq
from dataset_ende import DatasetCustom
from model_arguments import ModelArguments
from seq2seqtrainer_metrics_ende import Seq2SeqTrainerGenMetrics
from train_eval_ende_full import train

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.group_by_length = True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    from tokenizer import Tokenizer

    data_args.dataset = (
        "iu_xray" if "iu_xray" in data_args.annotation_file else "mimic_cxr"
    )
    data_args.threshold = 3 if "iu_xray" in data_args.annotation_file else 10

    train_image_tokenizer = image_tokenizer = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if data_args.dataset == "iu_xray":
        train_image_tokenizer = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    text_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        **tokenizer_kwargs,
    )

    id2tags, headers = Tokenizer.load_tag2ids(data_args.tag_path, None, True)
    tokenizer = Tokenizer(data_args, headers)
    logger.info("***************************")
    logger.info("***************************")
    print(data_args)
    logger.info("***************************")
    logger.info("***************************")

    num_layers = 3
    config = BartConfig(
        vocab_size=len(tokenizer.idx2token),
        max_position_embeddings=data_args.max_tgt_length,
        encoder_layers=num_layers,
        encoder_ffn_dim=512,
        encoder_attention_heads=8,
        decoder_layers=num_layers,
        decoder_ffn_dim=512,
        decoder_attention_heads=8,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="relu",
        d_model=512,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        is_encoder_decoder=True,
        forced_eos_token_id=0,
        alpha=model_args.alpha,
    )
    from src_plan.models.modeling_bart_custom import BartForConditionalGeneration

    visual_backbone = {
        "resnet101": ("resnet101", 2048),
    }
    config.visual_extractor, config.d_visual = visual_backbone["resnet101"]
    config.visual_extractor_pretrained = True
    config.chexpert_model_name_or_path = model_args.chexpert_model_name_or_path
    config.dataset = data_args.dataset
    config.max_tgt_length = data_args.max_tgt_length
    # 28 observations + [CLS] + [PAD]
    config.obs_num = len(tokenizer.idx2token) - 2
    config.region_num = 98 if "iu_xray" in data_args.annotation_file else 49

    with open(data_args.annotation_file, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    config.tag_size = config.obs_num

    model = BartForConditionalGeneration(
        config=config,
        tokenizer=tokenizer,
    )

    logger.info("***************************")
    logger.info("***** Model Structure *****")
    logger.info("***************************")

    logger.info(model)
    logger.info("***************************")
    logger.info("***************************")
    logger.info("***************************")
    train_dataset = eval_dataset = test_dataset = None

    id2tagpos = json.load(open(data_args.id2tagpos_path, "r", encoding="utf-8"))

    if data_args.debug_model:
        for key in annotation:
            annotation[key] = annotation[key][:16]

    if training_args.do_train:
        train_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            split="train",
            status="train",
            image_tokenizer=train_image_tokenizer,
            text_tokenizer=tokenizer,
            id2tags=(id2tagpos, id2tags, headers),
        )
        eval_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            split="valid",
            image_tokenizer=image_tokenizer,
            text_tokenizer=tokenizer,
            id2tags=(id2tagpos, id2tags, headers),
        )
    if training_args.do_predict:
        test_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            split="test",
            image_tokenizer=image_tokenizer,
            text_tokenizer=tokenizer,
            id2tags=(id2tagpos, id2tags, headers),
        )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        text_tokenizer=text_tokenizer,
        train_image_tokenizer=train_image_tokenizer,
        eval_image_tokenizer=image_tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args.max_tgt_length = data_args.max_tgt_length
    training_args.num_beams = model_args.num_beams
    training_args.fast_lr = model_args.fast_lr
    data_args.max_steps = training_args.max_steps
    from transformers import EarlyStoppingCallback

    trainer = Seq2SeqTrainerGenMetrics(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.data_args = data_args
    if training_args.do_train:
        logger.info("*** Train ***")
        train(
            training_args,
            data_args,
            last_checkpoint,
            trainer,
            train_dataset,
        )

    # Prediction
    if training_args.do_predict:
        logger.info("*** Test ***")
        if model_args.test_model_name_or_path is not None:
            logger.info(
                "*** Test: Loading %s ***" % (model_args.test_model_name_or_path)
            )
            state_dict = torch.load(
                os.path.join(
                    model_args.test_model_name_or_path,
                    WEIGHTS_NAME,  # pytorch_model.bin
                ),
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=False)
            model = model.cuda()
        from train_eval_ende_full import eval_text

        print(model_args.num_beams)
        eval_text(
            max_tgt_length=data_args.max_tgt_length,
            model=model,
            tokenizer=tokenizer,
            test_dataset=trainer.get_test_dataloader(test_dataset),
            output_path=training_args.output_dir,
            num_beams=model_args.num_beams,
        )


if __name__ == "__main__":
    main()
