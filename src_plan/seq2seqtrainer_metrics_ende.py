import collections
from typing import List, Optional

from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import logging

from optimizer import create_optimizer
from train_eval_ende_full import eval_text

logger = logging.get_logger(__name__)


class Seq2SeqTrainerGenMetrics(Seq2SeqTrainer):
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if self.args.past_index >= 0:
            self._past = None

        metrics = eval_text(
            max_tgt_length=self.args.max_tgt_length,
            model=self.model,
            tokenizer=self.tokenizer,
            test_dataset=dataloader,
            output_path=self.args.output_dir,
            prediction_file_name="prediction_eval_step_%d.txt"
            % (self.state.global_step),
            num_beams=self.args.num_beams,
        )

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(eval_dataset),
        )

    def create_optimizer(self):
        print("Create Optimizer with Different Learning Rate")
        print("Slow Learning Rate\t%0.5f" % self.args.learning_rate)
        print("Fast Learning Rate\t%0.5f" % self.args.fast_lr)
        print("Weight Decay\t%0.5f" % self.args.weight_decay)
        self.optimizer = create_optimizer(
            model=self.model,
            args=self.args,
            fast_lr=self.args.fast_lr,
        )
        return self.optimizer
