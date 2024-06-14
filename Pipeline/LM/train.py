#!/usr/bin/env python3
"""Recipe for training a Language Model with kdialectspeech 
transcript and lm_corpus.

To run this recipe, do the following:
> pip install datasets
> python train.py hparams/<hparam_file>.yaml \
    --data_folder <local_path_to_kdialectspeech_dataset>

Authors
 * Jianyuan Zhong 2021
 * Ju-Chieh Chou 2020
 * Dongwon Kim, Dongwoo Kim 2021
 * N Park, 2022
"""
### speechbrain.utils.train_logger.TensorboardLogger를 사용하기 위해 필요
# from torch.utils.tensorboard import SummaryWriter # 중요,
# 처음 임포트해야 함, speechbrain을 먼저 임포트하도 나중에 임포드 하면 segmentation fault 오류남.

### 아래 모듈을 설치하라고 권고함.
# !pip install -U torch-tb-profiler

import os
import sys
import glob
from pathlib import Path
import logging
import torch
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main



logger = logging.getLogger(__name__)


# Define training procedure
class LM(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the sentence batches
        to the output probabilities."""
        batch = batch.to(self.device)
        tokens_bos, _ = batch.tokens_bos
        logits = self.hparams.model(tokens_bos)
        pred = self.hparams.log_softmax(logits)
        return pred

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        batch = batch.to(self.device)
        tokens_eos, tokens_len = batch.tokens_eos
        loss = self.hparams.compute_cost(
            predictions, tokens_eos, length=tokens_len
        )
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        (loss / self.hparams.accu_steps).backward()

        if self.step % self.hparams.accu_steps == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if isinstance(
                self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
            ) or isinstance(
                self.hparams.lr_annealing,
                sb.nnet.schedulers.CyclicCosineScheduler,
            ):
                self.hparams.lr_annealing(self.optimizer)

        if isinstance(
            self.hparams.train_logger, sb.utils.train_logger.TensorboardLogger
        ):
            self.hparams.train_logger.log_stats(
                stats_meta={"step": self.step}, train_stats={"loss": loss},
            )

        return loss

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            if not (
                isinstance(
                    self.hparams.lr_annealing, sb.nnet.schedulers.NoamScheduler
                )
                or isinstance(
                    self.hparams.lr_annealing,
                    sb.nnet.schedulers.CyclicCosineScheduler,
                )
            ):
                old_lr, new_lr = self.hparams.lr_annealing(stage_loss)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            else:
                old_lr = self.hparams.lr_annealing.current_lr

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["loss"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )

    # test is separate
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    """Define text pipeline"""
    # TODO: implement text augmentations pipelines
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "tokens_bos", "tokens_eos")
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "wrd", "tokens_bos", "tokens_eos"],
    )
    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the dataloader objects as well as
    # tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # We download the tokenizer from HuggingFace (or elsewhere depending on
    # the path given in the YAML file).
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    lm_brain = LM(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    lm_brain.fit(
        lm_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # evaluation
    lm_brain.evaluate(
        test_data,
        min_key="loss",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # best model link
    def best_modle_link():
        model_dir = hparams["save_folder"]
        best_model_dir = sorted(glob.glob(model_dir + '/CKPT*'), key=os.path.getmtime)[0]
        best_model_dir = Path(best_model_dir).stem

        best_model = os.path.join(best_model_dir + '/model.ckpt')
        lm_model =  os.path.join(model_dir, 'lm.ckpt')

        os.symlink(best_model, lm_model)
        logger.info(f'symbolic link from {best_model} to {lm_model} is created')
        
    run_on_main(best_modle_link)