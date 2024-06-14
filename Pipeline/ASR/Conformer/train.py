#!/usr/bin/env python3
"""Recipe for training a Conformer ASR system with KdialectSpeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch
coupled with a neural language model.

To run this recipe, do the following:
> python train.py hparams/conformer_medium.yaml

With the default hyperparameters, the system employs
a convolutional frontend and a transformer.
The decoder is based on a Transformer decoder.
Beamsearch coupled with a Transformer language model is used
on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
KdialectSpeech dataset.

The best model is the average of the checkpoints from last 5 epochs.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.


Authors
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Titouan Parcollet 2021
 * Dongwon Kim, Dongwoo Kim 2021
 * N Park 2022
"""
### speechbrain.utils.train_logger.TensorboardLogger를 사용하기 위해 필요, yaml에서 설정됨
from torch.utils.tensorboard import SummaryWriter # 중요, 맨 
# 처음 임포트해야 함, speechbrain을 먼저 임포트하도 나중에 임포드 하면 segmentation fault 오류남 

### 아래 모듈을 설치하라고 권고함.
# !pip install -U torch-tb-profiler

import os
import sys
import numpy as np
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches
        to the output probabilities."""
        # print(f'compute_forward ----- 1')
        # print(f'type of batch : {batch}')
        batch = batch.to(self.device)
        # print(f'compute_forward ----- 2')
        wavs, wav_lens = batch.sig
        # print(f'wavs, wav_lens : {wavs}, {wav_lens}')
        # print(f'compute_forward ----- 3')
        # print(f'wavs : {wavs}')
        # print(f'wav_lens : {wav_lens}')
        tokens_bos, _ = batch.tokens_bos

        # Add augmentation if specified
        ### kdialectspeech, ksponspeech, librispeech 에서는 사용안함, template 예제에서 사용
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # forward modules
        src = self.modules.CNN(feats)
        # print(f'tokens_bos : {tokens_bos}')
        # print(f'pad_idx : {self.hparams.pad_index}')
        enc_out, pred = self.modules.Transformer( # pred : decoder out
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # print(f'enc_out size : {enc_out.size()}')
        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with
                # limited capacity and no LM to give user some idea of
                # how the AM is doing
                ####
                #### 시간이 많이 걸리는 부분 : 아래 valid_search
                ####
                # print(f' valid enc_out size : {enc_out.size()}')
                # print(f' valid wav_lens : {wav_lens}')
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
                # print(f' valid hyps : {hyps}')
        elif stage == sb.Stage.TEST:
            # print(f'compute_forward ----- 4')
            # print(f' test enc_out size : {enc_out.size()}')
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens) # test_search와 valid_search의 차이는 LM 사용 여부
            # print(f' test hyps : {hyps}')
            # print(f'compute_forward ----- 5')
        # print(f'compute_forward ------------------------------------')
        # print(f'compute_forward p_ctc ----- : {p_ctc}')
        # print(f'compute_forward p_seq ----- : {p_seq}')
        # print(f'compute_forward wav_lens ----- : {wav_lens}')
        # print(f'compute_forward hyps ----- : {hyps}')
        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        
        # print(f'compute_objectives ----- 1')
        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        # print(f'compute_objectives ids : {ids}')
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens
        
        # logger.info(f'compute_objectives tokens.size ----- : {tokens.size()}') # npark

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        # print(f' compute_objectives tokens_eos : {tokens_eos}')
        # print(f' compute_objectives p_seq : {p_seq}')
        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )
        if stage != sb.Stage.TRAIN:
            # print(f'compute_objectives stage is not train -------------')
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                predicted_swords = get_swords(target_words, hyps) # space normalized hyp words
                predicted_chars = [
                    list("".join(utt_seq)) for utt_seq in predicted_words
                ]
                target_chars = [list("".join(wrd.split())) for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)
                self.swer_metric.append(ids, predicted_swords, target_words)
                self.cer_metric.append(ids, predicted_chars, target_chars)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
            
        # logger.info(f'compute_objectives loss ----- : {loss}') # npark
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        
        # print(f'train length of batch : {len(batch)}')
        self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

            if isinstance(
                self.hparams.train_logger,
                sb.utils.train_logger.TensorboardLogger,
            ):
                self.hparams.train_logger.log_stats(
                    stats_meta={"step": self.step}, train_stats={"loss": loss},
                )

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        # print(f'stage : {stage}')
        # print(f'length of batch : {len(batch)}')
        # print(f'batch.id type -------- : {type(batch.id)}')
        # print(f'batch.id -------- : {batch.id}')
        # print(f'batch sig type : {type(batch.sig)}')
        # print(f'batch sig : {batch.sig[0]}')
        # print(f'batch sig size : {batch.sig[0].size()}')
        
        # for k, v in batch.sig:
        #     print(k)
        #     print(v)
        
        with torch.no_grad():
            # print('########## compute_forward #########')
            predictions = self.compute_forward(batch, stage=stage)
            # print(f'########## compute_objectives ######### stage : {stage}')
            loss = self.compute_objectives(predictions, batch, stage=stage)
            # print('########## eval end #########')
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.swer_metric = self.hparams.error_rate_computer()
            self.cer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["sWER"] = self.swer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
            else:
                lr = self.hparams.lr_sgd
                steps = -1

            epoch_stats = {"epoch": epoch, "lr": lr, "steps": steps}
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.swer_metric.write_stats(w)
                self.wer_metric.write_stats(w)
                self.cer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer
            # only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initialize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:

                # do not reload the weights if training is interrupted
                # right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible(
                    device=torch.device(self.device)
                )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
        
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration", reverse=True)

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration", reverse=True)

    datasets = [train_data, valid_data, test_data]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer

# 공백 정규화 비교
def char_tokenizer(s):
    result = []
    flag = False
    for c in s:
        if c == ' ':
            flag = True
            continue
        if flag == True:
            c = '_' + c
            flag = False
        result.append(c)
    return result

def get_swords(ref , hyp):
    refs = char_tokenizer(ref)
    hyps = char_tokenizer(hyp)
    ref_nospace = ref.replace(' ', '')
    hyp_nospace = hyp.replace(' ', '')
    rlen = len(refs)
    hlen = len(hyps)
    scores =  np.zeros((hlen+1, rlen+1), dtype=np.int32)

    # initialize, 공란을 무시하고 음절의 거리 매트릭스 만들기
    for r in range(rlen+1):
        scores[0, r] = r
    for h in range(1, hlen+1):
        scores[h, 0] = scores[h-1, 0] + 1
        for r in range(1, rlen+1):
            sub_or_cor = scores[h-1, r-1] + (0 if ref_nospace[r-1] == hyp_nospace[h-1] else 1)
            insert = scores[h-1, r] + 1
            delete = scores[h, r-1] + 1
            scores[h, r] = min(sub_or_cor, insert, delete)

    # traceback and compute alignment
    h, r = hlen, rlen
    ref_norm, hyp_norm = [], []

    while r > 0 or h > 0:
        if h == 0:
            last_r = r - 1
        elif r == 0:
            last_h = h - 1
            last_r = r
        else:
            sub_or_cor = scores[h-1, r-1] + (0 if ref_nospace[r-1] == hyp_nospace[h-1] else 1)
            insert = scores[h-1, r] + 1
            delete = scores[h, r-1] + 1

            if sub_or_cor < min(insert, delete):
                last_h, last_r = h - 1, r - 1
            else:
                last_h, last_r = (h-1, r) if insert < delete else (h, r-1)

            c_hyp = hyps[last_h] if last_h == h-1 else ''
            c_ref = refs[last_r] if last_r == r-1 else ''
            h, r = last_h, last_r

            # do word-spacing normalization
            if c_hyp.replace('_', '') == c_ref.replace('_', ''):
                c_hyp = c_ref

        ref_norm.append(c_ref)
        hyp_norm.append(c_hyp)

    # ref_norm[::-1], hyp_norm[::-1]
    shyp = ''.join(map(str, hyp_norm[::-1])).replace('_', ' ')
    return shyp



if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(f'hparams_file : {hparams_file}')
    print(f'run_opts : {run_opts}')
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing KsponSpeech)
    from kdialectspeech_prepare import prepare_kdialectspeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_kdialectspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "splited_wav_folder": hparams["splited_wav_folder"],
            "save_folder": hparams["data_folder"],
            "province_code": hparams["province_code"],
            "data_ratio": hparams["data_ratio"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data, tokenizer = dataio_prepare(hparams)

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    print(f'asr_brain.device : {asr_brain.device}')

    # adding objects to trainer:
    # asr_brain.tokenizer = hparams["tokenizer"]
    asr_brain.tokenizer = tokenizer
    

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    asr_brain.hparams.wer_file = os.path.join(
        hparams["output_folder"], "wer_test.txt"
    )
    asr_brain.evaluate(
        test_data,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
