#!/usr/bin/env python3
"""
 * N Park 2022 @ Starcell Inc.
"""
import sys
import torch
import logging
import speechbrain as sb
from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch):
        """Forward computations from the waveform batches
        to the output probabilities."""
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        # print(f'compute_forward self.modules.normalize ----- : \n {self.modules.normalize}')
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        # print(f'compute_forward self.modules.CNN ----- : \n {self.modules.CNN}')
        src = self.modules.CNN(feats)
        # print(f'compute_forward self.modules.Transformer ----- : \n {self.modules.Transformer}')
        # print(f'compute_forward src ----- : {src}')
        enc_out, pred = self.modules.Transformer( # pred : decoder out
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )
        # print(f'compute_forward enc_out ----- :\n{enc_out}')

        hyps = None
        hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens) # Valid
        # hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens) # Test
        return hyps

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


    ### for inferrence
    def transcribe_dataset(
            self,
            testdata,
            max_key, # We load the model with the lowest WER
            loader_kwargs # opts for the dataloading
        ):
        
        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(testdata, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            testdata = self.make_dataloader(
                testdata, sb.Stage.TEST, **loader_kwargs
            )

        self.on_evaluate_start(max_key=max_key) # We call the on_evaluate_start that will load the best model
        # self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            transcripts = []
            # for batch in tqdm(testdata, dynamic_ncols=True):
            for batch in testdata:

                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 
                out = self.compute_forward(batch)
                predicted_tokens = out

                # We go from tokens to words.
                tokenizer = hparams["tokenizer"]
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in predicted_tokens
                ]
                
                print(f'label : {batch.wrd}')
                print(f'hyp ----- : {predicted_words}')
                

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions."""
    data_folder = hparams["data_folder"]

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")
    
    # print(f'test_data.data.keys -----  : {test_data.data.keys()}')
    # print(f'test_data.data_ids[0] -----  : {test_data.data_ids[0]}')
    # print(f'test_data.data type -----  : {test_data.data[test_data.data_ids[0]]}')
    
    datasets = [test_data]

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
    return test_data, tokenizer


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # print(f'inference-1 ------------- : \n')
    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # 1.  # Dataset prep (parsing KsponSpeech)
    from ksponspeech_prepare import prepare_ksponspeech  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_ksponspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            # "splited_wav_folder": hparams["splited_wav_folder"],
            "save_folder": hparams["data_folder"],
            # "province_code": hparams["province_code"],
            # "data_ratio": hparams["data_ratio"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    test_data, tokenizer = dataio_prepare(hparams)

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

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]

    
    # Load best checkpoint for transcription !!!!!!
    # You need to create this function w.r.t your system architecture !!!!!!
    asr_brain.transcribe_dataset(
        test_data, # Must be obtained from the dataio_function
        max_key="ACC", # We load the model with the lowest WER
        loader_kwargs=hparams["test_dataloader_opts"], # opts for the dataloading
    )
