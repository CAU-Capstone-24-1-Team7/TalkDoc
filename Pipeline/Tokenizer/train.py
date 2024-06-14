#!/usr/bin/env python
"""Recipe for training a BPE tokenizer with kdialectspeech.
The tokenizer converts words into sub-word units that can
be used to train a language (LM) or an acoustic model (AM).
When doing a speech recognition experiment you have to make
sure that the acoustic and language models are trained with
the same tokenizer. Otherwise, a token mismatch is introduced
and beamsearch will produce bad results when combining AM and LM.
To run this recipe, do the following:
> python train.py hyperparams/5K_unigram_subword_bpe.yaml
Authors
 * Abdel Heba 2021
 * Dongwon Kim, Dongwoo Kim 2021
 * N Park 2022

*** kdialectspeech_prepare.py에서 ffmpeg필요 아래와 같이 설치해야 함.
apt install ffmpeg
pip install pydub # pydub는 ffmpeg를 필요로 함.
"""
# import logging
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

# logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])


    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
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

    # logger.info(f'---------------------------------------------')
    # logger.info(f'sys.argv[1:] : {sys.argv[1:]}')

    # logger.info(f'run_opts : {run_opts}')
    # logger.info(f'overrides : {overrides}')
    # logger.info(f'hparams["province_code"] : {hparams["province_code"]}')

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_kdialectspeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "splited_wav_folder": hparams["splited_wav_folder"],
            "save_folder": hparams["output_folder"],
            "province_code": hparams["province_code"],
            "data_ratio": hparams["data_ratio"],
        },
    )

    # Train tokenizer
    ### txt file은 speechbrain.tokenizers.SentencePiece 에서 만듬.
    ### _csv2text()
    hparams["tokenizer"]()