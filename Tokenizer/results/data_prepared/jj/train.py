#!/usr/bin/env python

import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])


    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.utils.distributed.ddp_init_group(run_opts)

    from kdialectspeech_prepare import prepare_kdialectspeech

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

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

    hparams["tokenizer"]()