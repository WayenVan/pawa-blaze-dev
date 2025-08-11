from __future__ import annotations

import collections

from axolotl.integrations.base import BasePlugin
from axolotl.common.datasets import TrainDatasetMeta
from axolotl.cli.args import TrainerCliArgs

from axolotl.utils.dict import DictDefault
from typing import Union
from transformers.hf_argparser import HfArgumentParser


class PawaTrainingPlugin(BasePlugin):
    def load_datasets(
        self, cfg: DictDefault, preprocess: bool = False
    ) -> Union["TrainDatasetMeta", None]:
        parser = HfArgumentParser(TrainerCliArgs)
        parsed_cli_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
        print("Loading datasets with PawaTrainingPlugin...")
        return super().load_datasets(cfg, preprocess)
