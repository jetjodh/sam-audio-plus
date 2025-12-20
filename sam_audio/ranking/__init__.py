# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

from sam_audio.model.config import (
    ClapRankerConfig,
    EnsembleRankerConfig,
    ImageBindRankerConfig,
    JudgeRankerConfig,
)
from sam_audio.ranking.ranker import EnsembleRanker


def create_ranker(config):
    if isinstance(config, ImageBindRankerConfig):
        from sam_audio.ranking.imagebind import ImageBindRanker

        return ImageBindRanker(config)
    elif isinstance(config, ClapRankerConfig):
        from sam_audio.ranking.clap import ClapRanker

        return ClapRanker(config)
    elif isinstance(config, JudgeRankerConfig):
        from sam_audio.ranking.judge import JudgeRanker

        return JudgeRanker(config)
    elif isinstance(config, EnsembleRankerConfig):
        ranker_cfgs, weights = zip(*config.rankers.values(), strict=False)
        return EnsembleRanker(
            rankers=[create_ranker(cfg) for cfg in ranker_cfgs],
            weights=weights,
        )
    else:
        assert config is None
        return None
