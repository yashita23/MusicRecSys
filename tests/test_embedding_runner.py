from dataclasses import dataclass

import pytest

from src.models.embedding_runner import EmbeddingSpec


@dataclass(frozen=True)
class ToyParams:
    size: int = 2
    decay: float = 0.9


def test_a_grid_without_decay_is_rejected_when_the_spec_is_built():
    with pytest.raises(ValueError, match="decay"):
        EmbeddingSpec(name="toy", params_cls=ToyParams, build=lambda params, random_seed: None, grid={"size": (2, 4)})


def test_a_grid_key_that_is_not_a_param_is_rejected():
    with pytest.raises(ValueError, match="window"):
        EmbeddingSpec(name="toy", params_cls=ToyParams, build=lambda params, random_seed: None, grid={"window": (3,), "decay": (0.5,)})
