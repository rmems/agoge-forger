from unittest.mock import MagicMock, patch

from agoge_forger.config import load_config
from agoge_forger.train.qlora import train_qlora


def test_train_qlora_honors_canary_yaml_load_in_4bit_true():
    config = load_config("configs/minicpm5_canary.yaml")
    assert config.quantization.load_in_4bit is True
    provenance = MagicMock()

    with (
        patch(
            "agoge_forger.train.qlora.producer_provenance_from_config",
            return_value=provenance,
        ),
        patch("agoge_forger.train.qlora.run_training") as mock_run,
    ):
        train_qlora(config)

    mock_run.assert_called_once_with(config, producer_provenance=provenance)
    passed = mock_run.call_args.args[0]
    assert passed.quantization.load_in_4bit is True
    assert config.quantization.load_in_4bit is True
