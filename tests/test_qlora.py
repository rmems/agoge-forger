from unittest.mock import patch

from agoge_forger.config import load_config
from agoge_forger.train.qlora import train_qlora


def test_train_qlora_honors_smoke_yaml_load_in_4bit_false():
    config = load_config("configs/smoke_test.yaml")
    assert config.quantization.load_in_4bit is False

    with patch("agoge_forger.train.qlora.run_training") as mock_run:
        train_qlora(config)

    mock_run.assert_called_once()
    passed = mock_run.call_args.args[0]
    assert passed.quantization.load_in_4bit is False
    assert config.quantization.load_in_4bit is False


def test_train_qlora_honors_minicpm5_canary_load_in_4bit_true():
    config = load_config("configs/minicpm5_canary.yaml")
    assert config.quantization.load_in_4bit is True

    with patch("agoge_forger.train.qlora.run_training") as mock_run:
        train_qlora(config)

    mock_run.assert_called_once()
    passed = mock_run.call_args.args[0]
    assert passed.quantization.load_in_4bit is True
    assert config.quantization.load_in_4bit is True
