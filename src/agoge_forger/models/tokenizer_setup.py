"""Tokenizer-only setup shared by training and offline data preparation."""

from ..logging import logger


def prepare_training_tokenizer(tokenizer):
    """Use identical padding semantics with or without loading model weights."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token. Cannot safely pad.")
        logger.info("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
