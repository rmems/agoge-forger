import pytest

from agoge_forger.role_text import last_assistant_completion_start_char


def test_last_assistant_offset_skips_earlier_turns() -> None:
    text = "System: s\nUser: u1\nAssistant: a1\nUser: u2\nAssistant: final\n"
    start = last_assistant_completion_start_char(text)
    assert text[start:] == "Assistant: final\n"
    assert start == text.rfind("Assistant: ")


def test_embedded_assistant_word_is_not_a_header() -> None:
    text = "User: see Assistant: in docs\nAssistant: ok\n"
    assert text[last_assistant_completion_start_char(text) :] == "Assistant: ok\n"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "User: only\n",
        "Assistant:    \n",
        "Assistant:",
    ],
)
def test_last_assistant_offset_rejects_invalid_text(text: str) -> None:
    with pytest.raises(ValueError):
        last_assistant_completion_start_char(text)
