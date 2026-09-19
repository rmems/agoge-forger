"""Offsets for role-capitalize chat dumps used as a single ``text`` field."""

_ASSISTANT_HEADER = "Assistant: "


def _line_start_assistant_offsets(text: str) -> list[int]:
    """Offsets of ``Assistant: `` headers that begin a line."""

    starts: list[int] = []
    cursor = 0
    while True:
        index = text.find(_ASSISTANT_HEADER, cursor)
        if index < 0:
            return starts
        # Include the role header in the completion so the Unicode offset
        # sits at a line start and does not bisect a tokenizer piece.
        if index == 0 or text[index - 1] == "\n":
            starts.append(index)
        cursor = index + 1


def last_assistant_completion_start_char(text: str) -> int:
    """Return the Unicode offset of the last assistant message body.

    ``role-capitalize-v1`` dumps lines as ``System: `` / ``User: `` /
    ``Assistant: ``. Loss and exact-match scoring apply to the final assistant
    turn including the ``Assistant: `` header so the offset is a line start.
    """

    if not isinstance(text, str) or not text:
        raise ValueError("role-capitalize text must be a nonempty string")
    starts = _line_start_assistant_offsets(text)
    if not starts:
        raise ValueError("role-capitalize text has no Assistant header")
    start = starts[-1]
    if not text[start + len(_ASSISTANT_HEADER) :].strip():
        raise ValueError("last assistant completion must be nonempty")
    return start
