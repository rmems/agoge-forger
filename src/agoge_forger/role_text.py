"""Offsets for role-capitalize chat dumps used as a single ``text`` field."""

_ASSISTANT_HEADER = "Assistant: "


def last_assistant_completion_start_char(text: str) -> int:
    """Return the Unicode offset of the last assistant message body.

    ``role-capitalize-v1`` dumps lines as ``System: `` / ``User: `` /
    ``Assistant: ``. Loss and exact-match scoring apply to the final assistant
    turn including the ``Assistant: `` header so the offset is a line start.
    """

    if not isinstance(text, str) or not text:
        raise ValueError("role-capitalize text must be a nonempty string")
    start: int | None = None
    cursor = 0
    while True:
        index = text.find(_ASSISTANT_HEADER, cursor)
        if index < 0:
            break
        if index == 0 or text[index - 1] == "\n":
            # Include the role header in the completion so the Unicode offset
            # sits at a line start and does not bisect a tokenizer piece.
            start = index
        cursor = index + 1
    if start is None:
        raise ValueError("role-capitalize text has no Assistant header")
    body = text[start + len(_ASSISTANT_HEADER) :]
    if start >= len(text) or not body.strip():
        raise ValueError("last assistant completion must be nonempty")
    return start
