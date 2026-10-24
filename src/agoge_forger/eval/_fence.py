"""Markdown code-fence extraction shared by the format-invariant scorers."""

from __future__ import annotations

import re

# CommonMark fence line: up to three spaces of indent, then 3+ backticks or tildes.
# A backtick fence's info string (the language tag) may not itself contain backticks.
_OPEN_BACKTICK = re.compile(r"^ {0,3}(`{3,})[^`]*$")
_OPEN_TILDE = re.compile(r"^ {0,3}(~{3,})[^\n]*$")
_CLOSE = {"`": re.compile(r"^ {0,3}(`{3,})[ \t]*$"), "~": re.compile(r"^ {0,3}(~{3,})[ \t]*$")}


_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _opening_fence(line: str) -> str | None:
    """Return the fence run (for example ```` ``` ````) when ``line`` opens a fence."""

    for pattern in (_OPEN_BACKTICK, _OPEN_TILDE):
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _is_closing_fence(line: str, opening: str) -> bool:
    match = _CLOSE[opening[0]].match(line)
    return match is not None and len(match.group(1)) >= len(opening)


def _drop_leading_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    while start < len(lines) and lines[start].strip() == "":
        start += 1
    return lines[start:]


def extract_fenced_code(text: str) -> str:
    """Return the code in a reply that leads with a markdown fenced block.

    ``exact-match-v1`` compares raw text, so a correct module wrapped in a fence is
    scored wrong. This keeps the comparison exact but removes the wrapper:

    * CRLF, lone CR, and LF all count as line breaks when looking for fences.
    * The reply "leads with a fence" when its first non-blank line opens one (three or
      more backticks or tildes, with or without a language tag). Blank lines before the
      fence are ignored.
    * Anything else is returned exactly as ``exact-match-v1`` would see it (CRLF becomes
      LF, trailing whitespace is dropped, nothing else changes, leading blank lines
      included), including replies whose fenced block comes after introductory prose.
      That keeps ``exact-match-v2`` equal to v1 plus fence unwrapping.
    * The block runs to the first closing fence: same character, at least as long as
      the opening one, nothing else on the line (CommonMark). To keep a literal fence
      inside the code, the model must open with a longer fence.
    * **Multiple blocks: the first one wins.** Whatever follows it (an explanation, a
      second "corrected" block) is ignored. Models that ignore "reply with the module
      and nothing else" are therefore scored on their first code block, not penalized
      for the chatter after it.
    * An opening fence that is never closed (a reply cut off by the token budget) is
      treated as a block that runs to the end of the text.
    """

    unchanged = text.replace("\r\n", "\n").rstrip()
    lines = _drop_leading_blank_lines(_LINE_BREAK.split(text.rstrip()))
    if not lines:
        return ""
    opening = _opening_fence(lines[0])
    if opening is None:
        return unchanged
    body = lines[1:]
    for index, line in enumerate(body):
        if _is_closing_fence(line, opening):
            body = body[:index]
            break
    return "\n".join(_drop_leading_blank_lines(body)).rstrip()
