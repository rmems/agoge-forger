from agoge_forger.datasets import normalize_row

def test_normalize_text():
    row = {"text": "hello"}
    assert normalize_row(row) == {"text": "hello"}

def test_normalize_messages():
    row = {"messages": [{"role": "user", "content": "hi"}]}
    res = normalize_row(row)
    assert "User: hi" in res["text"] or "hi" in res["text"]

def test_normalize_instruction():
    row = {"instruction": "do this", "output": "done"}
    res = normalize_row(row)
    assert "Instruction: do this" in res["text"]
    assert "Output: done" in res["text"]
