import pytest
from agoge_forger.datasets import normalize_row

def test_invalid_dataset_row_fails():
    with pytest.raises(ValueError, match="Line 1: 'text' field must be a string."):
        normalize_row({"text": 123}, index=1)
        
    with pytest.raises(ValueError, match="Line 2: 'messages' must be a list."):
        normalize_row({"messages": "hi"}, index=2)
        
    with pytest.raises(ValueError, match="Line 3: invalid role 'invalid'."):
        normalize_row({"messages": [{"role": "invalid", "content": "hi"}]}, index=3)
        
    with pytest.raises(ValueError, match="Line 4: Unknown format"):
        normalize_row({"unknown": "format"}, index=4)
