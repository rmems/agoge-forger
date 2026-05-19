import pytest
from agoge_forger.train.preflight import validate_lora_targets_exist

class DummyModule:
    def named_modules(self):
        return [("q_proj", self), ("v_proj", self)]

class DummyConfig:
    def __init__(self, targets, mode):
        self.target_modules = targets
        self.target_modules_mode = mode

def test_lora_target_validation_fails_on_missing_targets():
    model = DummyModule()
    
    # explicit failing
    cfg = DummyConfig(["non_existent"], "explicit")
    with pytest.raises(ValueError, match="Explicit target module 'non_existent' does not exist"):
        validate_lora_targets_exist(model, cfg)
        
    # discover required without targets
    cfg = DummyConfig([], "discover_required")
    with pytest.raises(ValueError, match="target_modules_mode is discover_required but no target_modules were provided"):
        validate_lora_targets_exist(model, cfg)
        
    # auto common works
    cfg = DummyConfig([], "auto_common")
    valid = validate_lora_targets_exist(model, cfg)
    assert set(valid) == {"q_proj", "v_proj"}
