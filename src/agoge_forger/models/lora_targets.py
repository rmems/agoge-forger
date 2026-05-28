import re
import json
import os
from .load import load_base_model
from ..logging import logger

def inspect_lora_targets(model_id: str, trust_remote_code: bool = True, out_path: str = None):
    logger.info(f"Finding potential LoRA targets in {model_id}...")
    model, _ = load_base_model(model_id, trust_remote_code, quant_config=None, bf16=True)
    
    leaf_groups = {}
    full_module_matches = []
    
    patterns = [r"q_proj", r"k_proj", r"v_proj", r"o_proj", r"gate_proj", r"up_proj", r"down_proj", r"linear", r"gka", r"kalman", r"ssm"]
    
    for name, module in model.named_modules():
        mod_class = type(module).__name__
        # Just grab linear or specific patterned modules
        if any(re.search(p, name.lower()) for p in patterns) or "Linear" in mod_class:
            leaf = name.split(".")[-1]
            
            if leaf not in leaf_groups:
                leaf_groups[leaf] = {"count": 0, "types": set()}
            
            leaf_groups[leaf]["count"] += 1
            leaf_groups[leaf]["types"].add(mod_class)
            
            params = sum(p.numel() for p in module.parameters())
            dtype = str(next(module.parameters()).dtype) if params > 0 else "unknown"
            
            full_module_matches.append({
                "name": name,
                "leaf": leaf,
                "class": mod_class,
                "params": params,
                "dtype": dtype
            })
            
    # Convert sets to lists for JSON
    for leaf in leaf_groups:
        leaf_groups[leaf]["types"] = list(leaf_groups[leaf]["types"])
        
    common_targets = [k for k in leaf_groups.keys() if k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]]
    
    report = {
        "model_id": model_id,
        "architectures": getattr(model.config, "architectures", []),
        "candidate_leaf_names": leaf_groups,
        "full_module_matches_count": len(full_module_matches),
        "recommended_common_targets": common_targets,
        "custom_architecture_warnings": [
            "Always inspect 'gka', 'kalman', or 'ssm' modules before adding them to target_modules."
        ]
    }
    
    logger.info("Candidate LoRA target modules grouped by leaf name:")
    for leaf, info in leaf_groups.items():
        logger.info(f" - {leaf}: count={info['count']}, types={info['types']}")
        
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote full report to {out_path}")
        
    return report
