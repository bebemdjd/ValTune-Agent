"""YAML config management and patching."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy
from agents.schemas import ConfigPatch
from agents.tuner_prompt import IMMUTABLE_FIELDS


class ConfigManager:
    """Manage YAML configs: load, patch, save."""
    
    def __init__(self, base_config_path: str, run_dir: str = "./runs"):
        """
        Initialize config manager.
        
        Args:
            base_config_path: Path to base.yaml
            run_dir: Directory to store run configs
        """
        self.base_config_path = Path(base_config_path)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Load YAML config file."""
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise RuntimeError(f"Failed to load YAML {yaml_path}: {e}")
    
    def save_yaml(self, config: Dict[str, Any], yaml_path: str) -> str:
        """Save config to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            return str(yaml_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save YAML {yaml_path}: {e}")
    
    def apply_patch(self, config: Dict[str, Any], patch: ConfigPatch) -> Dict[str, Any]:
        """
        Apply a patch to config, respecting constraints.
        
        Args:
            config: Current config dict
            patch: ConfigPatch object with new values
            
        Returns:
            Updated config dict
        """
        updated = deepcopy(config)
        
        # Map patch fields to YAML keys
        patch_dict = patch.dict(exclude_none=True)
        
        # Remove 'other_fields' key if empty
        other_fields = patch_dict.pop("other_fields", {})
        
        # Apply patch
        for key, value in patch_dict.items():
            if key in IMMUTABLE_FIELDS:
                print(f"Warning: Ignoring immutable field {key}")
                continue
            
            if value is not None:
                updated[key] = value
        
        # Merge other_fields if provided
        if other_fields:
            updated.update(other_fields)
        
        return updated
    
    def save_round_config(self, config: Dict[str, Any], round_id: int) -> str:
        """
        Save config for a specific round.
        
        Args:
            config: Config dict
            round_id: Round number
            
        Returns:
            Path to saved YAML
        """
        round_dir = self.run_dir / f"round_{round_id:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        yaml_path = round_dir / "config.yaml"
        return self.save_yaml(config, str(yaml_path))
    
    def load_previous_config(self, round_id: int) -> Optional[Dict[str, Any]]:
        """Load config from a previous round (for rollback)."""
        round_dir = self.run_dir / f"round_{round_id:03d}"
        yaml_path = round_dir / "config.yaml"
        
        if yaml_path.exists():
            return self.load_yaml(str(yaml_path))
        return None
