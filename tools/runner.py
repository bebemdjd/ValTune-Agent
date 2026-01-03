"""
Training runner: Execute training commands and manage run directories.

Responsibilities:
1. Generate independent run_dir based on config and round_number
2. Replace placeholders {config}, {run_dir} in training command template
3. Launch training process using subprocess
4. Capture training output (stdout/stderr)
5. Save round metadata (config + metrics + decision)
"""

import subprocess
import json
import yaml
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingRunner:
    """Execute training commands with config management."""
    
    def __init__(self, train_cmd_template: str, config_manager):
        """
        Initialize the training runner.
        
        Args:
            train_cmd_template: Training command template, e.g.:
                "python train.py --cfg {config} --run_dir {run_dir}"
            config_manager: ConfigManager instance for config file operations
        """
        self.train_cmd_template = train_cmd_template
        self.config_manager = config_manager
        self.runs_dir = Path("runs")
        self.runs_dir.mkdir(exist_ok=True)
        
    def launch_training(
        self, 
        config: Dict[str, Any], 
        round_num: int,
        timeout: Optional[int] = None
    ) -> Path:
        """
        Launch training process.
        
        Args:
            config: Configuration dict for current round
            round_num: Round number
            timeout: Training timeout in seconds, None for unlimited
            
        Returns:
            run_dir: Output directory path for this round
        """
        # Create run directory for this round
        run_dir = self.runs_dir / f"round_{round_num}"
        run_dir.mkdir(exist_ok=True, parents=True)
        
        # Save config as YAML file
        config_path = run_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved config to {config_path}")
        
        # Replace placeholders in command template
        train_cmd = self.train_cmd_template.format(
            config=str(config_path.absolute()),
            run_dir=str(run_dir.absolute())
        )
        
        logger.info(f"Executing command: {train_cmd}")
        
        # Launch training subprocess
        try:
            # Redirect command output to file
            stdout_file = run_dir / "train_stdout.log"
            stderr_file = run_dir / "train_stderr.log"
            
            with open(stdout_file, 'w', encoding='utf-8') as stdout_f, \
                 open(stderr_file, 'w', encoding='utf-8') as stderr_f:
                
                process = subprocess.run(
                    train_cmd,
                    shell=True,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    timeout=timeout,
                    cwd=str(run_dir),  # Run in run_dir
                    check=False  # Don't raise exception automatically
                )
            
            # Check if training succeeded
            if process.returncode != 0:
                logger.warning(
                    f"Training process exited with code {process.returncode}. "
                    f"Check {stderr_file} for errors."
                )
            else:
                logger.info(f"Training completed successfully in {run_dir}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Training timeout after {timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        return run_dir
    
    def save_round_metadata(
        self,
        round_num: int,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> None:
        """
        Save round metadata (config + metrics + decision).
        
        Args:
            round_num: Round number
            config: Configuration dict
            metrics: Metrics dict
            decision: Decision dict
        """
        run_dir = self.runs_dir / f"round_{round_num}"
        metadata_path = run_dir / "metadata.json"
        
        metadata = {
            "round": round_num,
            "config": config,
            "metrics": metrics,
            "decision": decision
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def get_run_dir(self, round_num: int) -> Path:
        """
        Get run directory for specified round.
        
        Args:
            round_num: Round number
            
        Returns:
            Run directory path
        """
        return self.runs_dir / f"round_{round_num}"
    
    def cleanup_runs(self, keep_last_n: int = 5) -> None:
        """
        Clean up old run directories, keeping only last N rounds.
        
        Args:
            keep_last_n: Number of recent rounds to keep
        """
        all_runs = sorted(self.runs_dir.glob("round_*"))
        
        if len(all_runs) > keep_last_n:
            for run_dir in all_runs[:-keep_last_n]:
                logger.info(f"Removing old run directory: {run_dir}")
                shutil.rmtree(run_dir)
    
    def load_round_metadata(self, round_num: int) -> Optional[Dict[str, Any]]:
        """
        Load metadata for specified round.
        
        Args:
            round_num: Round number
            
        Returns:
            Metadata dict, or None if not found
        """
        metadata_path = self.get_run_dir(round_num) / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata not found for round {round_num}")
            return None
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
