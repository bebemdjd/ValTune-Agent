"""Parse validation metrics from training output."""

import json
import csv
import re
from pathlib import Path
from typing import Optional, Dict, Any
from agents.schemas import MetricsSnapshot


class MetricsParser:
    """Parse metrics from various formats (JSON, CSV, log files)."""
    
    def __init__(self):
        pass
    
    def parse_metrics(self, run_dir: str) -> MetricsSnapshot:
        """
        Parse validation metrics from run directory.
        
        Tries multiple formats:
        1. results.json (standard format)
        2. val_metrics.json
        3. last line of results.csv
        4. Parse from log file (regex)
        
        Args:
            run_dir: Path to training run directory
            
        Returns:
            MetricsSnapshot with parsed metrics
        """
        run_path = Path(run_dir)
        
        # Try JSON files first
        for json_file in ["results.json", "val_metrics.json"]:
            json_path = run_path / json_file
            if json_path.exists():
                return self._parse_json(json_path)
        
        # Try CSV format
        csv_path = run_path / "results.csv"
        if csv_path.exists():
            return self._parse_csv(csv_path)
        
        # Try parsing from log
        log_path = run_path / "train.log"
        if log_path.exists():
            return self._parse_log(log_path)
        
        # Return empty metrics if nothing found
        return MetricsSnapshot(
            map_50=None,
            map_50_95=None,
            precision=None,
            recall=None,
            val_loss=None,
            epoch=0
        )
    
    def _parse_json(self, json_path: Path) -> MetricsSnapshot:
        """Parse metrics from JSON file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle list format (last entry is final)
            if isinstance(data, list):
                data = data[-1] if data else {}
            
            return MetricsSnapshot(
                map_50=data.get("metrics/mAP50(B)") or data.get("mAP50"),
                map_50_95=data.get("metrics/mAP50-95(B)") or data.get("mAP50-95"),
                precision=data.get("metrics/precision(B)") or data.get("precision"),
                recall=data.get("metrics/recall(B)") or data.get("recall"),
                val_loss=data.get("val/loss"),
                epoch=data.get("epoch", 0)
            )
        except Exception as e:
            print(f"Error parsing JSON {json_path}: {e}")
            return MetricsSnapshot(map_50=None, map_50_95=None, precision=None,
                                  recall=None, val_loss=None, epoch=0)
    
    def _parse_csv(self, csv_path: Path) -> MetricsSnapshot:
        """Parse metrics from CSV file (last row)."""
        try:
            rows = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                return MetricsSnapshot(map_50=None, map_50_95=None, precision=None,
                                      recall=None, val_loss=None, epoch=0)
            
            last_row = rows[-1]
            
            # Map CSV columns to schema (adjust based on your CSV format)
            return MetricsSnapshot(
                map_50=self._safe_float(last_row.get("metrics/mAP50(B)")),
                map_50_95=self._safe_float(last_row.get("metrics/mAP50-95(B)")),
                precision=self._safe_float(last_row.get("metrics/precision(B)")),
                recall=self._safe_float(last_row.get("metrics/recall(B)")),
                val_loss=self._safe_float(last_row.get("val/loss")),
                epoch=int(float(last_row.get("epoch", 0)))
            )
        except Exception as e:
            print(f"Error parsing CSV {csv_path}: {e}")
            return MetricsSnapshot(map_50=None, map_50_95=None, precision=None,
                                  recall=None, val_loss=None, epoch=0)
    
    def _parse_log(self, log_path: Path) -> MetricsSnapshot:
        """Parse metrics from log file using regex."""
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Example regex patterns (adjust based on your log format)
            patterns = {
                "map_50": r"mAP50[:\s]+([0-9.]+)",
                "map_50_95": r"mAP50-95[:\s]+([0-9.]+)",
                "precision": r"precision[:\s]+([0-9.]+)",
                "recall": r"recall[:\s]+([0-9.]+)",
                "val_loss": r"val_loss[:\s]+([0-9.]+)",
                "epoch": r"epoch[:\s]+(\d+)"
            }
            
            parsed = {}
            for key, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Take the last match
                    parsed[key] = float(matches[-1]) if key != "epoch" else int(matches[-1])
            
            return MetricsSnapshot(
                map_50=parsed.get("map_50"),
                map_50_95=parsed.get("map_50_95"),
                precision=parsed.get("precision"),
                recall=parsed.get("recall"),
                val_loss=parsed.get("val_loss"),
                epoch=parsed.get("epoch", 0)
            )
        except Exception as e:
            print(f"Error parsing log {log_path}: {e}")
            return MetricsSnapshot(map_50=None, map_50_95=None, precision=None,
                                  recall=None, val_loss=None, epoch=0)
    
    @staticmethod
    def _safe_float(val: Optional[str]) -> Optional[float]:
        """Safely convert string to float."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
