"""Pydantic schemas for tuning decisions and state."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class MetricsSnapshot(BaseModel):
    """Single validation metrics snapshot."""
    map_50: Optional[float] = Field(default=None, description="mAP@50")
    map_50_95: Optional[float] = Field(default=None, description="mAP@50-95")
    precision: Optional[float] = Field(default=None)
    recall: Optional[float] = Field(default=None)
    val_loss: Optional[float] = Field(default=None)
    epoch: int = Field(description="Epoch number")
    timestamp: Optional[str] = Field(default=None)


class ConfigPatch(BaseModel):
    """Configuration patch to apply."""
    lr: Optional[float] = Field(default=None, description="Learning rate")
    batch_size: Optional[int] = Field(default=None, description="Batch size")
    weight_decay: Optional[float] = Field(default=None, description="Weight decay")
    epochs: Optional[int] = Field(default=None, description="Total epochs")
    imgsz: Optional[int] = Field(default=None, description="Image size")
    other_fields: Optional[Dict[str, Any]] = Field(default=None, description="Other config fields")

    class Config:
        json_schema_extra = {
            "example": {
                "lr": 0.001,
                "batch_size": 16,
                "weight_decay": 0.0005
            }
        }


class TuneDecision(BaseModel):
    """LLM output: structured tuning decision."""
    action: str = Field(
        description="One of: 'improve', 'adjust', 'rollback', 'stop'"
    )
    reason: str = Field(
        description="Explanation of why this decision was made"
    )
    patch: Optional[ConfigPatch] = Field(
        default=None,
        description="Config changes to apply (only for 'improve'/'adjust')"
    )
    next_epochs: Optional[int] = Field(
        default=None,
        description="Number of epochs to run in next round"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "action": "adjust",
                "reason": "mAP50 plateaued, reduce LR to fine-tune",
                "patch": {
                    "lr": 0.0005,
                    "batch_size": None,
                    "weight_decay": None,
                    "epochs": None,
                    "imgsz": None
                },
                "next_epochs": 10
            }
        }


class TrainingRound(BaseModel):
    """Record of a single training round."""
    round_id: int = Field(description="Round number (0-indexed)")
    config_path: str = Field(description="Path to config YAML used")
    metrics: MetricsSnapshot = Field(description="Final validation metrics")
    decision: TuneDecision = Field(description="Tuning decision made")
    was_rolled_back: bool = Field(default=False, description="If decision was reverted")


class AgentState(BaseModel):
    """LangGraph state for the tuning loop."""
    round_number: int = Field(default=0)
    base_config_path: str = Field(description="Path to base YAML")
    current_config: Dict[str, Any] = Field(default_factory=dict)
    metrics_history: List[MetricsSnapshot] = Field(default_factory=list)
    decisions: List[TuneDecision] = Field(default_factory=list)
    should_continue: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)
    last_improvement_round: int = Field(default=-1, description="Last round with improvement")
