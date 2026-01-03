"""
ValTune-Agent CLI: Main entry point.

Usage:
    python cli.py --base-config configs/base.yaml --train-cmd "python train.py --cfg {config} --run_dir {run_dir}"

Setup:
    export DEEPSEEK_API_KEY="your-api-key"
    python cli.py --base-config configs/base.yaml --train-cmd "..."
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool

from agents.schemas import TuneDecision, MetricsSnapshot, ConfigPatch
from agents.tuner_prompt import SYSTEM_PROMPT, TUNING_PROMPT_TEMPLATE, CONSTRAINTS, format_history
from agents.graph import TuningGraph
from tools.config_io import ConfigManager
from tools.metrics import MetricsParser
from tools.runner import TrainingRunner
from tools.hardware import HardwareAnalyzer


# Load environment variables (DEEPSEEK_API_KEY, etc.)
load_dotenv()


def setup_llm(api_key_override=None):
    """Initialize ChatDeepSeek with structured output."""
    # Priority: command line argument > environment variable
    api_key = api_key_override or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n⚠️  Error: DEEPSEEK_API_KEY not set")
        print("Please set using one of the following methods:")
        print("  1. Command line: --api-key 'your-key'")
        print("  2. Environment: $env:DEEPSEEK_API_KEY='your-key' (PowerShell)")
        print("  3. Environment: export DEEPSEEK_API_KEY='your-key' (Bash)")
        print("\n或使用 --mock-llm 参数进行测试（无需 API）")
        raise ValueError("DEEPSEEK_API_KEY not set in environment")
    
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
    )
    
    # Enable structured output with Pydantic schema
    # 输出严格对准 TuneDecision
    llm_with_schema = llm.with_structured_output(
        TuneDecision,
        method="json_schema",
        strict=True
    )
    
    return llm_with_schema




def main():
    parser = argparse.ArgumentParser(description="ValTune-Agent CLI")
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Path to base.yaml"
    )
    parser.add_argument(
        "--train-cmd",
        type=str,
        required=True,
        help='Training command template, e.g., "python train.py --cfg {config} --run_dir {run_dir}"'
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="Maximum training rounds"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout per training round (seconds)"
    )
    parser.add_argument(
        "--simple-loop",
        action="store_true",
        help="Use simple loop instead of LangGraph workflow"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-",
        help="DeepSeek API key (overrides environment variable)"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    config_manager = ConfigManager(args.base_config)
    metrics_parser = MetricsParser()
    runner = TrainingRunner(args.train_cmd, config_manager)
    llm = setup_llm(api_key_override=args.api_key)
    
    # Load base config
    base_config = config_manager.load_yaml(args.base_config)
    
    print(f"Starting ValTune-Agent loop (max {args.max_rounds} rounds)")
    print(f"Base config: {args.base_config}")
    print(f"Training command: {args.train_cmd}")
    print(f"Mode: {'Simple Loop' if args.simple_loop else 'LangGraph (Default)'}")
    print(f"LLM: DeepSeek API\n")
    
    # Default to LangGraph workflow unless --simple-loop is specified
    if not args.simple_loop:
        # Use LangGraph workflow (default mode)
        print("Using LangGraph workflow (default)...\n")
        
        # Initialize LangGraph with all components
        tuning_graph = TuningGraph(
            llm_client=llm,
            config_manager=config_manager,
            runner=runner,
            metrics_parser=metrics_parser
        )
        
        # Prepare initial state for workflow
        initial_state = {
            "base_config_path": args.base_config,
            "current_config": base_config,
            "round_number": 0,
            "metrics_history": [],
            "decisions": [],
            "should_stop": False,
            "max_rounds": args.max_rounds
        }
        
        # Execute the LangGraph workflow
        final_state = tuning_graph.run(initial_state)
        
        # Print final summary
        print(f"\n{'=' * 60}")
        print("Tuning Loop Completed (LangGraph)")
        print(f"{'=' * 60}")
        print(f"Total rounds: {len(final_state.get('decisions', []))}")
        metrics_history = final_state.get('metrics_history', [])
        if metrics_history:
            print(f"Best mAP50: {max([m.get('map_50', 0) for m in metrics_history if m.get('map_50')], default=0):.4f}")
            print(f"Best mAP50-95: {max([m.get('map_50_95', 0) for m in metrics_history if m.get('map_50_95')], default=0):.4f}")
    
    else:
        # Use simple sequential loop mode
        print("Using simple loop...\n")
        
        # Track state across rounds: metrics and decisions
        state = {
            "round_number": 0,
            "base_config": base_config,
            "metrics_history": [],
            "decisions": [],
        }
        
        for round_num in range(args.max_rounds):
            print(f"\n{'=' * 60}")
            print(f"Round {round_num}")
            print(f"{'=' * 60}")
            
            state["round_number"] = round_num
            
            # 1. Launch training with current configuration
            config = state["base_config"]
            print(f"Launching training with config:")
            for k, v in config.items():
                print(f"  {k}: {v}")
            
            run_dir = runner.launch_training(config, round_num, timeout=args.timeout)
            
            # 2. Parse metrics from training output
            metrics = metrics_parser.parse_metrics(str(run_dir))
            state["metrics_history"].append(metrics)
            print(f"\nValidation Metrics:")
            print(f"  mAP50:    {metrics.map_50:.4f}" if metrics.map_50 else "  mAP50:    N/A")
            print(f"  mAP50-95: {metrics.map_50_95:.4f}" if metrics.map_50_95 else "  mAP50-95: N/A")
            print(f"  Precision: {metrics.precision:.4f}" if metrics.precision else "  Precision: N/A")
            print(f"  Recall:    {metrics.recall:.4f}" if metrics.recall else "  Recall: N/A")
            print(f"  Val Loss:  {metrics.val_loss:.4f}" if metrics.val_loss else "  Val Loss: N/A")
            
            # 3. Query LLM for optimization decision
            history_text = format_history(state["decisions"], state["metrics_history"])
            
            prompt = TUNING_PROMPT_TEMPLATE.format(
                round_number=round_num,
                history=history_text,
                map_50=metrics.map_50 or 0.0,
                map_50_95=metrics.map_50_95 or 0.0,
                precision=metrics.precision or 0.0,
                recall=metrics.recall or 0.0,
                val_loss=metrics.val_loss or 0.0,
                epoch=metrics.epoch,
                current_config=json.dumps(config, indent=2)
            )
            
            print(f"\nQuerying LLM for optimization suggestions...")
            
            try:
                decision = llm.invoke(SYSTEM_PROMPT + "\n\n" + prompt)
                # Ensure decision is a TuneDecision object
                if not isinstance(decision, TuneDecision):
                    decision = TuneDecision(**decision)
                state["decisions"].append(decision)
            except Exception as e:
                print(f"Error querying LLM: {e}")
                break
            
            print(f"\nOptimization Decision:")
            print(f"  Action: {decision.action}")
            print(f"  Reason: {decision.reason}")
            if decision.patch:
                print(f"  Suggested Changes: {decision.patch.dict(exclude_none=True)}")
            
            # 4. Check if LLM recommends stopping
            if decision.action == "stop":
                print(f"\nConvergence detected - stopping tuning.")
                break
            
            # 5. Apply suggested changes for next round
            if decision.patch:
                state["base_config"] = config_manager.apply_patch(
                    state["base_config"],
                    decision.patch
                )
                print(f"\nConfiguration updated for next round")
            
            # Record this round's metadata
            runner.save_round_metadata(
                round_num,
                config,
                metrics.dict(),
                decision.dict() if hasattr(decision, 'dict') else decision
            )
        
        # Print final optimization summary
        print(f"\n{'=' * 60}")
        print("Hyperparameter Tuning Complete")
        print(f"{'=' * 60}")
        print(f"Total optimization rounds: {len(state['decisions'])}")
        print(f"Best mAP50: {max([m.map_50 for m in state['metrics_history'] if m.map_50], default=0):.4f}")
        print(f"Best mAP50-95: {max([m.map_50_95 for m in state['metrics_history'] if m.map_50_95], default=0):.4f}")


if __name__ == "__main__":
    main()

