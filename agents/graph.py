"""LangGraph: training loop state machine."""

from langgraph.graph import StateGraph, END
from typing import Any, Dict, Optional
import logging

"""
LangGraph workflow for hyperparameter tuning.
State machine design: train -> evaluate -> decide -> apply -> (loop or stop)
Nodes execute tasks, Edges determine transitions, supporting iterative workflow with state evolution.

Workflow components:
- State: contains history, best_run, trial count, current config path
- Nodes: training, evaluation, decision, and patch application
- Conditional edges: stop on max_trials, minimal improvement, or performance degradation
"""



logger = logging.getLogger(__name__)


class TuningGraph:
    """
        LangGraph state machine for hyperparameter tuning loop.
        Closed-loop system ensuring continuous optimization.
    """
    
    def __init__(self, llm_client, config_manager, runner, metrics_parser):
        """
        Initialize the tuning graph.
        
        Args:
            llm_client: ChatDeepSeek or similar LangChain LLM for decision-making
            config_manager: ConfigManager instance for YAML read/write and patching
            runner: TrainingRunner instance for launching training
            metrics_parser: MetricsParser instance for parsing metrics
        """
        self.llm = llm_client
        self.config_manager = config_manager
        self.runner = runner
        self.metrics_parser = metrics_parser
        self.graph = self._build_graph()
    
    
    
    def _build_graph(self):
        """
            Build the LangGraph workflow state machine.
        """
        workflow = StateGraph(dict)  # State is a dict
        
        # Add nodes to the workflow
        workflow.add_node("load_config", self.node_load_config) # Load base YAML config
        workflow.add_node("train", self.node_train) # Launch training
        workflow.add_node("evaluate", self.node_evaluate) # Parse metrics
        workflow.add_node("decide", self.node_decide) # LLM decides next step (adjust params or stop)
        workflow.add_node("apply_patch", self.node_apply_patch) # Write LLM patch to config
        workflow.add_node("check_stopping", self.node_check_stopping) # Check stopping conditions
        
        # Add edges
        workflow.set_entry_point("load_config") # Entry point: start from load_config
        workflow.add_edge("load_config", "train") # Load config then train
        workflow.add_edge("train", "evaluate") # Train then evaluate
        workflow.add_edge("evaluate", "decide") # Evaluate then decide next step via LLM
        
        
        # Add conditional routing based on LLM decision to apply_patch or stop
        workflow.add_conditional_edges(
            "decide",
            self.route_decision,
            {"apply_patch": "apply_patch", "stop": END}
        )

        workflow.add_edge("apply_patch", "check_stopping") # After patch, check stopping conditions
        workflow.add_conditional_edges( # Decide to loop or end based on stopping conditions
            "check_stopping",
            self.route_continue, # Route to train (loop) or end
            {"loop": "train", "end": END}
        )
        
        return workflow.compile() # Compile the graph


    def node_load_config(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load base config from YAML file.
        """
        logger.info(f"[Node] load_config: round {state.get('round_number', 0)}")
        
        config_path = state.get("base_config_path")
        if config_path:
            config = self.config_manager.load_yaml(config_path) 
            state["current_config"] = config
        return state
    
    
    
    def node_train(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Launch training process."""
        logger.info(f"[Node] train: round {state.get('round_number', 0)}")
        
        round_num = state.get("round_number", 0) # Current round number
        config = state.get("current_config", {}) # Training parameters (lr, batch, wd, epoch, data path, etc.)
        
        # Write config and launch training
        run_dir = self.runner.launch_training(
            config=config,
            round_id=round_num
        )
        
        state["run_dir"] = run_dir
        
        return state
    
    
    
    def node_evaluate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse validation metrics from training output."""
        logger.info(f"[Node] evaluate: round {state.get('round_number', 0)}")
        
        run_dir = state.get("run_dir") # Training output directory
        if run_dir:
            metrics = self.metrics_parser.parse_metrics(run_dir) # Parse metrics file from run_dir
            state["last_metrics"] = metrics
            
            # Track history
            if "metrics_history" not in state:
                state["metrics_history"] = []
            # Add latest metrics
            state["metrics_history"].append(metrics)
        
        return state
    
    
    def node_decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM to decide next action."""
        logger.info(f"[Node] decide: round {state.get('round_number', 0)}")
        
        # Prepare context for LLM
        metrics = state.get("last_metrics") # Latest validation metrics
        config = state.get("current_config", {}) # Current training configuration
        
        # Call LLM with structured output to decide: action (stop/apply), patch (config changes), reason
        decision = self.llm.invoke(
            input={
                "round_number": state.get("round_number", 0),
                "metrics": metrics,
                "config": config,
                "history": state.get("metrics_history", [])
            }
        )
        
        state["last_decision"] = decision
        
        # Track decisions
        if "decisions" not in state:
            state["decisions"] = []
        state["decisions"].append(decision)
        
        return state
    
    
    
    def node_apply_patch(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply config patch from decision."""
        logger.info(f"[Node] apply_patch: round {state.get('round_number', 0)}")
        
        decision = state.get("last_decision") # Get decision from last decision node
        if decision and decision.get("action") != "stop":
            patch = decision.get("patch")
            if patch:
                config = state.get("current_config", {}) # Current training configuration
                # Apply patch to config (logic in config_manager)
                updated_config = self.config_manager.apply_patch(config, patch) # Apply patch to config
                state["current_config"] = updated_config # Update config for next round
        
        state["round_number"] = state.get("round_number", 0) + 1 # Increment round number
        
        return state
    
    
    
    
    
    def node_check_stopping(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check stopping conditions."""
        logger.info(f"[Node] check_stopping: round {state.get('round_number', 0)}")
        
        round_num = state.get("round_number", 0)
        decisions = state.get("decisions", [])
        
        # Check stopping conditions
        if round_num >= 20: # Max rounds reached - stop
            state["should_stop"] = True
            logger.info("Stopping: max rounds reached")
        elif len(decisions) >= 3: 
            # Check if last 3 decisions were 'stop' or 'rollback'
            last_actions = [d.get("action") for d in decisions[-3:]]
            # If last 3 actions are all rollback or stop, no improvement - stop
            if all(a in ["rollback", "stop"] for a in last_actions):
                state["should_stop"] = True
                logger.info("Stopping: no progress in last 3 rounds")
        else:
            state["should_stop"] = False 
        
        return state
    
    
    def route_decision(self, state: Dict[str, Any]) -> str:
        """Route based on LLM decision action."""
        decision = state.get("last_decision")
        action = decision.get("action", "stop") if decision else "stop"
        
        if action == "stop":
            return "stop"
        else:
            return "apply_patch"
    

    
    def route_continue(self, state: Dict[str, Any]) -> str:
        """Route based on stopping condition."""
        should_stop = state.get("should_stop", False)
        return "end" if should_stop else "loop"
    
    
    
    def run(self, initial_state: Dict[str, Any]):
        """
        Execute the tuning loop.
        """
        logger.info("Starting tuning loop...")
        final_state = self.graph.invoke(initial_state)
        logger.info("Tuning loop completed.")
        return final_state

