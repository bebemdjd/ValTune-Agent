"""
Mock training script for testing ValTune-Agent.
Generates simulated training metrics without actual model training.
"""

import argparse
import json
import time
import random
from pathlib import Path
import yaml


def simulate_training(config_path: str, run_dir: str, base_map50: float = 0.60):
    """
    Simulate training and generate mock validation metrics.
    
    Args:
        config_path: Path to configuration YAML file
        run_dir: Output directory for results
        base_map50: Base mAP50 value for this round
    """
    print(f"[Mock Training] Starting simulation...")
    print(f"[Mock Training] Config: {config_path}")
    print(f"[Mock Training] Output: {run_dir}")
    
    # Create output directory
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract hyperparameters and adjust metrics
    lr = config.get('lr', 0.001)
    batch_size = config.get('batch', 16)
    weight_decay = config.get('weight_decay', 0.0005)
    epochs = config.get('epochs', 100)
    
    # Simulate training time
    train_time = min(2.0, epochs * 0.02)
    print(f"[Mock Training] Simulating {train_time:.1f} seconds...")
    time.sleep(train_time)
    
    # Calculate metric adjustments based on hyperparameters
    # Rules:
    # - Very high lr (>0.01) -> unstable, low mAP
    # - Very low lr (<0.0001) -> slow convergence, medium mAP
    # - Optimal lr (0.0001-0.005) -> high mAP
    # - Small batch (<8) -> unstable
    # - Large batch (>=16) -> better stability
    
    lr_factor = 1.0
    if lr > 0.01:  # Learning rate too high
        lr_factor = 0.85
    elif lr < 0.0001:  # Learning rate too low
        lr_factor = 0.90
    elif 0.0001 <= lr <= 0.005:  # Learning rate optimal
        lr_factor = 1.1
    
    batch_factor = 1.0
    if batch_size < 8:  # Batch size too small
        batch_factor = 0.92
    elif batch_size >= 16:  # Batch size good
        batch_factor = 1.05
    
    wd_factor = 1.0
    if weight_decay > 0.001:  # Weight decay too high
        wd_factor = 0.95
    elif 0.0003 <= weight_decay <= 0.0007:  # Weight decay optimal
        wd_factor = 1.03
    
    # Combine factors and add random noise
    total_factor = lr_factor * batch_factor * wd_factor
    noise = random.uniform(-0.02, 0.02)
    
    map_50 = min(0.95, max(0.40, base_map50 * total_factor + noise))
    map_50_95 = map_50 * random.uniform(0.65, 0.75)
    precision = map_50 * random.uniform(0.85, 0.95)
    recall = map_50 * random.uniform(0.80, 0.90)
    val_loss = max(0.1, 1.0 - map_50 + random.uniform(-0.1, 0.1))
    
    # Generate progressive epoch metrics
    results_list = []
    log_path = run_path / "train_stdout.log"
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Mock Training Session\n")
        f.write("=" * 60 + "\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write("\n")
        f.write("Training Progress:\n")
        
        # Generate progressive metrics for each epoch
        for ep in range(1, epochs + 1):
            # Simulate learning curve (fast early, slow later)
            progress_ratio = ep / epochs
            learning_curve = 0.3 + 0.7 * (1 - (1 - progress_ratio) ** 2)
            
            epoch_map50 = map_50 * learning_curve + random.uniform(-0.01, 0.01)
            epoch_map50_95 = epoch_map50 * random.uniform(0.65, 0.75)
            epoch_precision = epoch_map50 * random.uniform(0.85, 0.95)
            epoch_recall = epoch_map50 * random.uniform(0.80, 0.90)
            epoch_val_loss = max(0.1, 1.0 - epoch_map50 + random.uniform(-0.05, 0.05))
            epoch_train_loss = epoch_val_loss * random.uniform(1.1, 1.3)
            
            # Record to list
            results_list.append({
                "epoch": ep,
                "metrics/mAP50(B)": round(epoch_map50, 4),
                "metrics/mAP50-95(B)": round(epoch_map50_95, 4),
                "metrics/precision(B)": round(epoch_precision, 4),
                "metrics/recall(B)": round(epoch_recall, 4),
                "val/loss": round(epoch_val_loss, 4),
                "train/box_loss": round(epoch_train_loss * 0.6, 4),
                "train/cls_loss": round(epoch_train_loss * 0.4, 4),
                "lr/pg0": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay
            })
            
            # Print periodic progress
            if ep % 10 == 0 or ep == epochs:
                f.write(f"Epoch {ep:3d}/{epochs}: ")
                f.write(f"mAP50={epoch_map50:.4f}, ")
                f.write(f"mAP50-95={epoch_map50_95:.4f}, ")
                f.write(f"val_loss={epoch_val_loss:.4f}\n")
        
        f.write("\n")
        f.write("Final Results:\n")
        f.write(f"  mAP50:    {map_50:.4f}\n")
        f.write(f"  mAP50-95: {map_50_95:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  Val Loss:  {val_loss:.4f}\n")
        f.write("\n")
        f.write("Training completed successfully!\n")
    
    # Save complete epoch list to results.json
    results_path = run_path / "results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2)
    
    # Save final results (for compatibility)
    final_results = results_list[-1] if results_list else {
        "epoch": epochs,
        "metrics/mAP50(B)": round(map_50, 4),
        "metrics/mAP50-95(B)": round(map_50_95, 4),
        "metrics/precision(B)": round(precision, 4),
        "metrics/recall(B)": round(recall, 4),
        "val/loss": round(val_loss, 4),
        "train/box_loss": round(val_loss * 1.2, 4),
        "train/cls_loss": round(val_loss * 0.8, 4),
        "lr/pg0": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay
    }
    
    final_path = run_path / "final_results.json"
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"[Mock Training] Simulation complete!")
    print(f"[Mock Training] mAP50: {map_50:.4f}")
    print(f"[Mock Training] Results saved to: {results_path}")
    
    return map_50


def main():
    parser = argparse.ArgumentParser(description="Mock training script for ValTune-Agent testing")
    parser.add_argument('--cfg', '--config', dest='config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Output directory for training results')
    
    print("=" * 60)
    print("Mock Training Script")
    print("=" * 60)
    
    args = parser.parse_args()
    
    # Extract round number from run_dir path
    round_num = 0
    if 'round_' in args.run_dir:
        try:
            round_num = int(args.run_dir.split('round_')[1].split('/')[0].split('\\')[0])
        except:
            pass
    
    # Simulate gradual improvement across rounds
    base_map50 = 0.60 + round_num * 0.02  # 2% improvement per round
    base_map50 = min(0.75, base_map50)  # Cap at 75%
    
    simulate_training(args.config, args.run_dir, base_map50)


if __name__ == "__main__":
    main()
