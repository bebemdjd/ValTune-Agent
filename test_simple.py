"""
Simple module test script - Quick validation of core components
"""

import sys
from pathlib import Path

print("=" * 60)
print("ValTune-Agent Module Test")
print("=" * 60)
print()

# Test 1: Import modules
print("1️⃣  Testing module imports...")
try:
    from agents.schemas import MetricsSnapshot, ConfigPatch, TuneDecision
    from agents.tuner_prompt import SYSTEM_PROMPT, CONSTRAINTS
    from tools.config_io import ConfigManager
    from tools.metrics import MetricsParser
    from tools.runner import TrainingRunner
    print("   ✅ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Configuration management
print("2️⃣  Testing configuration management...")
try:
    config_manager = ConfigManager("configs/base.yaml")
    config = config_manager.load_yaml("configs/base.yaml")
    print(f"   ✅ Config loaded successfully, {len(config)} fields")
    print(f"   📝 Sample: lr={config.get('lr')}, batch={config.get('batch')}")
except Exception as e:
    print(f"   ❌ Config management failed: {e}")

print()

# Test 3: Generate mock data
print("3️⃣  Generating mock training data...")
try:
    import subprocess
    import shutil
    
    test_dir = Path("runs/test_round_0")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Run mock_train.py
    cmd = [
        sys.executable,
        "mock_train.py",
        "--cfg", "configs/base.yaml",
        "--run_dir", str(test_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and (test_dir / "results.json").exists():
        print("   ✅ Mock data generated successfully")
        print(f"   📂 Output dir: {test_dir}")
    else:
        print(f"   ❌ Generation failed: {result.stderr}")
        
except Exception as e:
    print(f"   ❌ 生成失败: {e}")

print()

# Test 4: Metrics parsing
print("4️⃣  Testing metrics parsing...")
try:
    metrics_parser = MetricsParser()
    metrics = metrics_parser.parse_metrics(str(test_dir))
    
    if metrics.map_50:
        print("   ✅ Metrics parsed successfully")
        print(f"   📊 Results:")
        print(f"      - mAP50: {metrics.map_50:.4f}")
        print(f"      - mAP50-95: {metrics.map_50_95:.4f}")
        print(f"      - Precision: {metrics.precision:.4f}")
        print(f"      - Recall: {metrics.recall:.4f}")
    else:
        print("   ⚠️  Metrics parsing returned empty values")
        
except Exception as e:
    print(f"   ❌ 指标解析失败: {e}")

print()

# Test 5: Configuration patching
print("5️⃣  Testing configuration patching...")
try:
    patch = ConfigPatch(lr=0.0005, batch_size=32)
    updated_config = config_manager.apply_patch(config, patch)
    
    if updated_config['lr'] == 0.0005 and updated_config['batch'] == 32:
        print("   ✅ Config patching successful")
        print(f"   📝 Updated: lr={updated_config['lr']}, batch={updated_config['batch']}")
    else:
        print("   ⚠️  Config patch not applied correctly")
        
except Exception as e:
    print(f"   ❌ 配置补丁失败: {e}")

print()

# Test 6: Pydantic models
print("6️⃣  Testing Pydantic models...")
try:
    decision = TuneDecision(
        action="improve",
        reason="Test decision",
        patch=ConfigPatch(lr=0.001),
        next_epochs=10
    )
    
    decision_dict = decision.dict(exclude_none=True)
    print("   ✅ Pydantic model validation successful")
    print(f"   📝 Decision: action={decision.action}, reason={decision.reason}")
    
except Exception as e:
    print(f"   ❌ Model validation failed: {e}")

print()

# Summary
print("=" * 60)
print("🎉 Module Testing Complete!")
print()
print("💡 Next Steps:")
print("   1. Run: python test_workflow.py for full workflow test")
print("   2. Set DEEPSEEK_API_KEY to test real LLM calls")
print("   3. Use: python cli.py --help for all options")
print("=" * 60)
