#!/usr/bin/env python3
"""Quick test script for multi-modal health monitoring system.

This script provides a quick way to test the system functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.models import HealthMonitoringModel
        print("✅ Models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import models: {e}")
        return False
    
    try:
        from src.data import DataProcessor
        print("✅ Data modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import data modules: {e}")
        return False
    
    try:
        from src.eval import MedicalEvaluationMetrics
        print("✅ Evaluation modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import evaluation modules: {e}")
        return False
    
    try:
        from src.viz import HealthMonitoringVisualizer
        print("✅ Visualization modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import visualization modules: {e}")
        return False
    
    try:
        from src.utils import SafetyManager
        print("✅ Utility modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utility modules: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from src.models import HealthMonitoringModel
        model = HealthMonitoringModel(num_classes=6)
        print("✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False


def test_data_processing():
    """Test data processing."""
    print("\nTesting data processing...")
    
    try:
        from src.data import DataProcessor
        processor = DataProcessor()
        print("✅ Data processor created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create data processor: {e}")
        return False


def test_safety_system():
    """Test safety system."""
    print("\nTesting safety system...")
    
    try:
        from src.utils import SafetyManager
        with SafetyManager() as safety:
            warnings = safety.get_safety_warnings()
            print("✅ Safety system initialized successfully")
            print(f"   Safety warnings: {len(warnings)}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize safety system: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("MULTI-MODAL HEALTH MONITORING SYSTEM - QUICK TEST")
    print("=" * 60)
    print("IMPORTANT: This system is for research/educational purposes only.")
    print("NOT intended for clinical diagnosis or medical decision-making.")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_processing,
        test_safety_system,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run the Streamlit demo: streamlit run demo/app.py")
        print("2. Train the model: python scripts/train.py")
        print("3. Run inference: python scripts/inference.py --text 'Your health report'")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -e .")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
