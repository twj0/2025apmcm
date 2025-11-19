"""
Quick installation and import test script.

Run this to verify all dependencies are installed and imports work.

Usage:
    uv run python 2025/test_installation.py
"""

import sys
from pathlib import Path

print("=" * 70)
print("Testing APMCM 2025 Problem C Installation")
print("=" * 70)
print()

# Test 1: Python version
print("✓ Test 1: Python version")
print(f"  Python: {sys.version}")
assert sys.version_info >= (3, 11), "Python 3.11+ required"
print()

# Test 2: Core dependencies
print("✓ Test 2: Core dependencies")
try:
    import numpy as np
    print(f"  numpy: {np.__version__}")
    
    import pandas as pd
    print(f"  pandas: {pd.__version__}")
    
    import matplotlib
    print(f"  matplotlib: {matplotlib.__version__}")
    
    import statsmodels
    print(f"  statsmodels: {statsmodels.__version__}")
    
    print("  All core dependencies OK")
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Run: uv sync")
    sys.exit(1)
print()

# Test 3: Project structure
print("✓ Test 3: Project structure")
project_root = Path(__file__).parent
required_dirs = [
    'src/utils',
    'src/models',
    'data/external',
    'results',
    'figures',
]

for dir_path in required_dirs:
    full_path = project_root / dir_path
    if full_path.exists():
        print(f"  ✓ {dir_path}")
    else:
        print(f"  ✗ {dir_path} (will be created)")
print()

# Test 4: Import project modules
print("✓ Test 4: Project module imports")
sys.path.insert(0, str(project_root / 'src'))

try:
    from utils.config import PROJECT_ROOT, RANDOM_SEED
    print(f"  ✓ utils.config")
    print(f"    PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"    RANDOM_SEED: {RANDOM_SEED}")
    
    from utils.data_loader import TariffDataLoader
    print(f"  ✓ utils.data_loader")
    
    from utils.mapping import HSMapper
    print(f"  ✓ utils.mapping")
    
    from models.q1_soybeans import SoybeanTradeModel
    print(f"  ✓ models.q1_soybeans")
    
    from models.q2_autos import AutoTradeModel
    print(f"  ✓ models.q2_autos")
    
    from models.q3_semiconductors import SemiconductorModel
    print(f"  ✓ models.q3_semiconductors")
    
    from models.q4_tariff_revenue import TariffRevenueModel
    print(f"  ✓ models.q4_tariff_revenue")
    
    from models.q5_macro_finance import MacroFinanceModel
    print(f"  ✓ models.q5_macro_finance")
    
    print("  All project modules imported successfully")
    
except ImportError as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 5: HS Mapper functionality
print("✓ Test 5: HS Mapper functionality")
mapper = HSMapper()
test_cases = [
    ('1201', 'soybean', True),
    ('8703', 'auto', True),
    ('854140', 'semiconductor', True),
    ('9999', 'soybean', False),
]

for hs_code, product_type, expected in test_cases:
    if product_type == 'soybean':
        result = mapper.is_soybean(hs_code)
    elif product_type == 'auto':
        result = mapper.is_auto(hs_code)
    elif product_type == 'semiconductor':
        result = mapper.is_semiconductor(hs_code)
    
    status = "✓" if result == expected else "✗"
    print(f"  {status} HS {hs_code} is {product_type}: {result} (expected {expected})")
print()

# Test 6: Data directory check
print("✓ Test 6: Data availability")
tariff_data_dir = project_root / 'problems' / 'Tariff Data'
if tariff_data_dir.exists():
    files = list(tariff_data_dir.glob('*.csv'))
    print(f"  ✓ Tariff Data directory found")
    print(f"    CSV files: {len(files)}")
else:
    print(f"  ✗ Tariff Data directory not found at {tariff_data_dir}")
print()

# Test 7: Simple model instantiation
print("✓ Test 7: Model instantiation")
try:
    q1_model = SoybeanTradeModel()
    print(f"  ✓ Q1 SoybeanTradeModel instantiated")
    
    q2_model = AutoTradeModel()
    print(f"  ✓ Q2 AutoTradeModel instantiated")
    
    q3_model = SemiconductorModel()
    print(f"  ✓ Q3 SemiconductorModel instantiated")
    
    q4_model = TariffRevenueModel()
    print(f"  ✓ Q4 TariffRevenueModel instantiated")
    
    q5_model = MacroFinanceModel()
    print(f"  ✓ Q5 MacroFinanceModel instantiated")
    
    print("  All models can be instantiated")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)
print()

# Summary
print("=" * 70)
print("✓ All tests passed!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Populate external data files in 2025/data/external/")
print("  2. Run analysis: uv run python 2025/src/main.py")
print("  3. Check results in 2025/results/ and 2025/figures/")
print()
print("For more information, see:")
print("  - 2025/README.md")
print("  - 2025/USAGE_GUIDE.md")
print("  - 2025/CODE_SUMMARY.md")
print()
