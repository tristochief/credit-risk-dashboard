
import os
import sys

print("Checking required files...")

required_files = [
    'credit_risk_dashboard.py',
    'credit_default_data.xls',
    'credit_data_preprocessed.csv',
    'risk_segmentation.csv',
    'feature_importance_lr.csv',
    'feature_importance_rf.csv',
    'feature_importance_xgb.csv',
    'early_warning_indicators.csv',
    'lr_model.pkl',
    'rf_model.pkl',
    'xgb_model.pkl',
    'scaler.pkl',
    'requirements.txt',
    'README.md'
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - MISSING")
        missing_files.append(file)

if missing_files:
    print(f"\n❌ Missing {len(missing_files)} files")
    sys.exit(1)
else:
    print("\n✅ All required files present!")
    print("\nYou can now run: streamlit run credit_risk_dashboard.py")
