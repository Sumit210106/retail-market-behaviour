#!/usr/bin/env python3
"""Test script to debug the Apriori endpoint"""

from utils.data_loader import load_5lakh_data
from ml.apriori import run_apriori

print("=" * 60)
print("Testing Apriori Functionality")
print("=" * 60)

try:
    print("\n1. Loading data...")
    df = load_5lakh_data(nrows=10000)  # Start with just 10k rows
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    print("\n2. Running Apriori...")
    result = run_apriori(df, min_support=0.001, min_confidence=0.01)
    rules = result.get("rules", [])
    print(f"   ✓ {result.get('message', 'Done')}")
    
    if len(rules) > 0:
        print("\n3. Sample rules:")
        for i, rule in enumerate(rules[:3]):
            print(f"   Rule {i+1}: {rule}")
    else:
        print("\n   No rules found!")
        
    print("\n✅ Test completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
