#!/usr/bin/env python3
"""
Add tariff rates to q1_1_final.csv based on partner countries
"""

import pandas as pd
import numpy as np
from pathlib import Path

def add_tariff_rates():
    """Add tariff rates to the final CSV file"""
    
    # File paths
    input_file = r"d:\Mathematical Modeling\2025APMCM\SPEC\2025\data\processed\q1\q1_1_final.csv"
    output_file = r"d:\Mathematical Modeling\2025APMCM\SPEC\2025\data\processed\q1\q1_1_final_with_tariffs.csv"
    
    print(f"Reading data from: {input_file}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check current tariff rate distribution
    print(f"\nCurrent tariff rate distribution:")
    tariff_counts = df['tariff_rate'].value_counts(dropna=False)
    print(tariff_counts)
    
    # Define tariff rates by partner
    # Based on previous analysis and trade policy data
    tariff_mapping = {
        'USA': 0.03,      # 3% base tariff for USA soybeans
        'Brazil': 0.03,   # 3% base tariff for Brazil soybeans  
        'Argentina': 0.03 # 3% base tariff for Argentina soybeans
    }
    
    # Apply tariff rates based on partner
    df['tariff_rate'] = df['partner_desc'].map(tariff_mapping)
    
    # Handle any missing mappings
    missing_tariffs = df[df['tariff_rate'].isna()]
    if len(missing_tariffs) > 0:
        print(f"\nWarning: {len(missing_tariffs)} records have missing tariff rates")
        print("Missing partners:", missing_tariffs['partner_desc'].unique())
        # Fill missing with default rate
        df['tariff_rate'] = df['tariff_rate'].fillna(0.03)
    
    # Verify the new tariff distribution
    print(f"\nNew tariff rate distribution:")
    new_tariff_counts = df['tariff_rate'].value_counts().sort_index()
    print(new_tariff_counts)
    
    # Calculate summary statistics
    print(f"\nTariff Rate Summary by Partner:")
    tariff_summary = df.groupby('partner_desc')['tariff_rate'].agg(['count', 'mean', 'min', 'max'])
    print(tariff_summary)
    
    # Save the updated file
    try:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nSuccessfully saved updated file to: {output_file}")
        print(f"File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
        
        # Verify the save
        df_check = pd.read_csv(output_file)
        print(f"Verification: {len(df_check)} records saved successfully")
        print(f"Tariff rates applied: {df_check['tariff_rate'].notna().sum()}/{len(df_check)}")
        
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Generate a summary report
    print(f"\n" + "="*60)
    print("TARIFF RATE ADDITION SUMMARY")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total records processed: {len(df)}")
    print(f"Records with tariff rates applied: {df['tariff_rate'].notna().sum()}")
    print(f"Unique tariff rates: {df['tariff_rate'].nunique()}")
    print(f"Tariff rate range: {df['tariff_rate'].min():.3f} - {df['tariff_rate'].max():.3f}")
    
    print(f"\nPartner distribution:")
    partner_counts = df['partner_desc'].value_counts()
    for partner, count in partner_counts.items():
        avg_tariff = df[df['partner_desc'] == partner]['tariff_rate'].mean()
        print(f"  {partner}: {count} records, avg tariff: {avg_tariff:.3f}")

if __name__ == "__main__":
    add_tariff_rates()