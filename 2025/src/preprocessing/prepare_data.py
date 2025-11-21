"""
Data preparation script for APMCM 2025 Problem C.
Processes raw external data into structured formats for each question.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parents[1]))

from utils.config import DATA_EXTERNAL, DATA_PROCESSED, ensure_directories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_q1_data():
    """Prepare Q1 soybean trade data."""
    logger.info("Preparing Q1 data...")
    
    # Ensure output directory exists
    q1_dir = DATA_PROCESSED / 'q1'
    q1_dir.mkdir(parents=True, exist_ok=True)
    
    # Load monthly data
    monthly_file = DATA_EXTERNAL / 'q1_soybean_imports_comtrade_monthly.csv'
    if monthly_file.exists():
        monthly_data = pd.read_csv(monthly_file)
        
        # Process monthly data
        if 'Date' in monthly_data.columns:
            monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])
            monthly_data = monthly_data.sort_values('Date')
        
        # Save as q1_1.csv (monthly)
        monthly_data.to_csv(q1_dir / 'q1_1.csv', index=False)
        logger.info(f"Saved monthly data: {len(monthly_data)} records")
    
    # Create annual aggregation
    annual_data = []
    
    # Load China import data
    china_imports_files = [
        'china_imports_soybeans.csv',
        'china_imports_soybeans_official.csv',
        'q1_china_imports_soybeans_wits_candidate.csv'
    ]
    
    for file in china_imports_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            annual_data.append(df)
    
    if annual_data:
        # Combine and process annual data
        combined = pd.concat(annual_data, ignore_index=True)
        
        # Remove duplicates if any
        if 'year' in combined.columns:
            combined = combined.drop_duplicates(subset=['year'])
            combined = combined.sort_values('year')
        
        # Save as q1_0.csv (annual)
        combined.to_csv(q1_dir / 'q1_0.csv', index=False)
        logger.info(f"Saved annual data: {len(combined)} records")
    
    # Create supplementary data file
    tariffs_file = DATA_EXTERNAL / 'china_soybean_tariffs_candidate.csv'
    if tariffs_file.exists():
        tariffs = pd.read_csv(tariffs_file)
        tariffs.to_csv(q1_dir / 'q1_tariffs.csv', index=False)
        logger.info("Saved tariff data")
    
    logger.info("✓ Q1 data preparation completed")


def prepare_q2_data():
    """Prepare Q2 auto trade data."""
    logger.info("Preparing Q2 data...")
    
    q2_dir = DATA_PROCESSED / 'q2'
    q2_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto sales data
    auto_files = [
        'us_auto_sales_by_brand_complete.csv',
        'us_auto_sales_by_brand.csv',
        'us_auto_brand_origin_mapping.csv',
        'us_auto_indicators.csv',
        'us_auto_official_indicators_2015_2024.csv'
    ]
    
    for file in auto_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            output_name = file.replace('us_auto_', 'q2_')
            df.to_csv(q2_dir / output_name, index=False)
            logger.info(f"Processed {file}: {len(df)} records")
    
    # Official sales data
    official_files = [
        'us_motor_vehicle_retail_sales_official.csv',
        'us_total_light_vehicle_sales_official.csv'
    ]
    
    for file in official_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            output_name = 'q2_' + file.replace('us_', '').replace('_official', '')
            df.to_csv(q2_dir / output_name, index=False)
            logger.info(f"Processed {file}")
    
    logger.info("✓ Q2 data preparation completed")


def prepare_q3_data():
    """Prepare Q3 semiconductor data."""
    logger.info("Preparing Q3 data...")
    
    q3_dir = DATA_PROCESSED / 'q3'
    q3_dir.mkdir(parents=True, exist_ok=True)
    
    # Semiconductor data
    semi_files = [
        'us_semiconductor_output.csv',
        'us_semiconductor_output_index_official.csv',
        'hs_semiconductors_segmented.csv',
        'q3_us_semiconductor_policy_overall_candidate.csv'
    ]
    
    for file in semi_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            output_name = file.replace('us_semiconductor_', 'q3_').replace('hs_semiconductors_', 'q3_')
            df.to_csv(q3_dir / output_name, index=False)
            logger.info(f"Processed {file}: {len(df)} records")
    
    # Policy data
    policy_file = DATA_EXTERNAL / 'us_chip_policies.csv'
    if policy_file.exists():
        df = pd.read_csv(policy_file)
        df.to_csv(q3_dir / 'q3_policies.csv', index=False)
        logger.info("Processed policy data")
    
    logger.info("✓ Q3 data preparation completed")


def prepare_q4_data():
    """Prepare Q4 tariff revenue data."""
    logger.info("Preparing Q4 data...")
    
    q4_dir = DATA_PROCESSED / 'q4'
    q4_dir.mkdir(parents=True, exist_ok=True)
    
    # Tariff data
    tariff_files = [
        'q4_avg_tariff_by_year.csv',
        'q4_us_tariff_revenue_gemini3_scenario.csv',
        'q4_us_tariff_revenue_grok4_scenario.csv',
        'wb_tariff_mean_china_2015_2024.csv'
    ]
    
    for file in tariff_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            output_name = file.replace('wb_tariff_', 'q4_tariff_')
            df.to_csv(q4_dir / output_name, index=False)
            logger.info(f"Processed {file}: {len(df)} records")
    
    # JSON parameters
    json_files = [
        'q4_dynamic_import_params.json',
        'q4_tariff_scenarios.json'
    ]
    
    for file in json_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            output_path = q4_dir / file
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Copied {file}")
    
    logger.info("✓ Q4 data preparation completed")


def prepare_q5_data():
    """Prepare Q5 macro-financial data."""
    logger.info("Preparing Q5 data...")
    
    q5_dir = DATA_PROCESSED / 'q5'
    q5_dir.mkdir(parents=True, exist_ok=True)
    
    # Macro data
    macro_files = [
        'us_real_gdp_official.csv',
        'us_cpi_official.csv',
        'us_unemployment_rate_official.csv',
        'us_industrial_production_official.csv',
        'us_federal_funds_rate_official.csv',
        'us_treasury_10y_yield_official.csv',
        'us_sp500_index_official.csv',
        'us_macro_consolidated.csv'
    ]
    
    combined_macro = []
    for file in macro_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            output_name = 'q5_' + file.replace('us_', '').replace('_official', '')
            df.to_csv(q5_dir / output_name, index=False)
            logger.info(f"Processed {file}: {len(df)} records")
            combined_macro.append(df)
    
    # Reshoring data
    reshoring_files = [
        'us_reshoring.csv',
        'q5_us_reshoring_from_grok4_candidate.csv',
        'q5_us_macro_reshoring_from_gemini3_candidate.csv'
    ]
    
    for file in reshoring_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            output_name = file.replace('us_reshoring', 'q5_reshoring')
            df.to_csv(q5_dir / output_name, index=False)
            logger.info(f"Processed {file}")
    
    # Financial data
    financial_files = [
        'us_financial.csv',
        'q5_us_financial_from_grok4_candidate.csv',
        'q5_us_macro_from_grok4_candidate.csv'
    ]
    
    for file in financial_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            output_name = file.replace('us_financial', 'q5_financial')
            df.to_csv(q5_dir / output_name, index=False)
            logger.info(f"Processed {file}")
    
    # Retaliation index
    retaliation_files = [
        'retaliation_index.csv',
        'q5_retaliation_index_from_grok4_candidate.csv'
    ]
    
    for file in retaliation_files:
        filepath = DATA_EXTERNAL / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            df.to_csv(q5_dir / 'q5_retaliation.csv', index=False)
            logger.info(f"Processed {file}")
            break
    
    logger.info("✓ Q5 data preparation completed")


def validate_processed_data():
    """Validate that all required processed data exists."""
    logger.info("\nValidating processed data...")
    
    required_files = {
        'q1': ['q1_0.csv', 'q1_1.csv'],
        'q2': ['q2_sales_by_brand_complete.csv'],
        'q3': ['q3_output.csv'],
        'q4': ['q4_avg_tariff_by_year.csv'],
        'q5': ['q5_real_gdp.csv', 'q5_cpi.csv']
    }
    
    all_valid = True
    for q_num, files in required_files.items():
        q_dir = DATA_PROCESSED / q_num
        for file in files:
            filepath = q_dir / file
            if filepath.exists():
                size = filepath.stat().st_size
                logger.info(f"✓ {q_num}/{file} ({size:,} bytes)")
            else:
                logger.warning(f"✗ {q_num}/{file} missing")
                all_valid = False
    
    return all_valid


def main():
    """Main data preparation pipeline."""
    logger.info("=" * 70)
    logger.info("APMCM 2025 Problem C - Data Preparation")
    logger.info("=" * 70)
    
    # Ensure directories exist
    ensure_directories()
    
    # Prepare data for each question
    prepare_q1_data()
    prepare_q2_data()
    prepare_q3_data()
    prepare_q4_data()
    prepare_q5_data()
    
    # Validate
    if validate_processed_data():
        logger.info("\n✅ All data preparation completed successfully!")
    else:
        logger.warning("\n⚠️ Some data files are missing. Check logs above.")
    
    logger.info("\nProcessed data saved to: 2025/data/processed/")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
