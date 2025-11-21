"""
Q2: Japanese Automobiles in the U.S. Market

This module implements import-structure panel regressions and scenario-based
modeling of Japanese FDI and production shifts to analyze the impact of
tariffs on auto trade and U.S. domestic industry.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import (
    RESULTS_DIR,
    FIGURES_DIR,
    DATA_PROCESSED,
    RESULTS_LOGS,
)
from utils.data_loader import TariffDataLoader
from utils.mapping import HSMapper

logger = logging.getLogger(__name__)


class AutoTradeModel:
    """Model for analyzing auto trade and industry impacts."""
    
    def __init__(self):
        """Initialize the model."""
        self.loader = TariffDataLoader()
        self.mapper = HSMapper()
        self.import_data: Optional[pd.DataFrame] = None
        self.industry_data: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.results: Dict = {}
        
    def load_q2_data(self) -> pd.DataFrame:
        """Load and prepare data for Q2 analysis.
        
        Returns:
            Panel DataFrame with auto import data by partner
        """
        logger.info("Loading Q2 auto data")
        
        # Load imports data
        imports = self.loader.load_imports()
        
        # Filter for autos
        imports_tagged = self.mapper.tag_dataframe(imports)
        autos = imports_tagged[imports_tagged['is_auto']].copy()
        
        logger.info(f"Loaded {len(autos)} auto import records")
        
        # Aggregate by year and partner
        autos_agg = autos.groupby(['year', 'partner_country']).agg({
            'duty_collected': 'sum',
        }).reset_index()
        
        autos_agg.rename(columns={'duty_collected': 'auto_import_charges'}, inplace=True)
        
        self.import_data = autos_agg
        
        return autos_agg
    
    def load_external_auto_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load external auto sales and industry data.
        
        Returns:
            Tuple of (sales_data, industry_data)
        """
        # Auto sales by brand
        sales_file = DATA_EXTERNAL / 'us_auto_sales_by_brand.csv'
        if not sales_file.exists():
            logger.warning(f"Sales data not found: {sales_file}")
            # Create template
            template_sales = pd.DataFrame({
                'year': [2020, 2021, 2022],
                'brand': ['Toyota', 'Honda', 'Ford'],
                'total_sales': [0, 0, 0],
                'us_produced': [0, 0, 0],
                'mexico_produced': [0, 0, 0],
                'japan_imported': [0, 0, 0],
            })
            template_sales.to_csv(sales_file, index=False)
            logger.info(f"Template saved to {sales_file}")
        else:
            template_sales = pd.read_csv(sales_file)
        
        # Industry indicators
        industry_file = DATA_EXTERNAL / 'us_auto_indicators.csv'
        if not industry_file.exists():
            logger.warning(f"Industry data not found: {industry_file}")
            # Create template
            template_industry = pd.DataFrame({
                'year': [2020, 2021, 2022, 2023, 2024],
                'us_auto_production': [0, 0, 0, 0, 0],
                'us_auto_employment': [0, 0, 0, 0, 0],
                'us_auto_price_index': [100, 102, 105, 108, 110],
                'us_gdp_billions': [0, 0, 0, 0, 0],
                'fuel_price_index': [100, 95, 110, 105, 100],
            })
            template_industry.to_csv(industry_file, index=False)
            logger.info(f"Template saved to {industry_file}")
        else:
            template_industry = pd.read_csv(industry_file)
        
        self.industry_data = template_industry
        
        return template_sales, template_industry
    
    def estimate_import_structure_model(
        self,
        panel_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Estimate import structure model.
        
        Model: ln(M_j / M_ROW) = δ_j + φ1·τ_j + φ2·X_t + e_j,t
        
        Args:
            panel_df: Import panel data
            
        Returns:
            Model results dictionary
        """
        if panel_df is None:
            panel_df = self.import_data
        
        if panel_df is None or len(panel_df) == 0:
            logger.error("No import data available")
            return {}
        
        logger.info("Estimating import structure model")
        
        # Add necessary variables for estimation
        # For demonstration, assume we have tariff data merged
        # panel_df['effective_tariff'] = ...
        # panel_df['ln_import_value'] = ...
        
        # Placeholder: simple model with available data
        try:
            # Assume we compute import shares
            total_by_year = panel_df.groupby('year')['auto_import_charges'].transform('sum')
            panel_df = panel_df.copy()
            panel_df['import_share'] = panel_df['auto_import_charges'] / total_by_year
            panel_df['ln_import_share'] = np.log(panel_df['import_share'] + 1e-6)
            
            # Simple trend model as placeholder
            formula = 'ln_import_share ~ year + C(partner_country)'
            model = smf.ols(formula, data=panel_df).fit()
            
            self.models['import_structure'] = model
            
            results = {
                'rsquared': float(model.rsquared),
                'nobs': int(model.nobs),
                'params': {k: float(v) for k, v in model.params.items()},
                'pvalues': {k: float(v) for k, v in model.pvalues.items()},
            }
            
            logger.info(f"Import structure model R²: {results['rsquared']:.3f}")
            
        except Exception as e:
            logger.error(f"Error estimating import structure: {e}")
            results = {}
        
        # Save results
        output_file = RESULTS_DIR / 'q2_import_structure.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_file}")
        
        return results
    
    def estimate_industry_transmission_model(
        self,
        industry_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Estimate impact of import penetration on domestic industry.
        
        Model: Y_US = θ0 + θ1·ImportPenetration + θ2·Z_t + ν_t
        
        Args:
            industry_df: Industry indicators DataFrame
            
        Returns:
            Model results dictionary
        """
        if industry_df is None:
            _, industry_df = self.load_external_auto_data()
        
        if industry_df is None or len(industry_df) == 0:
            logger.warning("No industry data available")
            return {}
        
        logger.info("Estimating industry transmission model with import penetration")
        
        # Ensure we have import data aggregated by year
        if self.import_data is None:
            self.load_q2_data()
        
        if self.import_data is None or len(self.import_data) == 0:
            logger.warning("No import data available for import penetration computation")
            return {}
        
        # Aggregate auto imports proxy (charges) by year
        imports_by_year = (
            self.import_data
            .groupby('year')['auto_import_charges']
            .sum()
            .reset_index(name='auto_import_charges_total')
        )
        
        # Merge imports proxy with industry indicators
        df = industry_df.merge(imports_by_year, on='year', how='left')
        
        if 'us_auto_production' not in df.columns:
            logger.warning("us_auto_production column missing in industry data")
            return {}
        
        # Compute import penetration: imports / (imports + domestic production)
        df['auto_import_charges_total'] = df['auto_import_charges_total'].fillna(0)
        df['denominator'] = df['auto_import_charges_total'] + df['us_auto_production']
        df = df[df['denominator'] > 0].copy()
        
        if df.empty:
            logger.warning("No valid observations for import penetration computation")
            return {}
        
        df['import_penetration'] = df['auto_import_charges_total'] / df['denominator']
        
        try:
            # Prefer a model with controls when available
            if {'us_gdp_billions', 'fuel_price_index'}.issubset(df.columns):
                formula = 'us_auto_production ~ import_penetration + us_gdp_billions + fuel_price_index'
            else:
                formula = 'us_auto_production ~ import_penetration + year'
            
            model = smf.ols(formula, data=df).fit()
            
            self.models['industry_transmission'] = model
            
            results = {
                'rsquared': float(model.rsquared),
                'nobs': int(model.nobs),
                'params': {k: float(v) for k, v in model.params.items()},
            }
            
            logger.info(f"Industry model R²: {results['rsquared']:.3f}")
            logger.info(f"Coefficient on import_penetration: {results['params'].get('import_penetration', float('nan')):.3f}")
        except Exception as e:
            logger.error(f"Error estimating industry model: {e}")
            results = {}
        
        # Save results
        output_file = RESULTS_DIR / 'q2_industry_transmission.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def simulate_japan_response_scenarios(
        self,
        elasticities: Optional[Dict] = None,
        industry_model: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate different Japanese response scenarios.
        
        Scenarios:
        - S0: Only tariff increase, no Japanese adjustment
        - S1: Partial relocation to US/Mexico
        - S2: Aggressive local production
        
        Args:
            elasticities: Import structure elasticities
            industry_model: Industry transmission model
            
        Returns:
            Tuple of (import_scenarios, industry_scenarios)
        """
        logger.info("Simulating Japanese response scenarios")
        
        # Define scenarios
        scenarios = {
            'S0_no_response': {
                'japan_direct_import_share': 0.30,  # 30% from Japan
                'us_produced_share': 0.20,
                'mexico_produced_share': 0.50,
                'tariff_on_japan': 0.25,  # 25% tariff
                'tariff_on_mexico': 0.00,
            },
            'S1_partial_relocation': {
                'japan_direct_import_share': 0.15,  # Reduced
                'us_produced_share': 0.35,  # Increased
                'mexico_produced_share': 0.50,
                'tariff_on_japan': 0.25,
                'tariff_on_mexico': 0.00,
            },
            'S2_aggressive_localization': {
                'japan_direct_import_share': 0.05,
                'us_produced_share': 0.50,
                'mexico_produced_share': 0.45,
                'tariff_on_japan': 0.25,
                'tariff_on_mexico': 0.00,
            },
        }
        
        # Baseline values (assumed)
        baseline_japan_sales = 2000000  # 2 million units
        baseline_us_production = 10000000  # 10 million units
        baseline_employment = 950000  # 950k workers
        
        import_results = []
        industry_results = []
        
        for scenario_name, params in scenarios.items():
            # Compute effective imports (tariff-exposed)
            effective_imports = (
                baseline_japan_sales * params['japan_direct_import_share']
            )
            
            # Compute total imports (including Mexico, which avoids tariff via USMCA)
            total_imports = effective_imports + (
                baseline_japan_sales * params['mexico_produced_share']
            )
            
            # US production by Japanese brands
            us_japanese_production = (
                baseline_japan_sales * params['us_produced_share']
            )
            
            # Import penetration
            total_supply = baseline_us_production + total_imports
            import_penetration = total_imports / total_supply
            
            # Impact on US industry (simplified)
            # Assume: 1% increase in import penetration reduces US production by 0.5%
            production_impact = -0.5 * (import_penetration - 0.25) * baseline_us_production
            
            new_us_production = baseline_us_production + production_impact
            new_employment = baseline_employment * (new_us_production / baseline_us_production)
            
            import_results.append({
                'scenario': scenario_name,
                'japan_direct_imports': effective_imports,
                'mexico_production': baseline_japan_sales * params['mexico_produced_share'],
                'us_production_japanese': us_japanese_production,
                'total_japanese_sales': baseline_japan_sales,
                'import_penetration': import_penetration * 100,
            })
            
            industry_results.append({
                'scenario': scenario_name,
                'us_auto_production': new_us_production,
                'us_employment': new_employment,
                'production_change_pct': (new_us_production / baseline_us_production - 1) * 100,
                'employment_change_pct': (new_employment / baseline_employment - 1) * 100,
            })
        
        import_df = pd.DataFrame(import_results)
        industry_df = pd.DataFrame(industry_results)
        
        # Save results
        import_df.to_csv(RESULTS_DIR / 'q2_scenario_imports.csv', index=False)
        industry_df.to_csv(RESULTS_DIR / 'q2_scenario_industry.csv', index=False)
        
        logger.info("Saved scenario results")
        
        return import_df, industry_df
    
    def plot_q2_results(self) -> None:
        """Generate plots for Q2 results."""
        import matplotlib.pyplot as plt
        from utils.config import apply_plot_style
        
        apply_plot_style()
        
        # Load scenario results
        try:
            import_df = pd.read_csv(RESULTS_DIR / 'q2_scenario_imports.csv')
            industry_df = pd.read_csv(RESULTS_DIR / 'q2_scenario_industry.csv')
        except FileNotFoundError:
            logger.error("Scenario results not found")
            return
        
        logger.info("Creating Q2 plots")
        
        # Plot 1: Import structure across scenarios
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(import_df))
        width = 0.25
        
        ax.bar(x - width, import_df['japan_direct_imports'], width, label='Japan Direct')
        ax.bar(x, import_df['mexico_production'], width, label='Mexico Production')
        ax.bar(x + width, import_df['us_production_japanese'], width, label='US Production')
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Volume (units)')
        ax.set_title('Japanese Auto Sales Composition by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(import_df['scenario'], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'q2_import_structure.pdf')
        plt.close()
        
        # Plot 2: US industry impact
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.bar(industry_df['scenario'], industry_df['us_auto_production'])
        ax1.set_ylabel('Production (units)')
        ax1.set_title('US Auto Production by Scenario')
        ax1.tick_params(axis='x', rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(industry_df['scenario'], industry_df['us_employment'])
        ax2.set_ylabel('Employment')
        ax2.set_title('US Auto Employment by Scenario')
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'q2_industry_impact.pdf')
        plt.close()
        
        logger.info("Q2 plots saved")


def run_q2_analysis() -> None:
    """Run complete Q2 analysis pipeline."""
    logger.info("="*60)
    logger.info("Starting Q2 Auto Trade Analysis")
    logger.info("="*60)
    
    model = AutoTradeModel()
    
    # Step 1: Load data
    model.load_q2_data()
    model.load_external_auto_data()
    
    # Step 2: Estimate models
    model.estimate_import_structure_model()
    model.estimate_industry_transmission_model()
    
    # Step 3: Simulate scenarios
    model.simulate_japan_response_scenarios()
    
    # Step 4: Plot results
    model.plot_q2_results()
    
    logger.info("Q2 analysis complete")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q2_analysis()
