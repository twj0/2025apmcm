"""
Q1: Soybean Trade among China, U.S., Brazil, and Argentina

This module implements the panel trade and source-substitution model
to analyze how U.S. tariff adjustments and Chinese countermeasures
affect soybean exports from the three major suppliers to China.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import RESULTS_DIR, FIGURES_DIR, DATA_EXTERNAL, DATA_PROCESSED
from utils.data_loader import TariffDataLoader
from utils.mapping import HSMapper

logger = logging.getLogger(__name__)


class SoybeanTradeModel:
    """Model for analyzing soybean trade redistribution among exporters."""
    
    def __init__(self):
        """Initialize the model."""
        self.loader = TariffDataLoader()
        self.mapper = HSMapper()
        self.panel_data: Optional[pd.DataFrame] = None
        self.elasticities: Dict[str, float] = {}
        self.model_results: Dict[str, RegressionResultsWrapper] = {}
        
    def load_q1_data(self) -> pd.DataFrame:
        """Load and prepare data for Q1 analysis.
        
        Returns:
            Panel DataFrame with soybean trade data
        """
        logger.info("Loading Q1 soybean data")
        
        # Load exports data
        exports = self.loader.load_exports()
        
        # Filter for soybeans
        exports_tagged = self.mapper.tag_dataframe(exports)
        soybeans = exports_tagged[exports_tagged['is_soybean']].copy()
        
        # Focus on China as destination
        soybeans_china = soybeans[
            soybeans['partner_country'].str.contains('China', case=False, na=False)
        ].copy()
        
        logger.info(f"Loaded {len(soybeans_china)} soybean export records to China")
        
        # We need to supplement this with Chinese import data from external sources
        # For now, we'll work with what we have
        
        # Add derived variables
        if 'export_value' in soybeans_china.columns:
            soybeans_china['export_value_millions'] = soybeans_china['export_value'] / 1e6
        
        # Create exporter identifier (US is implicit in US export data)
        soybeans_china['exporter'] = 'US'
        soybeans_china['importer'] = 'China'
        
        self.panel_data = soybeans_china
        
        return soybeans_china
    
    def load_external_china_imports(self) -> pd.DataFrame:
        """Load Chinese import data for soybeans from external file.
        
        This should include imports from US, Brazil, and Argentina.
        
        Returns:
            DataFrame with columns: year, exporter, import_value, import_quantity, etc.
        """
        processed_file = DATA_PROCESSED / 'q1' / 'q1_0.csv'
        
        if processed_file.exists():
            df = pd.read_csv(processed_file)
            logger.info(f"Loaded {len(df)} records from standard Q1 processed data: {processed_file}")
        else:
            external_file = DATA_EXTERNAL / 'china_imports_soybeans.csv'
            
            if not external_file.exists():
                logger.warning(f"External China imports file not found: {external_file}")
                logger.info("Creating template file for manual data entry")
                
                # Create a template
                template = pd.DataFrame({
                    'year': [2020, 2020, 2020, 2021, 2021, 2021],
                    'exporter': ['US', 'Brazil', 'Argentina'] * 2,
                    'import_value_usd': [0, 0, 0, 0, 0, 0],
                    'import_quantity_tonnes': [0, 0, 0, 0, 0, 0],
                    'tariff_cn_on_exporter': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                })
                template.to_csv(external_file, index=False)
                logger.info(f"Template saved to {external_file} - please fill with actual data")
                
                df = template
            else:
                df = pd.read_csv(external_file)
                logger.info(f"Loaded {len(df)} records from Chinese imports data")
        
        if 'unit_value' not in df.columns:
            if 'unit_price_usd_per_ton' in df.columns:
                df['unit_value'] = df['unit_price_usd_per_ton']
            else:
                df['unit_value'] = df['import_value_usd'] / df['import_quantity_tonnes']
        
        df['price_with_tariff'] = df['unit_value'] * (1 + df['tariff_cn_on_exporter'])
        
        return df
    
    def prepare_panel_for_estimation(self) -> pd.DataFrame:
        """Prepare panel data for econometric estimation.
        
        Returns:
            Clean panel ready for regression
        """
        df = self.load_external_china_imports()
        
        # Add log transforms
        df['ln_import_value'] = np.log(df['import_value_usd'] + 1)
        df['ln_import_quantity'] = np.log(df['import_quantity_tonnes'] + 1)
        df['ln_price_with_tariff'] = np.log(df['price_with_tariff'] + 1e-6)
        df['ln_unit_value'] = np.log(df['unit_value'] + 1e-6)
        
        # Compute market shares
        total_by_year = df.groupby('year')['import_value_usd'].transform('sum')
        df['market_share'] = df['import_value_usd'] / total_by_year
        
        # Create relative shares (vs US as baseline)
        us_share = df[df['exporter'] == 'US'].set_index('year')['market_share']
        df['us_share_ref'] = df['year'].map(us_share)
        df['ln_share_ratio'] = np.log((df['market_share'] + 1e-6) / (df['us_share_ref'] + 1e-6))
        
        # Create relative tariff
        us_tariff = df[df['exporter'] == 'US'].set_index('year')['tariff_cn_on_exporter']
        df['us_tariff_ref'] = df['year'].map(us_tariff)
        df['tariff_diff_vs_us'] = df['tariff_cn_on_exporter'] - df['us_tariff_ref']
        
        logger.info(f"Prepared panel with shape {df.shape}")
        
        self.panel_data = df
        return df
    
    def estimate_trade_elasticities(self, panel_df: Optional[pd.DataFrame] = None) -> Dict:
        """Estimate trade elasticities using panel regression.
        
        Model 1: ln(import_value) = α + β1·ln(price_with_tariff) + controls
        
        Args:
            panel_df: Panel DataFrame (if None, uses self.panel_data)
            
        Returns:
            Dictionary with elasticity estimates and statistics
        """
        if panel_df is None:
            panel_df = self.panel_data
        
        if panel_df is None or len(panel_df) == 0:
            logger.error("No panel data available for estimation")
            return {}
        
        logger.info("Estimating trade elasticities")
        
        # Model 1: Import value response to price
        try:
            # Simple OLS with exporter fixed effects
            formula = 'ln_import_value ~ ln_price_with_tariff + C(exporter)'
            model1 = smf.ols(formula, data=panel_df).fit()
            
            self.model_results['trade_elasticity'] = model1
            
            # Extract price elasticity
            price_elasticity = model1.params.get('ln_price_with_tariff', np.nan)
            price_se = model1.bse.get('ln_price_with_tariff', np.nan)
            
            logger.info(f"Price elasticity: {price_elasticity:.3f} (SE: {price_se:.3f})")
            logger.info(f"R-squared: {model1.rsquared:.3f}")
            
            self.elasticities['price_elasticity'] = price_elasticity
            self.elasticities['price_se'] = price_se
            
        except Exception as e:
            logger.error(f"Error estimating trade elasticity: {e}")
            self.elasticities['price_elasticity'] = -1.0  # Default assumption
            self.elasticities['price_se'] = 0.3
        
        # Model 2: Share model (relative to US)
        try:
            # Filter out US (since it's the reference)
            non_us = panel_df[panel_df['exporter'] != 'US'].copy()
            
            if len(non_us) > 0:
                formula_share = 'ln_share_ratio ~ tariff_diff_vs_us + C(exporter)'
                model2 = smf.ols(formula_share, data=non_us).fit()
                
                self.model_results['share_elasticity'] = model2
                
                share_elasticity = model2.params.get('tariff_diff_vs_us', np.nan)
                share_se = model2.bse.get('tariff_diff_vs_us', np.nan)
                
                logger.info(f"Share elasticity (tariff diff): {share_elasticity:.3f} (SE: {share_se:.3f})")
                
                self.elasticities['share_elasticity'] = share_elasticity
                self.elasticities['share_se'] = share_se
                
        except Exception as e:
            logger.error(f"Error estimating share elasticity: {e}")
            self.elasticities['share_elasticity'] = -2.0
            self.elasticities['share_se'] = 0.5
        
        # Save results
        self._save_elasticities()
        
        return self.elasticities
    
    def _save_elasticities(self) -> None:
        """Save elasticity estimates to JSON."""
        output_file = RESULTS_DIR / 'q1_elasticities.json'
        
        # Prepare serializable dict
        results = {
            'elasticities': self.elasticities,
            'model_summaries': {}
        }
        
        for name, model in self.model_results.items():
            results['model_summaries'][name] = {
                'rsquared': float(model.rsquared),
                'rsquared_adj': float(model.rsquared_adj),
                'nobs': int(model.nobs),
                'params': {k: float(v) for k, v in model.params.items()},
                'pvalues': {k: float(v) for k, v in model.pvalues.items()},
            }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved elasticities to {output_file}")
    
    def simulate_tariff_scenarios(
        self,
        panel_df: Optional[pd.DataFrame] = None,
        elasticities: Optional[Dict] = None,
        scenarios: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Simulate trade under different tariff scenarios.
        
        Args:
            panel_df: Panel data
            elasticities: Elasticity estimates (if None, uses self.elasticities)
            scenarios: Dict of scenario definitions
            
        Returns:
            DataFrame with scenario results
        """
        if panel_df is None:
            panel_df = self.panel_data
        
        if elasticities is None:
            elasticities = self.elasticities
        
        if scenarios is None:
            # Default scenarios
            scenarios = {
                'baseline': {
                    'US': 0.0,
                    'Brazil': 0.0,
                    'Argentina': 0.0,
                },
                'reciprocal_tariff': {
                    'US': 0.25,  # 25% additional tariff from China on US soybeans
                    'Brazil': 0.0,
                    'Argentina': 0.0,
                },
                'full_retaliation': {
                    'US': 0.50,  # 50% tariff
                    'Brazil': 0.0,
                    'Argentina': 0.0,
                },
            }
        
        logger.info("Simulating tariff scenarios")
        
        # Get latest year data as baseline
        latest_year = panel_df['year'].max()
        baseline_data = panel_df[panel_df['year'] == latest_year].copy()
        
        price_elast = elasticities.get('price_elasticity', -1.0)
        
        results = []
        
        for scenario_name, tariff_changes in scenarios.items():
            logger.info(f"Simulating scenario: {scenario_name}")
            
            for _, row in baseline_data.iterrows():
                exporter = row['exporter']
                
                # Apply tariff change
                new_tariff = row['tariff_cn_on_exporter'] + tariff_changes.get(exporter, 0.0)
                
                # Compute new price
                old_price_with_tariff = row['price_with_tariff']
                new_price_with_tariff = row['unit_value'] * (1 + new_tariff)
                
                # Apply elasticity to compute new import value
                price_change_pct = (new_price_with_tariff / old_price_with_tariff) - 1
                import_change_pct = price_elast * price_change_pct
                
                new_import_value = row['import_value_usd'] * (1 + import_change_pct)
                
                results.append({
                    'scenario': scenario_name,
                    'exporter': exporter,
                    'baseline_import_value': row['import_value_usd'],
                    'baseline_tariff': row['tariff_cn_on_exporter'],
                    'new_tariff': new_tariff,
                    'tariff_change': tariff_changes.get(exporter, 0.0),
                    'simulated_import_value': new_import_value,
                    'import_change_pct': import_change_pct * 100,
                })
        
        results_df = pd.DataFrame(results)
        
        # Compute shares within each scenario
        for scenario in results_df['scenario'].unique():
            mask = results_df['scenario'] == scenario
            total = results_df.loc[mask, 'simulated_import_value'].sum()
            results_df.loc[mask, 'market_share'] = (
                results_df.loc[mask, 'simulated_import_value'] / total * 100
            )
        
        # Save results
        output_file = RESULTS_DIR / 'q1_scenario_exports.csv'
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved scenario results to {output_file}")
        
        return results_df
    
    def plot_q1_results(
        self,
        scenario_results: Optional[pd.DataFrame] = None
    ) -> None:
        """Generate figures for Q1 results.
        
        Args:
            scenario_results: DataFrame with scenario simulation results
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from utils.config import apply_plot_style
        
        apply_plot_style()
        
        if scenario_results is None:
            # Load from file
            results_file = RESULTS_DIR / 'q1_scenario_exports.csv'
            if not results_file.exists():
                logger.error("No scenario results found to plot")
                return
            scenario_results = pd.read_csv(results_file)
        
        logger.info("Creating Q1 plots")
        
        # Plot 1: Market shares before and after
        fig, ax = plt.subplots(figsize=(12, 6))
        
        scenarios = scenario_results['scenario'].unique()
        exporters = scenario_results['exporter'].unique()
        
        x = np.arange(len(exporters))
        width = 0.25
        
        for i, scenario in enumerate(scenarios):
            data = scenario_results[scenario_results['scenario'] == scenario]
            shares = [
                data[data['exporter'] == exp]['market_share'].values[0]
                for exp in exporters
            ]
            ax.bar(x + i * width, shares, width, label=scenario)
        
        ax.set_xlabel('Exporter')
        ax.set_ylabel('Market Share (%)')
        ax.set_title('Soybean Export Market Shares to China by Scenario')
        ax.set_xticks(x + width)
        ax.set_xticklabels(exporters)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        output_file = FIGURES_DIR / 'q1_shares_before_after.pdf'
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved figure to {output_file}")
        
        # Plot 2: Import value changes
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot = scenario_results.pivot(
            index='exporter',
            columns='scenario',
            values='simulated_import_value'
        )
        pivot.plot(kind='bar', ax=ax)
        
        ax.set_xlabel('Exporter')
        ax.set_ylabel('Import Value (USD)')
        ax.set_title('Simulated Soybean Import Values under Different Scenarios')
        ax.legend(title='Scenario')
        ax.grid(axis='y', alpha=0.3)
        
        output_file = FIGURES_DIR / 'q1_import_values_scenarios.pdf'
        fig.savefig(output_file, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved figure to {output_file}")


def run_q1_analysis() -> None:
    """Run complete Q1 analysis pipeline."""
    logger.info("="*60)
    logger.info("Starting Q1 Soybean Trade Analysis")
    logger.info("="*60)
    
    model = SoybeanTradeModel()
    
    # Step 1: Load data
    model.prepare_panel_for_estimation()
    
    # Step 2: Estimate elasticities
    model.estimate_trade_elasticities()
    
    # Step 3: Simulate scenarios
    results = model.simulate_tariff_scenarios()
    
    # Step 4: Plot results
    model.plot_q1_results(results)
    
    logger.info("Q1 analysis complete")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q1_analysis()
