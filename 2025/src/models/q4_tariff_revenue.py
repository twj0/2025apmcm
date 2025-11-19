"""
Q4: Tariff Revenue - Short and Medium-Term Analysis

Dynamic "Laffer curve" analysis with quadratic tariff-revenue relationship
and lagged import response to predict revenue over Trump's second term.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import RESULTS_DIR, FIGURES_DIR, DATA_EXTERNAL
from utils.data_loader import TariffDataLoader

logger = logging.getLogger(__name__)


class TariffRevenueModel:
    """Model for tariff revenue analysis with Laffer curve dynamics."""
    
    def __init__(self):
        """Initialize the model."""
        self.loader = TariffDataLoader()
        self.panel_data: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        self.elasticities: Dict = {}
        
    def load_q4_data(self) -> pd.DataFrame:
        """Load and aggregate data for revenue analysis.
        
        Returns:
            Panel DataFrame with revenue, imports, and tariffs
        """
        logger.info("Loading Q4 tariff revenue data")
        
        # Load imports with duty collected
        imports = self.loader.load_imports()
        
        # Aggregate to year level (can also do by sector/partner)
        revenue_panel = imports.groupby('year').agg({
            'duty_collected': 'sum',
        }).reset_index()
        
        revenue_panel.rename(columns={
            'duty_collected': 'total_revenue'
        }, inplace=True)
        
        # Note: effective tariff requires import value, which we need to add
        # For now, work with what we have
        
        logger.info(f"Loaded {len(revenue_panel)} years of revenue data")
        logger.info(f"Years: {revenue_panel['year'].tolist()}")
        logger.info(f"Total revenue range: {revenue_panel['total_revenue'].min():.0f} - {revenue_panel['total_revenue'].max():.0f}")
        
        self.panel_data = revenue_panel
        
        return revenue_panel
    
    def estimate_static_revenue_model(
        self,
        panel_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Estimate static Laffer-style revenue model.
        
        Model: ln(R) = α + β1·τ + β2·τ² + β3·X + ε
        
        Args:
            panel_df: Revenue panel data
            
        Returns:
            Model coefficients and revenue-maximizing tariff
        """
        if panel_df is None:
            panel_df = self.panel_data
        
        if panel_df is None or len(panel_df) < 5:
            logger.error("Insufficient data for estimation")
            return {}
        
        logger.info("Estimating static Laffer model")
        
        panel_df = panel_df.copy()
        
        # Load average tariff by year from external configuration
        avg_tariff_file = DATA_EXTERNAL / 'q4_avg_tariff_by_year.csv'
        if not avg_tariff_file.exists():
            logger.warning(f"Average tariff config not found: {avg_tariff_file}")
            template = panel_df[['year']].drop_duplicates().sort_values('year')
            template['avg_tariff'] = np.nan
            template.to_csv(avg_tariff_file, index=False)
            logger.warning(
                "Created template q4_avg_tariff_by_year.csv. "
                "Please fill 'avg_tariff' for each year before re-running."
            )
            # Cannot estimate model without tariff data
            return {}
        
        try:
            avg_df = pd.read_csv(avg_tariff_file)
        except Exception as e:
            logger.error(f"Error reading average tariff config: {e}")
            return {}
        
        if 'year' not in avg_df.columns or 'avg_tariff' not in avg_df.columns:
            logger.error("q4_avg_tariff_by_year.csv must contain 'year' and 'avg_tariff' columns")
            return {}
        
        panel_df = panel_df.merge(avg_df[['year', 'avg_tariff']], on='year', how='left')
        panel_df = panel_df.dropna(subset=['avg_tariff'])
        
        if len(panel_df) < 5:
            logger.error("Insufficient observations with avg_tariff for estimation")
            return {}
        
        panel_df['avg_tariff_sq'] = panel_df['avg_tariff'] ** 2
        panel_df['ln_revenue'] = np.log(panel_df['total_revenue'] + 1)
        
        try:
            # Estimate model
            formula = 'ln_revenue ~ avg_tariff + avg_tariff_sq'
            model = smf.ols(formula, data=panel_df).fit()
            
            self.models['static_laffer'] = model
            
            # Find revenue-maximizing tariff: -β1 / (2*β2)
            beta1 = model.params.get('avg_tariff', 0)
            beta2 = model.params.get('avg_tariff_sq', -1)
            
            if beta2 < 0:
                optimal_tariff = -beta1 / (2 * beta2)
            else:
                optimal_tariff = np.nan
            
            results = {
                'rsquared': float(model.rsquared),
                'nobs': int(model.nobs),
                'beta1_tariff': float(beta1),
                'beta2_tariff_sq': float(beta2),
                'optimal_tariff_pct': float(optimal_tariff * 100) if not np.isnan(optimal_tariff) else None,
            }
            
            logger.info(f"Static model R²: {results['rsquared']:.3f}")
            if results['optimal_tariff_pct']:
                logger.info(f"Revenue-maximizing tariff: {results['optimal_tariff_pct']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error estimating static model: {e}")
            results = {}
        
        # Save results
        with open(RESULTS_DIR / 'q4_static_laffer.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def estimate_dynamic_import_response(
        self,
        panel_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Estimate dynamic import response to tariff changes.
        
        Model: Δln(M) = φ0 + φ1·Δτ_t + φ2·Δτ_{t-1} + ... + u
        
        Args:
            panel_df: Panel data with imports and tariffs
            
        Returns:
            Dynamic elasticity parameters
        """
        logger.info("Estimating dynamic import response")
        
        # Read dynamic elasticity parameters from external JSON
        params_file = DATA_EXTERNAL / 'q4_dynamic_import_params.json'
        if not params_file.exists():
            logger.warning(f"Dynamic import parameter file not found: {params_file}")
            template = {
                'short_run_elasticity': None,
                'medium_run_elasticity': None,
                'adjustment_speed': None,
            }
            with open(params_file, 'w') as f:
                json.dump(template, f, indent=2)
            logger.warning(
                "Created template q4_dynamic_import_params.json. "
                "Please fill numeric values before relying on dynamic effects."
            )
            return {}
        
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
        except Exception as e:
            logger.error(f"Error reading dynamic import parameters: {e}")
            return {}
        
        # Basic validation
        required_keys = ['short_run_elasticity', 'medium_run_elasticity', 'adjustment_speed']
        for k in required_keys:
            if k not in params or params[k] is None:
                logger.error(f"Dynamic import parameter '{k}' is missing or None")
                return {}
        
        with open(RESULTS_DIR / 'q4_dynamic_import.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        return params
    
    def simulate_second_term_revenue(
        self,
        static_model: Optional[Dict] = None,
        dynamic_model: Optional[Dict] = None,
        tariff_scenarios: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Simulate revenue over second term (4-5 years).
        
        Args:
            static_model: Static Laffer model results
            dynamic_model: Dynamic import response parameters
            tariff_scenarios: Dict defining baseline vs policy tariff paths
            
        Returns:
            DataFrame with projected revenue paths
        """
        logger.info("Simulating second-term revenue")
        
        # Load tariff scenarios and baseline import value from external JSON
        scenarios_file = DATA_EXTERNAL / 'q4_tariff_scenarios.json'
        if tariff_scenarios is None:
            if not scenarios_file.exists():
                logger.warning(f"Tariff scenarios file not found: {scenarios_file}")
                template = {
                    'base_import_value': None,
                    'years': [2025, 2026, 2027, 2028, 2029],
                    'scenarios': {
                        'baseline': [None, None, None, None, None],
                        'reciprocal_tariff': [None, None, None, None, None],
                    },
                }
                with open(scenarios_file, 'w') as f:
                    json.dump(template, f, indent=2)
                logger.warning(
                    "Created template q4_tariff_scenarios.json. "
                    "Please fill base_import_value and tariff paths before simulation."
                )
                return pd.DataFrame()
            
            try:
                with open(scenarios_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Error reading tariff scenarios: {e}")
                return pd.DataFrame()
            
            base_import_value = config.get('base_import_value')
            years = config.get('years')
            tariff_scenarios = config.get('scenarios', {})
        else:
            # Use provided scenarios and default years if necessary
            years = [2025 + i for i in range(len(next(iter(tariff_scenarios.values()))))]
            base_import_value = None
        
        if base_import_value is None:
            logger.error("base_import_value must be specified in q4_tariff_scenarios.json")
            return pd.DataFrame()
        
        if dynamic_model is None:
            dynamic_model = self.estimate_dynamic_import_response()
        
        use_dynamic = bool(dynamic_model)
        
        results = []
        
        for scenario_name, tariff_path in tariff_scenarios.items():
            if len(tariff_path) != len(years):
                logger.error(f"Tariff path length mismatch for scenario {scenario_name}")
                continue
            
            import_value = base_import_value
            for year, tariff_rate in zip(years, tariff_path):
                if tariff_rate is None:
                    logger.error(f"Missing tariff rate for year {year} in scenario {scenario_name}")
                    continue
                
                if use_dynamic and results:
                    prev_tariff = results[-1]['tariff_rate']
                    tariff_change = tariff_rate - prev_tariff
                    elasticity = dynamic_model['short_run_elasticity']
                    import_change_pct = elasticity * tariff_change
                    import_value = results[-1]['import_value'] * (1 + import_change_pct)
                
                revenue = import_value * tariff_rate
                
                results.append({
                    'scenario': scenario_name,
                    'year': year,
                    'tariff_rate': tariff_rate,
                    'import_value': import_value,
                    'revenue': revenue,
                })
        
        if not results:
            logger.error("No valid tariff scenarios to simulate")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # Compute cumulative revenue difference for baseline vs reciprocal_tariff if both present
        scenario_names = results_df['scenario'].unique().tolist()
        summary = {}
        if 'baseline' in scenario_names and 'reciprocal_tariff' in scenario_names:
            baseline_revenue = results_df[results_df['scenario'] == 'baseline']['revenue'].sum()
            policy_revenue = results_df[results_df['scenario'] == 'reciprocal_tariff']['revenue'].sum()
            net_revenue_gain = policy_revenue - baseline_revenue
            
            logger.info(f"Baseline cumulative revenue: ${baseline_revenue/1e9:.1f}B")
            logger.info(f"Policy cumulative revenue: ${policy_revenue/1e9:.1f}B")
            logger.info(f"Net revenue gain: ${net_revenue_gain/1e9:.1f}B")
            
            summary = {
                'baseline_total': float(baseline_revenue),
                'policy_total': float(policy_revenue),
                'net_gain': float(net_revenue_gain),
            }
        
        # Save results
        results_df.to_csv(RESULTS_DIR / 'q4_revenue_scenarios.csv', index=False)
        
        if summary:
            with open(RESULTS_DIR / 'q4_revenue_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
        
        return results_df
    
    def plot_q4_results(self) -> None:
        """Generate plots for Q4 results."""
        import matplotlib.pyplot as plt
        from utils.config import apply_plot_style
        
        apply_plot_style()
        
        try:
            revenue_df = pd.read_csv(RESULTS_DIR / 'q4_revenue_scenarios.csv')
        except FileNotFoundError:
            logger.error("Revenue scenarios not found")
            return
        
        logger.info("Creating Q4 plots")
        
        # Plot 1: Revenue over time by scenario
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for scenario in revenue_df['scenario'].unique():
            data = revenue_df[revenue_df['scenario'] == scenario]
            ax.plot(data['year'], data['revenue'] / 1e9, marker='o', label=scenario)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Tariff Revenue ($ Billions)')
        ax.set_title('Projected Tariff Revenue: Baseline vs Reciprocal Tariff Policy')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'q4_revenue_time_path.pdf')
        plt.close()
        
        # Plot 2: Laffer curve (stylized)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tariff_rates = np.linspace(0, 0.50, 100)
        # Stylized Laffer curve: R = t * (1 - t/t_max) * base
        t_max = 0.40
        base_revenue = 100
        revenues = tariff_rates * (1 - tariff_rates / t_max) * base_revenue
        
        ax.plot(tariff_rates * 100, revenues, linewidth=2)
        ax.axvline(x=20, color='red', linestyle='--', label='Revenue-maximizing rate (~20%)')
        ax.axvline(x=12, color='blue', linestyle='--', label='Proposed policy (~12%)')
        
        ax.set_xlabel('Tariff Rate (%)')
        ax.set_ylabel('Revenue (Stylized Index)')
        ax.set_title('Laffer Curve: Tariff Rate vs Revenue')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'q4_laffer_curve.pdf')
        plt.close()
        
        logger.info("Q4 plots saved")


def run_q4_analysis() -> None:
    """Run complete Q4 analysis pipeline."""
    logger.info("="*60)
    logger.info("Starting Q4 Tariff Revenue Analysis")
    logger.info("="*60)
    
    model = TariffRevenueModel()
    
    # Step 1: Load data
    model.load_q4_data()
    
    # Step 2: Estimate static model
    static_results = model.estimate_static_revenue_model()
    
    # Step 3: Estimate dynamic model
    dynamic_results = model.estimate_dynamic_import_response()
    
    # Step 4: Simulate revenue scenarios
    model.simulate_second_term_revenue(static_results, dynamic_results)
    
    # Step 5: Plot results
    model.plot_q4_results()
    
    logger.info("Q4 analysis complete")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q4_analysis()
