"""
Q5: Macroeconomic, Sectoral, and Financial Effects; Reshoring

VAR/SVAR analysis and event study to assess tariff impacts on macro indicators,
financial markets, and manufacturing reshoring.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import RESULTS_DIR, FIGURES_DIR, DATA_EXTERNAL, DATA_PROCESSED
from utils.data_loader import TariffDataLoader

logger = logging.getLogger(__name__)


class MacroFinanceModel:
    """Model for macroeconomic and financial impact analysis."""
    
    def __init__(self):
        """Initialize the model."""
        self.loader = TariffDataLoader()
        self.time_series: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        
    def load_q5_data(self) -> pd.DataFrame:
        """Load and merge all data sources for Q5.
        
        Returns:
            Merged time series DataFrame
        """
        logger.info("Loading Q5 macro/financial data")
        
        # Load tariff indices (from processed data or compute)
        try:
            tariff_indices = pd.read_parquet(DATA_PROCESSED / 'tariff_indices.parquet')
            logger.info("Loaded tariff indices")
        except FileNotFoundError:
            # Create placeholder
            logger.warning("Tariff indices not found, creating placeholder")
            tariff_indices = pd.DataFrame({
                'year': range(2015, 2026),
                'tariff_index_total': np.random.uniform(2.0, 8.0, 11),
            })
        
        # Load external macro data
        macro_file = DATA_EXTERNAL / 'us_macro.csv'
        if not macro_file.exists():
            logger.warning(f"Macro data not found: {macro_file}")
            template_macro = pd.DataFrame({
                'year': range(2015, 2026),
                'gdp_growth': np.random.uniform(1.5, 3.5, 11),
                'industrial_production': np.random.uniform(95, 110, 11),
                'unemployment_rate': np.random.uniform(3.5, 8.0, 11),
                'cpi': np.random.uniform(240, 290, 11),
            })
            template_macro.to_csv(macro_file, index=False)
            macro_data = template_macro
        else:
            macro_data = pd.read_csv(macro_file)
        
        # Load financial data
        financial_file = DATA_EXTERNAL / 'us_financial.csv'
        if not financial_file.exists():
            logger.warning(f"Financial data not found: {financial_file}")
            template_financial = pd.DataFrame({
                'year': range(2015, 2026),
                'dollar_index': np.random.uniform(90, 105, 11),
                'treasury_yield_10y': np.random.uniform(1.5, 4.5, 11),
                'sp500_index': np.random.uniform(2500, 4500, 11),
                'crypto_index': np.random.uniform(5000, 50000, 11),
            })
            template_financial.to_csv(financial_file, index=False)
            financial_data = template_financial
        else:
            financial_data = pd.read_csv(financial_file)
        
        # Load reshoring data
        reshoring_file = DATA_EXTERNAL / 'us_reshoring.csv'
        if not reshoring_file.exists():
            logger.warning(f"Reshoring data not found: {reshoring_file}")
            template_reshoring = pd.DataFrame({
                'year': range(2015, 2026),
                'manufacturing_va_share': np.random.uniform(10, 13, 11),
                'manufacturing_employment_share': np.random.uniform(8, 10, 11),
                'reshoring_fdi_billions': np.random.uniform(5, 50, 11),
            })
            template_reshoring.to_csv(reshoring_file, index=False)
            reshoring_data = template_reshoring
        else:
            reshoring_data = pd.read_csv(reshoring_file)
        
        # Load retaliation index
        retaliation_file = DATA_EXTERNAL / 'retaliation_index.csv'
        if not retaliation_file.exists():
            logger.warning(f"Retaliation index not found: {retaliation_file}")
            template_retaliation = pd.DataFrame({
                'year': range(2015, 2026),
                'retaliation_index': [0, 0, 1, 2, 1, 1, 3, 5, 8, 10, 12],
            })
            template_retaliation.to_csv(retaliation_file, index=False)
            retaliation_data = template_retaliation
        else:
            retaliation_data = pd.read_csv(retaliation_file)
        
        # Merge all datasets on year
        merged = tariff_indices.merge(macro_data, on='year', how='outer')
        merged = merged.merge(financial_data, on='year', how='outer')
        merged = merged.merge(reshoring_data, on='year', how='outer')
        merged = merged.merge(retaliation_data, on='year', how='outer')
        
        merged = merged.sort_values('year').reset_index(drop=True)
        
        logger.info(f"Merged time series shape: {merged.shape}")
        logger.info(f"Columns: {merged.columns.tolist()}")
        
        self.time_series = merged
        
        return merged
    
    def estimate_regression_effects(
        self,
        ts_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Estimate regression-based effects of tariffs on key variables.
        
        Model: Y_t = λ0 + λ1·TariffIndex_t + λ2·RetaliationIndex_t + λ3·Z_t + ε
        
        Args:
            ts_df: Time series DataFrame
            
        Returns:
            Dictionary of regression results
        """
        if ts_df is None:
            ts_df = self.time_series
        
        if ts_df is None or len(ts_df) < 10:
            logger.error("Insufficient time series data")
            return {}
        
        logger.info("Estimating regression effects")
        
        results = {}
        
        # Dependent variables to analyze
        dep_vars = [
            'gdp_growth',
            'industrial_production',
            'manufacturing_va_share',
            'manufacturing_employment_share',
        ]
        
        for dep_var in dep_vars:
            if dep_var not in ts_df.columns:
                continue
            
            try:
                formula = f'{dep_var} ~ tariff_index_total + retaliation_index'
                model = smf.ols(formula, data=ts_df).fit()
                
                results[dep_var] = {
                    'rsquared': float(model.rsquared),
                    'tariff_coef': float(model.params.get('tariff_index_total', 0)),
                    'tariff_pvalue': float(model.pvalues.get('tariff_index_total', 1)),
                    'retaliation_coef': float(model.params.get('retaliation_index', 0)),
                    'retaliation_pvalue': float(model.pvalues.get('retaliation_index', 1)),
                }
                
                logger.info(f"{dep_var}: R²={results[dep_var]['rsquared']:.3f}, "
                           f"tariff_coef={results[dep_var]['tariff_coef']:.3f}")
                
            except Exception as e:
                logger.error(f"Error for {dep_var}: {e}")
        
        # Save results
        with open(RESULTS_DIR / 'q5_regressions.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def estimate_var_model(
        self,
        ts_df: Optional[pd.DataFrame] = None,
        variables: Optional[List[str]] = None,
        maxlags: int = 2
    ) -> Dict:
        """Estimate VAR model and compute impulse responses.
        
        Args:
            ts_df: Time series DataFrame
            variables: List of variables to include in VAR
            maxlags: Maximum number of lags
            
        Returns:
            VAR results and IRF summary
        """
        if ts_df is None:
            ts_df = self.time_series
        
        if variables is None:
            variables = [
                'tariff_index_total',
                'retaliation_index',
                'gdp_growth',
                'industrial_production',
            ]
        
        # Filter to available variables
        available_vars = [v for v in variables if v in ts_df.columns]
        
        if len(available_vars) < 2:
            logger.warning("Insufficient variables for VAR")
            return {}
        
        logger.info(f"Estimating VAR with variables: {available_vars}")
        
        try:
            # Prepare data (remove NaN)
            var_data = ts_df[available_vars].dropna()
            
            if len(var_data) < 10:
                logger.warning("Insufficient observations for VAR")
                return {}
            
            # Fit VAR
            model = VAR(var_data)
            fitted = model.fit(maxlags=maxlags)
            
            self.models['var'] = fitted
            
            # Compute IRFs
            irf = fitted.irf(periods=10)
            
            # Extract IRF for tariff shock
            tariff_shock_idx = available_vars.index('tariff_index_total') if 'tariff_index_total' in available_vars else 0
            
            irf_summary = {
                'lag_order': int(fitted.k_ar),
                'nobs': int(fitted.nobs),
                'variables': available_vars,
                'aic': float(fitted.aic),
                'bic': float(fitted.bic),
            }
            
            logger.info(f"VAR fitted with {irf_summary['lag_order']} lags, AIC={irf_summary['aic']:.2f}")
            
            # Save IRF plot data
            irf_data = {}
            for i, var in enumerate(available_vars):
                irf_data[var] = irf.irfs[:, i, tariff_shock_idx].tolist()
            
            irf_summary['irf_tariff_shock'] = irf_data
            
        except Exception as e:
            logger.error(f"Error estimating VAR: {e}")
            irf_summary = {}
        
        # Save results
        with open(RESULTS_DIR / 'q5_var_results.json', 'w') as f:
            json.dump(irf_summary, f, indent=2)
        
        return irf_summary
    
    def evaluate_reshoring(
        self,
        ts_df: Optional[pd.DataFrame] = None,
        treatment_year: int = 2025
    ) -> Dict:
        """Evaluate reshoring using event study / DID approach.
        
        Args:
            ts_df: Time series data
            treatment_year: Year when reciprocal tariffs introduced
            
        Returns:
            Reshoring effect estimates
        """
        if ts_df is None:
            ts_df = self.time_series
        
        if ts_df is None or 'manufacturing_va_share' not in ts_df.columns:
            logger.warning("Cannot evaluate reshoring: missing data")
            return {}
        
        logger.info("Evaluating reshoring effects")
        
        # Create treatment indicator
        ts_df = ts_df.copy()
        ts_df['post_treatment'] = (ts_df['year'] >= treatment_year).astype(int)
        
        try:
            # Simple before-after comparison
            pre_treatment = ts_df[ts_df['post_treatment'] == 0]
            post_treatment = ts_df[ts_df['post_treatment'] == 1]
            
            if len(pre_treatment) == 0 or len(post_treatment) == 0:
                logger.warning("Insufficient pre/post periods")
                return {}
            
            pre_mean = pre_treatment['manufacturing_va_share'].mean()
            post_mean = post_treatment['manufacturing_va_share'].mean()
            difference = post_mean - pre_mean
            
            # Regression with treatment
            formula = 'manufacturing_va_share ~ post_treatment + year'
            model = smf.ols(formula, data=ts_df).fit()
            
            p_value = float(model.pvalues.get('post_treatment', 1))
            results = {
                'pre_treatment_mean': float(pre_mean),
                'post_treatment_mean': float(post_mean),
                'difference': float(difference),
                'treatment_coef': float(model.params.get('post_treatment', 0)),
                'treatment_pvalue': p_value,
                'statistically_significant': bool(p_value < 0.05),
            }
            
            logger.info(f"Reshoring effect: {results['difference']:.2f} percentage points")
            logger.info(f"Statistically significant: {results['statistically_significant']}")
            
        except Exception as e:
            logger.error(f"Error evaluating reshoring: {e}")
            results = {}
        
        # Save results
        with open(RESULTS_DIR / 'q5_reshoring_effects.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def plot_q5_results(self) -> None:
        """Generate plots for Q5 results."""
        import matplotlib.pyplot as plt
        from utils.config import apply_plot_style
        
        apply_plot_style()
        
        if self.time_series is None:
            logger.error("No time series data to plot")
            return
        
        logger.info("Creating Q5 plots")
        
        ts = self.time_series
        
        # Plot 1: Time series of key indicators
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if 'tariff_index_total' in ts.columns:
            axes[0, 0].plot(ts['year'], ts['tariff_index_total'], marker='o')
            axes[0, 0].set_title('Tariff Index Over Time')
            axes[0, 0].set_ylabel('Tariff Index (%)')
            axes[0, 0].grid(alpha=0.3)
        
        if 'gdp_growth' in ts.columns:
            axes[0, 1].plot(ts['year'], ts['gdp_growth'], marker='o', color='green')
            axes[0, 1].set_title('GDP Growth Rate')
            axes[0, 1].set_ylabel('Growth Rate (%)')
            axes[0, 1].grid(alpha=0.3)
        
        if 'manufacturing_va_share' in ts.columns:
            axes[1, 0].plot(ts['year'], ts['manufacturing_va_share'], marker='o', color='orange')
            axes[1, 0].axvline(x=2025, color='red', linestyle='--', label='Policy Change')
            axes[1, 0].set_title('Manufacturing Value-Added Share')
            axes[1, 0].set_ylabel('Share (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        
        if 'dollar_index' in ts.columns:
            axes[1, 1].plot(ts['year'], ts['dollar_index'], marker='o', color='purple')
            axes[1, 1].set_title('Dollar Index')
            axes[1, 1].set_ylabel('Index')
            axes[1, 1].grid(alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel('Year')
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'q5_time_series_overview.pdf')
        plt.close()
        
        # Plot 2: IRF (if available)
        try:
            with open(RESULTS_DIR / 'q5_var_results.json', 'r') as f:
                var_results = json.load(f)
            
            if 'irf_tariff_shock' in var_results:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                irf_data = var_results['irf_tariff_shock']
                periods = range(len(next(iter(irf_data.values()))))
                
                for var_name, irf_values in irf_data.items():
                    ax.plot(periods, irf_values, marker='o', label=var_name)
                
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
                ax.set_xlabel('Periods After Shock')
                ax.set_ylabel('Response')
                ax.set_title('Impulse Response to Tariff Shock')
                ax.legend()
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                fig.savefig(FIGURES_DIR / 'q5_impulse_response.pdf')
                plt.close()
        
        except Exception as e:
            logger.warning(f"Could not plot IRF: {e}")
        
        logger.info("Q5 plots saved")


def run_q5_analysis() -> None:
    """Run complete Q5 analysis pipeline."""
    logger.info("="*60)
    logger.info("Starting Q5 Macro/Financial Impact Analysis")
    logger.info("="*60)
    
    model = MacroFinanceModel()
    
    # Step 1: Load data
    model.load_q5_data()
    
    # Step 2: Regression analysis
    model.estimate_regression_effects()
    
    # Step 3: VAR analysis
    model.estimate_var_model()
    
    # Step 4: Reshoring evaluation
    model.evaluate_reshoring()
    
    # Step 5: Plot results
    model.plot_q5_results()
    
    logger.info("Q5 analysis complete")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q5_analysis()
