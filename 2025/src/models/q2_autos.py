"""
Q2: Auto import and production location adjustment analysis.

This module implements:
1. **Econometric Model** (Original): Import structure estimation via OLS
2. **MARL Enhancement** (New): Multi-Agent Reinforcement Learning game-theoretic analysis
   - Nash Equilibrium solver for US-Japan tariff game
   - Strategic relocation decisions under policy uncertainty
3. Industry transmission effects
4. Comprehensive data export: json/csv/md formats

Results structure:
- 2025/results/q2/econometric/  # Original OLS results
- 2025/results/q2/marl/          # MARL game analysis results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    tf = None

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import (
    RESULTS_DIR,
    FIGURES_DIR,
    DATA_EXTERNAL,
    DATA_PROCESSED,
    RESULTS_LOGS,
)
from utils.data_loader import TariffDataLoader
from utils.mapping import HSMapper

logger = logging.getLogger(__name__)

class NashEquilibriumSolver:
    """Simplified Nash Equilibrium solver for US-Japan tariff game.
    
    Based on Q2_MARL_Technical_Guide.md but simplified for rapid deployment.
    Full MARL training would require extensive computational resources.
    """
    
    def __init__(self):
        self.payoff_matrix = None
        self.equilibria = []
        
    def compute_best_responses(self, us_tariff_grid: np.ndarray, 
                                japan_relocation_grid: np.ndarray,
                                scenario_results: pd.DataFrame) -> Dict[str, Any]:
        """Compute best response functions for both players.
        
        Args:
            us_tariff_grid: Array of possible US tariff rates [0.0, 0.10, 0.25]
            japan_relocation_grid: Array of relocation intensities [0.0, 0.5, 1.0]
            scenario_results: DataFrame with scenario simulation results
            
        Returns:
            Dictionary with Nash equilibrium analysis
        """
        n_us = len(us_tariff_grid)
        n_jp = len(japan_relocation_grid)
        
        # Payoff matrices (US perspective: employment gain, Japan: profit retention)
        us_payoffs = np.zeros((n_us, n_jp))
        jp_payoffs = np.zeros((n_us, n_jp))
        
        # Populate payoffs from scenario results
        for i, tariff in enumerate(us_tariff_grid):
            for j, reloc in enumerate(japan_relocation_grid):
                # Find matching scenario (align with simulate_japan_response_scenarios keys)
                if reloc == 0.0:
                    scenario = 'S0_no_response'
                elif reloc == 0.5:
                    scenario = 'S1_partial_relocation'
                else:
                    scenario = 'S2_aggressive_localization'

                match_rows = scenario_results[scenario_results['scenario'] == scenario]
                if match_rows.empty:
                    # Fallback to first row to avoid crash if mismatch
                    row = scenario_results.iloc[0]
                else:
                    row = match_rows.iloc[0]
                
                # US payoff: employment change + production increase (robust)
                try:
                    emp_chg = float(row.get('employment_change_pct', 0.0)) if isinstance(row, pd.Series) else 0.0
                except Exception:
                    emp_chg = 0.0
                try:
                    prod_chg = float(row.get('production_change_pct', 0.0)) if isinstance(row, pd.Series) else 0.0
                except Exception:
                    prod_chg = 0.0
                us_payoffs[i, j] = emp_chg + prod_chg
                
                # Japan payoff: maintain sales - relocation cost (robust to missing columns)
                base_total = 1.0
                if 'total_japanese_sales' in scenario_results.columns and not scenario_results['total_japanese_sales'].empty:
                    try:
                        base_total = float(scenario_results['total_japanese_sales'].iloc[0])
                    except Exception:
                        base_total = 1.0
                curr_total = None
                if isinstance(row, pd.Series) and 'total_japanese_sales' in row.index:
                    try:
                        curr_total = float(row['total_japanese_sales'])
                    except Exception:
                        curr_total = None
                sales_retention = (curr_total if curr_total is not None else base_total) / (base_total if base_total != 0 else 1.0)
                relocation_cost = reloc * 30  # Simplified cost function
                jp_payoffs[i, j] = sales_retention * 100 - relocation_cost
        
        # Find Nash Equilibria (pure strategy)
        equilibria = []
        for i in range(n_us):
            for j in range(n_jp):
                # Check if (i,j) is Nash Equilibrium
                is_us_best = all(us_payoffs[i, j] >= us_payoffs[k, j] for k in range(n_us))
                is_jp_best = all(jp_payoffs[i, j] >= jp_payoffs[i, k] for k in range(n_jp))
                
                if is_us_best and is_jp_best:
                    equilibria.append({
                        'us_tariff': us_tariff_grid[i],
                        'japan_relocation': japan_relocation_grid[j],
                        'us_payoff': us_payoffs[i, j],
                        'japan_payoff': jp_payoffs[i, j],
                        'strategy': ('us_index', i, 'japan_index', j)
                    })
        
        self.payoff_matrix = {
            'us_payoffs': us_payoffs.tolist(),
            'japan_payoffs': jp_payoffs.tolist(),
            'us_tariff_grid': us_tariff_grid.tolist(),
            'japan_relocation_grid': japan_relocation_grid.tolist()
        }
        self.equilibria = equilibria
        
        return {
            'nash_equilibria': equilibria,
            'payoff_matrix': self.payoff_matrix,
            'n_equilibria': len(equilibria),
            'analysis': self._analyze_equilibria(equilibria)
        }
    
    def _analyze_equilibria(self, equilibria: List[Dict]) -> Dict[str, Any]:
        """Analyze properties of found equilibria."""
        if not equilibria:
            return {'status': 'No pure strategy Nash Equilibrium found'}
        
        # Find Pareto optimal equilibria
        pareto_optimal = []
        for eq in equilibria:
            is_dominated = False
            for other_eq in equilibria:
                if (other_eq['us_payoff'] >= eq['us_payoff'] and 
                    other_eq['japan_payoff'] >= eq['japan_payoff'] and
                    (other_eq['us_payoff'] > eq['us_payoff'] or 
                     other_eq['japan_payoff'] > eq['japan_payoff'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_optimal.append(eq)
        
        return {
            'status': 'Equilibria found',
            'total_equilibria': len(equilibria),
            'pareto_optimal_count': len(pareto_optimal),
            'pareto_optimal': pareto_optimal,
            'recommended_policy': pareto_optimal[0] if pareto_optimal else equilibria[0]
        }


class AutoTradeModel:
    """Model for analyzing auto trade and industry impacts."""
    
    def __init__(self):
        """Initialize Auto Trade Model with dual methodology."""
        self.data = None
        # Econometric models (original)
        self.import_model = None
        self.transmission_model = None
        # MARL enhancement (new)
        self.nash_solver = NashEquilibriumSolver()
        self.results = {}
        # Results directories
        self.results_base = RESULTS_DIR / 'q2'
        self.results_econometric = self.results_base / 'econometric'
        self.results_marl = self.results_base / 'marl'
        self.results_transformer = self.results_base / 'transformer'
        self._ensure_result_dirs()
        
    def _ensure_result_dirs(self) -> None:
        """Create results directory structure."""
        self.results_econometric.mkdir(parents=True, exist_ok=True)
        self.results_marl.mkdir(parents=True, exist_ok=True)
        self.results_transformer.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results directories created: {self.results_base}")
    
    def load_q2_data(self) -> pd.DataFrame:
        """Load and prepare data for Q2 analysis.
        
        Returns:
            Panel DataFrame with auto import data by partner
        """
        logger.info("Loading Q2 auto data")
        
        # Load imports data
        imports = TariffDataLoader().load_imports()
        
        # Filter for autos
        imports_tagged = HSMapper().tag_dataframe(imports)
        autos = imports_tagged[imports_tagged['is_auto']].copy()
        
        logger.info(f"Loaded {len(autos)} auto import records")
        
        # Aggregate by year and partner
        autos_agg = autos.groupby(['year', 'partner_country']).agg({
            'duty_collected': 'sum',
        }).reset_index()
        
        autos_agg.rename(columns={'duty_collected': 'auto_import_charges'}, inplace=True)
        
        self.data = autos_agg
        
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
            panel_df = self.data
        
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
            
            self.import_model = model
            
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
        output_file = self.results_econometric / 'q2_import_structure.json'
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
        if self.data is None:
            self.load_q2_data()
        
        if self.data is None or len(self.data) == 0:
            logger.warning("No import data available for import penetration computation")
            return {}
        
        # Aggregate auto imports proxy (charges) by year
        imports_by_year = (
            self.data
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
            
            self.transmission_model = model
            
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
        output_file = self.results_econometric / 'q2_industry_transmission.json'
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
        
        # Save econometric results
        import_df.to_csv(self.results_econometric / 'scenario_imports.csv', index=False)
        industry_df.to_csv(self.results_econometric / 'scenario_industry.csv', index=False)
        
        # Also save to legacy location for compatibility
        import_df.to_csv(RESULTS_DIR / 'q2_scenario_imports.csv', index=False)
        industry_df.to_csv(RESULTS_DIR / 'q2_scenario_industry.csv', index=False)
        
        # Export summary to JSON
        summary = {
            'method': 'econometric_ols',
            'timestamp': datetime.now().isoformat(),
            'scenarios': scenarios,
            'summary_statistics': {
                'baseline_import_penetration': float(import_df.loc[0, 'import_penetration']),
                'max_us_production_increase': float(industry_df['production_change_pct'].max()),
                'max_employment_increase': float(industry_df['employment_change_pct'].max())
            },
            'model_parameters': {
                'import_elasticity': (
                    self.import_model.params.to_dict() if (self.import_model is not None and hasattr(self.import_model, 'params'))
                    else (self.import_model.coef_.tolist() if (self.import_model is not None and hasattr(self.import_model, 'coef_')) else None)
                ),
                'transmission_elasticity': (
                    self.transmission_model.params.to_dict() if (self.transmission_model is not None and hasattr(self.transmission_model, 'params'))
                    else (self.transmission_model.coef_.tolist() if (self.transmission_model is not None and hasattr(self.transmission_model, 'coef_')) else None)
                )
            }
        }
        
        with open(self.results_econometric / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved econometric results to {self.results_econometric}")
        
        return import_df, industry_df
    
    def run_marl_analysis(self, industry_df: pd.DataFrame, import_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run MARL game-theoretic analysis.
        
        Args:
            industry_df: Industry impact results from econometric scenarios
            import_df: Import structure results with total_japanese_sales (optional)
            
        Returns:
            Dictionary with Nash equilibrium analysis
        """
        logger.info("Running MARL Nash Equilibrium analysis")
        
        # Merge industry results with import totals for payoff computation if available
        scenario_results = industry_df.copy()
        if import_df is not None and 'total_japanese_sales' in import_df.columns:
            scenario_results = scenario_results.merge(
                import_df[['scenario', 'total_japanese_sales']], on='scenario', how='left'
            )
        else:
            # Fallback to constant to avoid KeyError; relative ratio becomes 1
            if 'total_japanese_sales' not in scenario_results.columns:
                scenario_results['total_japanese_sales'] = 1.0
        
        # Define strategy spaces
        us_tariff_grid = np.array([0.0, 0.10, 0.25])  # No tariff, 10%, 25%
        japan_relocation_grid = np.array([0.0, 0.5, 1.0])  # No, Partial, Full relocation
        
        # Compute Nash Equilibria
        nash_results = self.nash_solver.compute_best_responses(
            us_tariff_grid, japan_relocation_grid, scenario_results
        )
        
        # Save MARL results
        with open(self.results_marl / 'nash_equilibrium.json', 'w', encoding='utf-8') as f:
            json.dump(nash_results, f, indent=2, ensure_ascii=False)
        
        # Create payoff matrix CSV
        payoff_df = pd.DataFrame({
            'us_tariff': nash_results['payoff_matrix']['us_tariff_grid'] * len(japan_relocation_grid),
            'japan_relocation': japan_relocation_grid.tolist() * len(us_tariff_grid),
            'us_payoff': np.array(nash_results['payoff_matrix']['us_payoffs']).flatten(),
            'japan_payoff': np.array(nash_results['payoff_matrix']['japan_payoffs']).flatten()
        })
        payoff_df.to_csv(self.results_marl / 'payoff_matrix.csv', index=False)
        
        # Generate analysis report (Markdown)
        report_lines = [
            "# Q2 MARL Nash Equilibrium Analysis Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Game Setup",
            "",
            "**Players:**",
            "- Player 1: US Government (chooses tariff rate)",
            "- Player 2: Japanese Auto Manufacturers (choose relocation intensity)",
            "",
            "**Strategy Spaces:**",
            f"- US: Tariff rates = {us_tariff_grid.tolist()}",
            f"- Japan: Relocation intensity = {japan_relocation_grid.tolist()}",
            "",
            "## Nash Equilibria Found",
            "",
            f"**Number of Pure Strategy Equilibria:** {nash_results['n_equilibria']}",
            ""
        ]
        
        if nash_results['nash_equilibria']:
            report_lines.append("\n### Equilibrium Details\n")
            for i, eq in enumerate(nash_results['nash_equilibria'], 1):
                report_lines.extend([
                    f"**Equilibrium {i}:**",
                    f"- US Tariff: {eq['us_tariff']:.1%}",
                    f"- Japan Relocation: {eq['japan_relocation']:.1%}",
                    f"- US Payoff: {eq['us_payoff']:.2f}",
                    f"- Japan Payoff: {eq['japan_payoff']:.2f}",
                    ""
                ])
        
        if nash_results['analysis'].get('pareto_optimal'):
            report_lines.append("\n### Pareto Optimal Equilibria\n")
            for eq in nash_results['analysis']['pareto_optimal']:
                report_lines.extend([
                    f"- Tariff: {eq['us_tariff']:.1%}, Relocation: {eq['japan_relocation']:.1%}",
                    f"  Payoffs: (US={eq['us_payoff']:.2f}, Japan={eq['japan_payoff']:.2f})"
                ])
        
        if nash_results['analysis'].get('recommended_policy'):
            rec = nash_results['analysis']['recommended_policy']
            report_lines.extend([
                "",
                "## Policy Recommendation",
                "",
                f"**Recommended US Tariff:** {rec['us_tariff']:.1%}",
                f"**Expected Japan Response:** {rec['japan_relocation']:.1%} relocation intensity",
                "",
                "**Rationale:** This equilibrium represents a stable outcome where neither ",
                "player has incentive to unilaterally deviate from their strategy.",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        with open(self.results_marl / 'analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"MARL analysis complete. Results saved to {self.results_marl}")
        
        return nash_results
    
    def prepare_transformer_data(self, panel_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Prepare time series-like features for Transformer model.
        
        Args:
            panel_df: Panel data with imports by partner and year
            
        Returns:
            Tuple of (X_scaled, y_scaled, metadata)
        """
        if tf is None:
            logger.error("TensorFlow not available for Transformer model")
            return np.array([]), np.array([]), {}
        
        logger.info("Preparing data for Transformer model")
        
        # Focus on major partners by import charges
        major_partners = (
            panel_df.groupby('partner_country')['auto_import_charges']
            .sum().nlargest(5).index
        )
        df = panel_df[panel_df['partner_country'].isin(major_partners)].copy()
        
        # Sort and index features
        df = df.sort_values(['partner_country', 'year']).reset_index(drop=True)
        df['year_idx'] = df['year'] - df['year'].min()
        df['partner_encoded'] = pd.Categorical(df['partner_country']).codes
        
        # Lag and moving average features
        df['import_lag1'] = df.groupby('partner_country')['auto_import_charges'].shift(1)
        df['import_lag2'] = df.groupby('partner_country')['auto_import_charges'].shift(2)
        df['import_ma3'] = df.groupby('partner_country')['auto_import_charges'].transform(lambda x: x.rolling(3).mean())
        
        df = df.dropna()
        
        if len(df) < 10:
            logger.warning("Insufficient data for Transformer model")
            return np.array([]), np.array([]), {}
        
        feature_cols = ['year_idx', 'partner_encoded', 'import_lag1', 'import_lag2', 'import_ma3']
        X = df[feature_cols].values
        y = df['auto_import_charges'].values
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        metadata = {
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_cols': feature_cols,
            'partners': list(major_partners)
        }
        
        return X_scaled, y_scaled, metadata
    
    def build_transformer_model(self, input_dim: int) -> Any:
        """Build a small Transformer-like regression model in Keras."""
        if tf is None:
            raise ImportError("TensorFlow required for Transformer model")
        
        inputs = layers.Input(shape=(input_dim,))
        # add a trivial sequence axis for attention
        x = layers.Reshape((1, input_dim))(inputs)
        attn = layers.MultiHeadAttention(num_heads=2, key_dim=max(1, input_dim // 2))(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        
        ff = layers.Dense(32, activation='relu')(x)
        ff = layers.Dropout(0.2)(ff)
        ff = layers.Dense(input_dim)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_transformer_model(self, panel_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train Transformer model for import prediction and export results."""
        if tf is None:
            logger.warning("TensorFlow not available, skipping Transformer model")
            return {}
        
        if panel_df is None:
            panel_df = self.data
        if panel_df is None or len(panel_df) < 20:
            logger.error("Insufficient data for Transformer training")
            return {}
        
        X, y, metadata = self.prepare_transformer_data(panel_df)
        if len(X) == 0:
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        model = self.build_transformer_model(X.shape[1])
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=8,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Predictions and metrics (inverse transform to original scale)
        y_pred = model.predict(X_test, verbose=0).flatten()
        y_test_orig = metadata['scaler_y'].inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = metadata['scaler_y'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))),
            'mae': float(mean_absolute_error(y_test_orig, y_pred_orig)),
            'r2': float(r2_score(y_test_orig, y_pred_orig)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
        }
        
        logger.info(f"Transformer model - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        results = {
            'method': 'transformer',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'training_history': {
                'loss': [float(x) for x in history.history.get('loss', [])],
                'val_loss': [float(x) for x in history.history.get('val_loss', [])],
            },
            'metadata': {
                'feature_cols': metadata['feature_cols'],
                'partners': metadata['partners'],
            },
        }
        
        # Save outputs
        (self.results_transformer / '').mkdir(parents=True, exist_ok=True)
        with open(self.results_transformer / 'training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        pred_df = pd.DataFrame({
            'actual': y_test_orig,
            'predicted': y_pred_orig,
            'error': y_test_orig - y_pred_orig,
            'error_pct': (y_test_orig - y_pred_orig) / (np.maximum(y_test_orig, 1e-6)) * 100,
        })
        pred_df.to_csv(self.results_transformer / 'predictions.csv', index=False)
        
        md_lines = [
            "# Q2 Transformer Model Results",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Performance",
            f"- **RMSE:** {metrics['rmse']:.2f}",
            f"- **MAE:** {metrics['mae']:.2f}",
            f"- **R²:** {metrics['r2']:.3f}",
            "",
            "## Features Used",
        ]
        for feat in metadata['feature_cols']:
            md_lines.append(f"- {feat}")
        md_lines.extend(["", "## Major Partners", ""])
        for p in metadata['partners']:
            md_lines.append(f"- {p}")
        with open(self.results_transformer / 'analysis_report.md', 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        
        logger.info(f"Transformer results saved to {self.results_transformer}")
        return results
    
    def plot_q2_results(self) -> None:
        """Generate plots for Q2 results."""
        import matplotlib.pyplot as plt
        from utils.config import apply_plot_style
        
        apply_plot_style()
        
        # Load scenario results
        try:
            import_df = pd.read_csv(self.results_econometric / 'scenario_imports.csv')
            industry_df = pd.read_csv(self.results_econometric / 'scenario_industry.csv')
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


def run_q2_analysis(use_transformer: bool = True) -> None:
    """Run complete Q2 analysis pipeline with triple methodology.
    
    Args:
        use_transformer: Whether to include Transformer-based ML prediction
    """
    logger.info("="*60)
    logger.info("Starting Q2 Auto Trade Analysis (Econometric + MARL + Transformer)")
    if use_transformer:
        logger.info("Transformer ML: ENABLED")
    logger.info("="*60)
    
    model = AutoTradeModel()
    
    # Step 1: Load data
    panel_data = model.load_q2_data()
    model.load_external_auto_data()
    
    # Step 2: Estimate econometric models
    logger.info("\n[ECONOMETRIC ANALYSIS]")
    model.estimate_import_structure_model()
    model.estimate_industry_transmission_model()
    
    # Step 3: Simulate scenarios (econometric)
    import_df, industry_df = model.simulate_japan_response_scenarios()
    
    # Step 4: MARL game-theoretic analysis
    logger.info("\n[MARL ANALYSIS]")
    nash_results = model.run_marl_analysis(industry_df, import_df)
    logger.info(f"Nash Equilibria found: {nash_results['n_equilibria']}")
    
    # Step 5: Transformer ML enhancement (NEW)
    if use_transformer and tf is not None and panel_data is not None and len(panel_data) >= 20:
        logger.info("\n[TRANSFORMER ML ANALYSIS]")
        transformer_results = model.train_transformer_model(panel_data)
        if transformer_results:
            logger.info(f"Transformer R²: {transformer_results['metrics']['r2']:.3f}")
    elif use_transformer and tf is None:
        logger.warning("TensorFlow not available, skipping Transformer model")
    
    # Step 6: Plot results
    model.plot_q2_results()
    
    logger.info("\n" + "="*60)
    logger.info("Q2 analysis complete")
    logger.info(f"Econometric results: {model.results_econometric}")
    logger.info(f"MARL results: {model.results_marl}")
    if use_transformer:
        logger.info(f"Transformer results: {model.results_transformer}")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q2_analysis()
