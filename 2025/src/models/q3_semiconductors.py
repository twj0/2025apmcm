"""
Q3: Semiconductors: Trade, Manufacturing, and Security

Segment-specific (high/mid/low) trade regressions and partial equilibrium
to analyze efficiency-security trade-offs under tariffs, export controls,
and subsidies.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import statsmodels.formula.api as smf

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import RESULTS_DIR, FIGURES_DIR, DATA_EXTERNAL
from utils.data_loader import TariffDataLoader
from utils.mapping import HSMapper

logger = logging.getLogger(__name__)


class SemiconductorModel:
    """Model for semiconductor trade, production, and security analysis."""
    
    def __init__(self):
        """Initialize the model."""
        self.loader = TariffDataLoader()
        self.mapper = HSMapper()
        self.trade_data: Optional[pd.DataFrame] = None
        self.output_data: Optional[pd.DataFrame] = None
        self.models: Dict = {}
        
    def load_q3_data(self) -> pd.DataFrame:
        """Load and segment semiconductor trade data.
        
        Returns:
            DataFrame tagged by segment (high/mid/low)
        """
        logger.info("Loading Q3 semiconductor data")
        
        # Load imports
        imports = self.loader.load_imports()
        
        # Tag semiconductors
        imports_tagged = self.mapper.tag_dataframe(imports)
        chips = imports_tagged[imports_tagged['is_semiconductor']].copy()
        
        logger.info(f"Loaded {len(chips)} semiconductor records")
        logger.info(f"Segments: {chips['semiconductor_segment'].value_counts().to_dict()}")
        
        # Aggregate by year, partner, and segment
        chips_agg = chips.groupby([
            'year', 'partner_country', 'semiconductor_segment'
        ]).agg({
            'duty_collected': 'sum',
        }).reset_index()
        
        chips_agg.rename(columns={
            'duty_collected': 'chip_import_charges',
            'semiconductor_segment': 'segment'
        }, inplace=True)
        
        self.trade_data = chips_agg
        
        return chips_agg
    
    def load_external_chip_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load external semiconductor output and policy data.
        
        Returns:
            Tuple of (output_data, policy_data)
        """
        # Output data
        output_file = DATA_EXTERNAL / 'us_semiconductor_output.csv'
        if not output_file.exists():
            template = pd.DataFrame({
                'year': [2020, 2021, 2022, 2023, 2024],
                'segment': ['high'] * 5,
                'us_chip_output_billions': [0, 0, 0, 0, 0],
                'global_chip_demand_index': [100, 105, 110, 115, 120],
            })
            template.to_csv(output_file, index=False)
            logger.info(f"Template saved to {output_file}")
        else:
            template = pd.read_csv(output_file)
        
        # Policy data
        policy_file = DATA_EXTERNAL / 'us_chip_policies.csv'
        if not policy_file.exists():
            template_policy = pd.DataFrame({
                'year': [2020, 2021, 2022, 2023, 2024],
                'subsidy_index': [0, 0, 5, 10, 15],  # CHIPS Act starting 2022
                'export_control_china': [0, 0, 1, 1, 1],  # Export controls from 2022
            })
            template_policy.to_csv(policy_file, index=False)
            logger.info(f"Template saved to {policy_file}")
        else:
            template_policy = pd.read_csv(policy_file)
        
        self.output_data = template
        
        return template, template_policy
    
    def estimate_trade_response(self, panel_df: Optional[pd.DataFrame] = None) -> Dict:
        """Estimate trade response by segment.
        
        Model: ln(M_s,j) = α_s,j + β1·τ_s,j + β2·EC_s,j + β3·W_s + ε
        
        Args:
            panel_df: Trade panel by segment and partner
            
        Returns:
            Dictionary of segment-specific coefficients
        """
        if panel_df is None:
            panel_df = self.trade_data
        
        if panel_df is None or len(panel_df) == 0:
            logger.error("No trade data available")
            return {}
        
        logger.info("Estimating trade response by segment")
        
        results = {}
        
        for segment in ['high', 'mid', 'low']:
            seg_data = panel_df[panel_df['segment'] == segment].copy()
            
            if len(seg_data) < 10:
                logger.warning(f"Insufficient data for segment: {segment}")
                continue
            
            try:
                # Placeholder: simple trend model
                seg_data['ln_import_charges'] = np.log(seg_data['chip_import_charges'] + 1)
                formula = 'ln_import_charges ~ year + C(partner_country)'
                model = smf.ols(formula, data=seg_data).fit()
                
                results[segment] = {
                    'rsquared': float(model.rsquared),
                    'nobs': int(model.nobs),
                    'year_coef': float(model.params.get('year', 0)),
                }
                
                logger.info(f"Segment {segment} - R²: {results[segment]['rsquared']:.3f}")
                
            except Exception as e:
                logger.error(f"Error for segment {segment}: {e}")
        
        # Save results
        with open(RESULTS_DIR / 'q3_trade_response.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def estimate_output_response(self, output_df: Optional[pd.DataFrame] = None) -> Dict:
        """Estimate domestic output response by segment.
        
        Model: ln(Q_US_s) = γ_s + δ1·Subsidy + δ2·τ_eff_s + δ3·D_s + η
        
        Args:
            output_df: Output data by segment
            
        Returns:
            Dictionary of segment-specific output elasticities
        """
        if output_df is None:
            output_df, policy_df = self.load_external_chip_data()
        else:
            # Ensure we have policy data as well
            policy_file = DATA_EXTERNAL / 'us_chip_policies.csv'
            if policy_file.exists():
                policy_df = pd.read_csv(policy_file)
            else:
                logger.warning(f"Policy data not found: {policy_file}")
                policy_df = pd.DataFrame()
        
        logger.info("Estimating output response")
        
        # Merge output and policy data on year
        df = output_df.merge(policy_df, on='year', how='left')
        
        if 'us_chip_output_billions' not in df.columns:
            logger.warning("us_chip_output_billions column missing in output data")
            return {}
        
        # Drop rows without key variables
        df = df.dropna(subset=['us_chip_output_billions', 'subsidy_index'])
        
        if df.empty:
            logger.warning("No valid observations for output response estimation")
            return {}
        
        # Log-transform output so coefficients are interpretable as elasticities
        df['ln_output'] = np.log(df['us_chip_output_billions'] + 1e-6)
        
        try:
            formula = 'ln_output ~ subsidy_index + export_control_china + global_chip_demand_index'
            model = smf.ols(formula, data=df).fit()
            
            self.models['output_response'] = model
            
            subsidy_coef = float(model.params.get('subsidy_index', 0.0))
            logger.info(f"Estimated subsidy elasticity (approx.): {subsidy_coef:.3f}")
        except Exception as e:
            logger.error(f"Error estimating output response: {e}")
            subsidy_coef = 0.0
        
        # Use a common elasticity across segments unless richer data are added
        results = {}
        for segment in ['high', 'mid', 'low']:
            results[segment] = {
                'subsidy_elasticity': subsidy_coef,
                'tariff_elasticity': 0.0,
            }
        
        with open(RESULTS_DIR / 'q3_output_response.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def compute_security_metrics(
        self,
        panel_df: Optional[pd.DataFrame] = None,
        output_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Compute self-sufficiency and dependence metrics.
        
        Args:
            panel_df: Trade data
            output_df: Output data
            
        Returns:
            DataFrame with security metrics by year and segment
        """
        logger.info("Computing security metrics")
        
        # Load trade data if not provided
        if panel_df is None:
            if self.trade_data is None:
                panel_df = self.load_q3_data()
            else:
                panel_df = self.trade_data
        
        # Load output data if not provided
        if output_df is None:
            output_df, _ = self.load_external_chip_data()
        
        if panel_df is None or output_df is None:
            logger.warning("Insufficient data to compute security metrics")
            return pd.DataFrame()
        
        # Aggregate imports by year and segment
        imports_agg = panel_df.groupby(['year', 'segment'])['chip_import_charges'].sum().reset_index()
        imports_agg.rename(columns={'chip_import_charges': 'import_proxy'}, inplace=True)
        
        # Aggregate output by year and segment when available
        output_cols = ['year', 'segment', 'us_chip_output_billions']
        if not set(output_cols).issubset(output_df.columns):
            # Fallback: treat output as non-segmented and replicate across segments
            logger.warning("Segmented output data not available; using total output for all segments")
            output_total = output_df.groupby('year')['us_chip_output_billions'].sum().reset_index()
            segments = imports_agg['segment'].unique()
            expanded = []
            for _, row in output_total.iterrows():
                for seg in segments:
                    expanded.append({
                        'year': row['year'],
                        'segment': seg,
                        'us_chip_output_billions': row['us_chip_output_billions'],
                    })
            output_by_seg = pd.DataFrame(expanded)
        else:
            output_by_seg = output_df[output_cols].copy()
        
        # Merge imports and output
        metrics_df = imports_agg.merge(output_by_seg, on=['year', 'segment'], how='left')
        
        metrics_df['import_proxy'] = metrics_df['import_proxy'].clip(lower=0)
        metrics_df['us_chip_output_billions'] = metrics_df['us_chip_output_billions'].fillna(0)
        metrics_df['total_supply'] = metrics_df['import_proxy'] + metrics_df['us_chip_output_billions']
        
        # Self-sufficiency: domestic output share of total supply
        metrics_df['self_sufficiency_pct'] = np.where(
            metrics_df['total_supply'] > 0,
            metrics_df['us_chip_output_billions'] / metrics_df['total_supply'] * 100.0,
            np.nan,
        )
        
        # China dependence: share of imports sourced from China
        china_mask = panel_df['partner_country'].str.contains('China', case=False, na=False)
        china_imports = (
            panel_df[china_mask]
            .groupby(['year', 'segment'])['chip_import_charges']
            .sum()
            .reset_index(name='china_import_proxy')
        )
        metrics_df = metrics_df.merge(china_imports, on=['year', 'segment'], how='left')
        metrics_df['china_import_proxy'] = metrics_df['china_import_proxy'].fillna(0)
        metrics_df['china_dependence_pct'] = np.where(
            metrics_df['import_proxy'] > 0,
            metrics_df['china_import_proxy'] / metrics_df['import_proxy'] * 100.0,
            np.nan,
        )
        
        # Supply risk: higher when self-sufficiency is low and China dependence is high
        metrics_df['supply_risk_index'] = (
            (100.0 - metrics_df['self_sufficiency_pct'].fillna(0)) * 0.5
            + metrics_df['china_dependence_pct'].fillna(0) * 0.5
        ) / 100.0
        
        metrics_df.to_csv(RESULTS_DIR / 'q3_security_metrics.csv', index=False)
        
        return metrics_df
    
    def simulate_policy_combinations(
        self,
        trade_elasticities: Optional[Dict] = None,
        output_elasticities: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Simulate policy combinations and compute efficiency-security trade-offs.
        
        Policies:
        - A: Subsidies only
        - B: Tariffs only
        - C: Tariffs + Subsidies + Export Controls
        
        Args:
            trade_elasticities: Trade response parameters
            output_elasticities: Output response parameters
            
        Returns:
            DataFrame with policy scenarios and outcomes
        """
        logger.info("Simulating policy combinations")
        
        scenarios = {
            'Policy_A_subsidy_only': {
                'subsidy_level': 10,
                'tariff_level': 0,
                'export_control': 0,
            },
            'Policy_B_tariff_only': {
                'subsidy_level': 0,
                'tariff_level': 25,
                'export_control': 0,
            },
            'Policy_C_comprehensive': {
                'subsidy_level': 10,
                'tariff_level': 25,
                'export_control': 1,
            },
        }
        
        # Use estimated output elasticities if not provided
        if output_elasticities is None:
            output_elasticities = self.estimate_output_response()
        
        # Load baseline security metrics if available
        baseline_metrics_file = RESULTS_DIR / 'q3_security_metrics.csv'
        baseline_self_suff: Dict[str, float] = {}
        if baseline_metrics_file.exists():
            try:
                metrics_df = pd.read_csv(baseline_metrics_file)
                latest_year = metrics_df['year'].max()
                latest = metrics_df[metrics_df['year'] == latest_year]
                for seg in ['high', 'mid', 'low']:
                    vals = latest.loc[latest['segment'] == seg, 'self_sufficiency_pct']
                    if not vals.empty:
                        baseline_self_suff[seg] = float(vals.iloc[0])
            except Exception as e:
                logger.warning(f"Could not load baseline security metrics: {e}")
        
        # Default baseline if none loaded
        for seg in ['high', 'mid', 'low']:
            baseline_self_suff.setdefault(seg, 25.0)
        
        results = []
        
        for policy_name, params in scenarios.items():
            for segment in ['high', 'mid', 'low']:
                seg_elast = output_elasticities.get(segment, {})
                subsidy_elast = float(seg_elast.get('subsidy_elasticity', 0.0))
                tariff_elast = float(seg_elast.get('tariff_elasticity', 0.0))
                
                # Approximate change in self-sufficiency (percentage points)
                delta_self_suff = (
                    subsidy_elast * params['subsidy_level']
                    + tariff_elast * params['tariff_level']
                )
                
                base_ss = baseline_self_suff.get(segment, 25.0)
                self_sufficiency = base_ss + delta_self_suff
                
                # Simple cost index based on policy intensity
                cost_index = abs(params['subsidy_level']) + abs(params['tariff_level'])
                
                # Security index aligned with self-sufficiency and export controls
                security_index = self_sufficiency + params['export_control'] * 5.0
                
                results.append({
                    'policy': policy_name,
                    'segment': segment,
                    'baseline_self_sufficiency_pct': base_ss,
                    'self_sufficiency_pct': self_sufficiency,
                    'delta_self_sufficiency_pct': delta_self_suff,
                    'cost_index': cost_index,
                    'security_index': security_index,
                    'efficiency_security_ratio': (
                        security_index / (cost_index + 1e-6)
                    ),
                })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_DIR / 'q3_policy_scenarios.csv', index=False)
        
        logger.info("Saved policy scenarios")
        
        return results_df
    
    def plot_q3_results(self) -> None:
        """Generate plots for Q3 results."""
        import matplotlib.pyplot as plt
        from utils.config import apply_plot_style
        
        apply_plot_style()
        
        try:
            policy_df = pd.read_csv(RESULTS_DIR / 'q3_policy_scenarios.csv')
        except FileNotFoundError:
            logger.error("Policy scenarios not found")
            return
        
        logger.info("Creating Q3 plots")
        
        # Plot: Efficiency-Security Trade-off
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for policy in policy_df['policy'].unique():
            data = policy_df[policy_df['policy'] == policy]
            ax.scatter(
                data['cost_index'],
                data['security_index'],
                label=policy,
                s=100,
                alpha=0.7
            )
        
        ax.set_xlabel('Cost Index')
        ax.set_ylabel('Security Index')
        ax.set_title('Efficiency-Security Trade-offs: Semiconductor Policies')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'q3_efficiency_security_tradeoff.pdf')
        plt.close()
        
        logger.info("Q3 plots saved")


def run_q3_analysis() -> None:
    """Run complete Q3 analysis pipeline."""
    logger.info("="*60)
    logger.info("Starting Q3 Semiconductor Analysis")
    logger.info("="*60)
    
    model = SemiconductorModel()
    
    # Step 1: Load data
    model.load_q3_data()
    model.load_external_chip_data()
    
    # Step 2: Estimate models
    model.estimate_trade_response()
    model.estimate_output_response()
    
    # Step 3: Compute security metrics
    model.compute_security_metrics()
    
    # Step 4: Simulate policies
    model.simulate_policy_combinations()
    
    # Step 5: Plot results
    model.plot_q3_results()
    
    logger.info("Q3 analysis complete")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q3_analysis()
