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
from dataclasses import dataclass
from datetime import datetime
import math

from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

import sys
sys.path.append(str(Path(__file__).parents[1]))
from utils.config import (
    RESULTS_DIR,
    FIGURES_DIR,
    DATA_EXTERNAL,
    DATA_PROCESSED,
    ensure_directories,
)
from utils.data_loader import TariffDataLoader
from utils.mapping import HSMapper

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except ImportError:  # pragma: no cover - TensorFlow unavailable during linting
    tf = None

logger = logging.getLogger(__name__)


MONTHLY_DATA_FILE = DATA_PROCESSED / 'q1' / 'q1_1.csv'
ANNUAL_DATA_FILE = DATA_PROCESSED / 'q1' / 'q1_0.csv'
TARGET_COLUMNS = ['import_quantity', 'unit_price', 'market_share']
Q1_RESULTS_DIR = RESULTS_DIR / 'q1'


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
        output_dir = Q1_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'q1_elasticities.json'
        
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
        output_dir = Q1_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'q1_scenario_exports.csv'
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
            results_file = Q1_RESULTS_DIR / 'q1_scenario_exports.csv'
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


class SoybeanMonthlyDataset:
    """Load and align monthly soybean import data for LSTM pipeline."""

    def __init__(self, monthly_file: Path = MONTHLY_DATA_FILE, annual_file: Path = ANNUAL_DATA_FILE):
        self.monthly_file = monthly_file
        self.annual_file = annual_file

    def load(self) -> pd.DataFrame:
        if not self.monthly_file.exists():
            raise FileNotFoundError(f"Monthly soybean data not found: {self.monthly_file}")

        monthly = pd.read_csv(self.monthly_file)
        monthly['period'] = monthly['period'].astype(str)
        monthly['date'] = pd.to_datetime(monthly['period'], format='%Y%m')
        monthly['year'] = monthly['date'].dt.year
        monthly['month'] = monthly['date'].dt.month
        monthly['exporter'] = monthly['partnerDesc'].str.strip()
        monthly['exporter'] = monthly['exporter'].replace({
            'USA': 'US',
            'United States': 'US',
            'United States of America': 'US',
        })

        if '数量(万吨)' in monthly.columns:
            monthly['import_quantity'] = monthly['数量(万吨)'] * 10000
        else:
            monthly['import_quantity'] = monthly['netWgt'] / 1000

        monthly['import_value'] = monthly['primaryValue']
        monthly['unit_price'] = monthly['import_value'] / (monthly['import_quantity'] + 1e-6)

        annual = pd.read_csv(self.annual_file)
        tariff_lookup = annual[['year', 'exporter', 'tariff_cn_on_exporter']]
        monthly = monthly.merge(tariff_lookup, on=['year', 'exporter'], how='left')

        monthly['tariff_rate'] = monthly['tariff_cn_on_exporter']
        monthly = monthly.sort_values(['exporter', 'date']).reset_index(drop=True)
        monthly['total_value_by_month'] = monthly.groupby('date')['import_value'].transform('sum')
        monthly['market_share'] = monthly['import_value'] / (monthly['total_value_by_month'] + 1e-6)

        cols = ['date', 'year', 'month', 'exporter', 'import_quantity', 'import_value', 'unit_price', 'tariff_cn_on_exporter', 'tariff_rate', 'market_share']
        monthly = monthly[cols].copy()
        monthly = monthly.replace([np.inf, -np.inf], np.nan)

        grouped = monthly.groupby('exporter', group_keys=False)
        monthly['tariff_rate'] = grouped['tariff_cn_on_exporter'].ffill()
        n_fill_cols = [
            'import_quantity', 'import_value', 'unit_price', 'tariff_rate', 'market_share'
        ]
        monthly[n_fill_cols] = grouped[n_fill_cols].ffill()
        monthly = monthly.drop(columns=['tariff_cn_on_exporter'])
        monthly[n_fill_cols] = monthly[n_fill_cols].fillna(0)
        return monthly


@dataclass
class LSTMConfig:
    window_size: int = 12
    forecast_horizon: int = 12
    batch_size: int = 32
    epochs: int = 80
    validation_split: float = 0.2


class SoybeanDataProcessor:
    """Feature engineering and sequence construction for LSTM."""

    def __init__(self, config: LSTMConfig):
        self.config = config
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = TARGET_COLUMNS
        self._scaled_feature_matrix: Optional[np.ndarray] = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        year_min, year_max = features['year'].min(), features['year'].max()
        features['year_trend'] = (features['year'] - year_min) / max(1, (year_max - year_min))

        for lag in [1, 3, 6, 12]:
            features[f'quantity_lag_{lag}'] = features.groupby('exporter')['import_quantity'].shift(lag)
            features[f'price_lag_{lag}'] = features.groupby('exporter')['unit_price'].shift(lag)

        for window in [3, 6, 12]:
            features[f'quantity_ma_{window}'] = features.groupby('exporter')['import_quantity'].transform(lambda s: s.rolling(window).mean())
            features[f'price_ma_{window}'] = features.groupby('exporter')['unit_price'].transform(lambda s: s.rolling(window).mean())

        features['price_elasticity'] = (
            features.groupby('exporter')['import_quantity'].pct_change() /
            (features.groupby('exporter')['unit_price'].pct_change() + 1e-6)
        )
        features['tariff_impact'] = features['tariff_rate'] * features['unit_price']
        features['effective_price'] = features['unit_price'] * (1 + features['tariff_rate'])
        features['volume_growth'] = features.groupby('exporter')['import_quantity'].pct_change()

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.sort_values(['exporter', 'date']).reset_index(drop=True)
        grouped = features.groupby('exporter', group_keys=False)
        fill_cols = [col for col in features.columns if col not in ['exporter', 'date']]
        features[fill_cols] = grouped[fill_cols].ffill()
        features[fill_cols] = features[fill_cols].fillna(0)

        self.feature_columns = [
            'import_quantity', 'unit_price', 'tariff_rate', 'market_share',
            'month_sin', 'month_cos', 'year_trend',
            'quantity_lag_1', 'quantity_lag_3', 'quantity_lag_6', 'quantity_lag_12',
            'price_lag_1', 'price_lag_3', 'price_lag_6', 'price_lag_12',
            'quantity_ma_3', 'quantity_ma_6', 'quantity_ma_12',
            'price_ma_3', 'price_ma_6', 'price_ma_12',
            'price_elasticity', 'tariff_impact', 'effective_price', 'volume_growth'
        ]

        return features

    def build_supervised_arrays(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        scaled_features = feature_scaler.fit_transform(features[self.feature_columns])
        scaled_targets = target_scaler.fit_transform(features[self.target_columns])
        self.scalers['features'] = feature_scaler
        self.scalers['targets'] = target_scaler
        self._scaled_feature_matrix = scaled_features

        sequences: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        metadata: List[Dict] = []

        feature_matrix = pd.DataFrame(scaled_features, columns=self.feature_columns)
        feature_matrix['exporter'] = features['exporter'].values
        feature_matrix['date'] = features['date'].values

        target_matrix = pd.DataFrame(scaled_targets, columns=self.target_columns)
        target_matrix['exporter'] = features['exporter'].values
        target_matrix['date'] = features['date'].values

        for exporter, group in feature_matrix.groupby('exporter'):
            target_group = target_matrix[target_matrix['exporter'] == exporter]
            if len(group) <= self.config.window_size + self.config.forecast_horizon:
                continue
            group = group.sort_values('date').reset_index(drop=True)
            target_group = target_group.sort_values('date').reset_index(drop=True)

            for idx in range(self.config.window_size, len(group) - self.config.forecast_horizon + 1):
                seq = group.loc[idx - self.config.window_size: idx - 1, self.feature_columns].values
                future = target_group.loc[idx: idx + self.config.forecast_horizon - 1, self.target_columns].values
                sequences.append(seq)
                labels.append(future.reshape(-1))
                metadata.append({
                    'exporter': exporter,
                    'target_start_date': target_group.loc[idx, 'date']
                })

        if not sequences:
            raise ValueError(
                "Not enough observations to build sequences. Adjust window/horizon or extend dataset."
            )

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32), metadata

    def get_scaled_feature_matrix(self) -> np.ndarray:
        if self._scaled_feature_matrix is None:
            raise ValueError("Scaled feature matrix not available. Run build_supervised_arrays first.")
        return self._scaled_feature_matrix


class SoybeanLSTMModel:
    """Multi-output LSTM forecaster for soybean imports."""

    def __init__(self, input_shape: Tuple[int, int], output_steps: int, target_dim: int):
        if tf is None:  # pragma: no cover
            raise ImportError(
                "TensorFlow is required for the LSTM pipeline. Install tensorflow>=2.10 to enable it."
            )
        self.input_shape = input_shape
        self.output_steps = output_steps
        self.target_dim = target_dim
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            layers.LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.output_steps * self.target_dim)
        ])

        def custom_loss(y_true, y_pred):
            y_true_rs = tf.reshape(y_true, (-1, self.output_steps, self.target_dim))
            y_pred_rs = tf.reshape(y_pred, (-1, self.output_steps, self.target_dim))
            mse = tf.reduce_mean(tf.square(y_true_rs - y_pred_rs), axis=[1, 2])
            share_true = y_true_rs[:, :, 2]
            tariff_weight = tf.where(share_true > 0.1, 2.0, 1.0)
            weighted = mse * tf.reduce_mean(tariff_weight, axis=1)
            return tf.reduce_mean(weighted)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=custom_loss,
            metrics=['mae', 'mape']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100, batch_size: int = 32):
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        return history

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.model.predict(inputs, verbose=0)


class SoybeanLSTMPipeline:
    """End-to-end pipeline orchestrating data prep, training, and outputs."""

    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.dataset_loader = SoybeanMonthlyDataset()
        self.processor = SoybeanDataProcessor(self.config)
        self.model: Optional[SoybeanLSTMModel] = None

    def run(self) -> Dict:
        monthly_df = self.dataset_loader.load()
        feature_df = self.processor.prepare_features(monthly_df)
        X, y, _ = self.processor.build_supervised_arrays(feature_df)

        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        self.model = SoybeanLSTMModel(
            input_shape=(self.config.window_size, len(self.processor.feature_columns)),
            output_steps=self.config.forecast_horizon,
            target_dim=len(self.processor.target_columns)
        )

        history = self.model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size
        )

        metrics = self._evaluate_model(X_val, y_val)
        predictions = self._generate_forecasts(feature_df)
        self._persist_outputs(predictions, metrics, history.history)
        return {
            'metrics': metrics,
            'predictions': predictions,
            'history': history.history,
        }

    def _evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        if X_val.size == 0:
            return {}
        y_pred = self.model.predict(X_val)
        target_dim = len(self.processor.target_columns)
        horizon = self.config.forecast_horizon

        y_true_flat = y_val.reshape(-1, target_dim)
        y_pred_flat = y_pred.reshape(-1, target_dim)

        y_true = self.processor.scalers['targets'].inverse_transform(y_true_flat)
        y_pred_inv = self.processor.scalers['targets'].inverse_transform(y_pred_flat)

        metrics: Dict[str, Dict[str, float]] = {}
        for idx, col in enumerate(self.processor.target_columns):
            err = y_pred_inv[:, idx] - y_true[:, idx]
            mae = float(np.mean(np.abs(err)))
            rmse = float(math.sqrt(np.mean(err ** 2)))
            mape = float(np.mean(np.abs(err / (y_true[:, idx] + 1e-6))) * 100)
            metrics[col] = {'mae': mae, 'rmse': rmse, 'mape': mape}

        metrics['meta'] = {'target_dim': target_dim, 'horizon': horizon}
        return metrics

    def _generate_forecasts(self, feature_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        scaler = self.processor.scalers['features']
        scaled_matrix = scaler.transform(feature_df[self.processor.feature_columns])
        feature_df = feature_df.reset_index(drop=True)
        predictions: Dict[str, pd.DataFrame] = {}

        for exporter in feature_df['exporter'].unique():
            mask = feature_df['exporter'] == exporter
            exporter_scaled = scaled_matrix[mask]
            exporter_raw = feature_df[mask]
            if len(exporter_scaled) < self.config.window_size:
                continue
            latest_window = exporter_scaled[-self.config.window_size:]
            window_input = np.expand_dims(latest_window, axis=0)
            pred_scaled = self.model.predict(window_input)
            pred_scaled = pred_scaled.reshape(self.config.forecast_horizon, len(self.processor.target_columns))
            pred_actual = self.processor.scalers['targets'].inverse_transform(pred_scaled)

            last_date = exporter_raw['date'].max()
            future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=self.config.forecast_horizon, freq='MS')

            pred_df = pd.DataFrame(pred_actual, columns=self.processor.target_columns)
            pred_df['date'] = future_dates
            pred_df['exporter'] = exporter
            predictions[exporter] = pred_df

        return predictions

    def _persist_outputs(self, predictions: Dict[str, pd.DataFrame], metrics: Dict, history: Dict) -> None:
        ensure_directories()
        output_dir = Q1_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        if not predictions:
            logger.warning("No predictions generated; skipping persistence.")
            return

        combined = pd.concat(predictions.values(), ignore_index=True)
        combined = combined[['date', 'exporter'] + self.processor.target_columns]

        csv_path = output_dir / 'q1_lstm_predictions.csv'
        json_path = output_dir / 'q1_lstm_predictions.json'
        md_path = output_dir / 'q1_lstm_summary.md'
        log_path = output_dir / 'q1_lstm_run.json'
        residuals_path = output_dir / 'q1_lstm_residuals.csv'
        history_path = output_dir / 'q1_lstm_history.csv'

        combined.to_csv(csv_path, index=False)
        combined.to_json(json_path, orient='records', indent=2, date_format='iso')

        md_lines = [
            '# Q1 LSTM Forecast Summary',
            '',
            f"Generated on: {datetime.utcnow().isoformat()} UTC",
            '',
            '## Evaluation Metrics',
        ]
        for target, stats in metrics.items():
            if target == 'meta':
                continue
            md_lines.append(
                f"- **{target}**: MAE={stats['mae']:.2f}, RMSE={stats['rmse']:.2f}, MAPE={stats['mape']:.2f}%"
            )
        md_lines += ['', '## Sample Forecasts', combined.head(12).to_markdown(index=False)]
        md_path.write_text('\n'.join(md_lines), encoding='utf-8')

        log_entries = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'history': history,
            'records': len(combined),
            'targets': self.processor.target_columns,
            'horizon': self.config.forecast_horizon,
        }
        log_path.write_text(json.dumps(log_entries, indent=2), encoding='utf-8')

        # Save residual diagnostics when validation data exists
        if history:
            history_df = pd.DataFrame(history)
            history_df.to_csv(history_path, index=False)

        if metrics:
            residual_rows = []
            for target, stats in metrics.items():
                if target == 'meta':
                    continue
                residual_rows.append({
                    'target': target,
                    **stats,
                })
            pd.DataFrame(residual_rows).to_csv(residuals_path, index=False)


def run_q1_analysis() -> None:
    """Run complete Q1 analysis pipeline."""
    logger.info("="*60)
    logger.info("Starting Q1 Soybean Trade Analysis")
    logger.info("="*60)
    ensure_directories()
    
    model = SoybeanTradeModel()
    
    # Step 1: Load data
    model.prepare_panel_for_estimation()
    
    # Step 2: Estimate elasticities
    model.estimate_trade_elasticities()
    
    # Step 3: Simulate scenarios
    results = model.simulate_tariff_scenarios()
    
    # Step 4: Plot results
    model.plot_q1_results(results)
    
    # Step 5: LSTM forecasting pipeline
    try:
        lstm_pipeline = SoybeanLSTMPipeline()
        lstm_pipeline.run()
    except ImportError as exc:
        logger.warning("Skipping LSTM pipeline: %s", exc)
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("LSTM pipeline failed: %s", exc)

    logger.info("Q1 analysis complete")
    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    run_q1_analysis()
