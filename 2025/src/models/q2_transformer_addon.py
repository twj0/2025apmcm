"""
Transformer addon methods for Q2 Auto Trade Model.
These methods should be integrated into the AutoTradeModel class in q2_autos.py
"""

def prepare_transformer_data(self, panel_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Prepare time series data for Transformer model.
    
    Args:
        panel_df: Panel data with imports by partner and year
        
    Returns:
        Tuple of (X, y, metadata)
    """
    if tf is None:
        logger.error("TensorFlow not available for Transformer model")
        return np.array([]), np.array([]), {}
    
    logger.info("Preparing data for Transformer model")
    
    # Focus on major partners
    major_partners = panel_df.groupby('partner_country')['auto_import_charges'].sum().nlargest(5).index
    df = panel_df[panel_df['partner_country'].isin(major_partners)].copy()
    
    # Create features
    df = df.sort_values(['partner_country', 'year']).reset_index(drop=True)
    df['year_idx'] = df['year'] - df['year'].min()
    df['partner_encoded'] = pd.Categorical(df['partner_country']).codes
    
    # Add lag features
    df['import_lag1'] = df.groupby('partner_country')['auto_import_charges'].shift(1)
    df['import_lag2'] = df.groupby('partner_country')['auto_import_charges'].shift(2)
    df['import_ma3'] = df.groupby('partner_country')['auto_import_charges'].transform(lambda x: x.rolling(3).mean())
    
    df = df.dropna()
    
    if len(df) < 10:
        logger.warning("Insufficient data for Transformer model")
        return np.array([]), np.array([]), {}
    
    # Prepare sequences
    feature_cols = ['year_idx', 'partner_encoded', 'import_lag1', 'import_lag2', 'import_ma3']
    X = df[feature_cols].values
    y = df['auto_import_charges'].values
    
    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    metadata = {
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_cols': feature_cols,
        'partners': major_partners.tolist()
    }
    
    return X_scaled, y_scaled, metadata

def build_transformer_model(self, input_dim: int) -> Any:
    """Build Transformer-based prediction model.
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Compiled Keras model
    """
    if tf is None:
        raise ImportError("TensorFlow required for Transformer model")
    
    # Simplified Transformer architecture
    inputs = layers.Input(shape=(input_dim,))
    
    # Expand dims for attention
    x = layers.Reshape((1, input_dim))(inputs)
    
    # Multi-head attention
    attention = layers.MultiHeadAttention(num_heads=2, key_dim=input_dim//2)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    # Feed-forward network
    ff = layers.Dense(32, activation='relu')(x)
    ff = layers.Dropout(0.2)(ff)
    ff = layers.Dense(input_dim)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    
    # Output
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def train_transformer_model(self, panel_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Train Transformer model for import prediction.
    
    Args:
        panel_df: Panel data with imports
        
    Returns:
        Training results and metrics
    """
    if tf is None:
        logger.warning("TensorFlow not available, skipping Transformer model")
        return {}
    
    if panel_df is None:
        panel_df = self.data
    
    if panel_df is None or len(panel_df) < 20:
        logger.error("Insufficient data for Transformer training")
        return {}
    
    logger.info("Training Transformer model for import prediction")
    
    # Prepare data
    X, y, metadata = self.prepare_transformer_data(panel_df)
    
    if len(X) == 0:
        return {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Build and train model
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
    
    # Evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Inverse transform for metrics
    y_test_orig = metadata['scaler_y'].inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = metadata['scaler_y'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))),
        'mae': float(mean_absolute_error(y_test_orig, y_pred_orig)),
        'r2': float(r2_score(y_test_orig, y_pred_orig)),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    }
    
    logger.info(f"Transformer model - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
    
    # Save results
    results = {
        'method': 'transformer',
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        },
        'metadata': {
            'feature_cols': metadata['feature_cols'],
            'partners': metadata['partners']
        }
    }
    
    # Save to JSON
    with open(self.results_transformer / 'training_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save predictions to CSV
    pred_df = pd.DataFrame({
        'actual': y_test_orig,
        'predicted': y_pred_orig,
        'error': y_test_orig - y_pred_orig,
        'error_pct': (y_test_orig - y_pred_orig) / (y_test_orig + 1e-6) * 100
    })
    pred_df.to_csv(self.results_transformer / 'predictions.csv', index=False)
    
    # Save metrics summary as markdown
    md_lines = [
        "# Q2 Transformer Model Results",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Model Performance",
        "",
        f"- **RMSE:** {metrics['rmse']:.2f}",
        f"- **MAE:** {metrics['mae']:.2f}",
        f"- **R²:** {metrics['r2']:.3f}",
        "",
        "## Training Details",
        "",
        f"- **Training Samples:** {metrics['train_samples']}",
        f"- **Test Samples:** {metrics['test_samples']}",
        f"- **Final Training Loss:** {metrics['final_train_loss']:.4f}",
        f"- **Final Validation Loss:** {metrics['final_val_loss']:.4f}",
        "",
        "## Features Used",
        "",
    ]
    
    for feat in metadata['feature_cols']:
        md_lines.append(f"- {feat}")
    
    md_lines.extend([
        "",
        "## Major Partners Analyzed",
        ""
    ])
    
    for partner in metadata['partners']:
        md_lines.append(f"- {partner}")
    
    with open(self.results_transformer / 'analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    
    logger.info(f"Transformer results saved to {self.results_transformer}")
    
    return results
