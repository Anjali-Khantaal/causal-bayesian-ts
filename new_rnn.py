import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Reshape
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set plotting style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.2)

# Load the preprocessed data
data = pd.read_csv("financial_timeseries_preprocessed.csv", index_col=0, parse_dates=True)

# We'll focus on the same variables as in the VAR model
rnn_data = data[['SP500_log_return', 'FedFunds_diff']].copy()

# Print basic information about the data
print("Data shape:", rnn_data.shape)
print("\nFirst few rows:")
print(rnn_data.head())
print("\nStatistics:")
print(rnn_data.describe())

# Define function to create sequences for RNN
def create_sequences(data, seq_length):
    """
    Create sequences for RNN from time series data
    
    Parameters:
    -----------
    data : ndarray
        Time series data
    seq_length : int
        Length of sequences (lookback window)
        
    Returns:
    --------
    X : ndarray
        Input sequences
    y : ndarray
        Target values
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length (lookback window)
seq_length = 6  # 6 months of history

# Scale the data for better neural network performance
scaler = StandardScaler()
scaled_data = scaler.fit_transform(rnn_data.values)

# Create sequences
X, y = create_sequences(scaled_data, seq_length)

print(f"Sequence shape: X {X.shape}, y {y.shape}")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, shuffle=False  # No shuffling for time series
)

# Create a custom layer for Monte Carlo Dropout
class MCDropout(Dropout):
    def call(self, inputs):
        # Always apply dropout during both training and inference
        return super().call(inputs, training=True)

# Create a Bayesian LSTM model with MC Dropout
def create_bayesian_lstm_model(input_shape, outputs, dropout_rate=0.2, lstm_units=32):
    """
    Create a Bayesian LSTM model using Monte Carlo Dropout
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input sequences (seq_length, features)
    outputs : int
        Number of output features
    dropout_rate : float
        Dropout rate for Bayesian approximation
    lstm_units : int
        Number of LSTM units
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Bayesian LSTM model
    """
    inputs = Input(shape=input_shape, name="input_layer")
    
    # LSTM layer with dropout
    x = LSTM(lstm_units, return_sequences=True)(inputs)
    x = MCDropout(dropout_rate)(x)  # Use the custom MCDropout layer
    
    # Second LSTM layer
    x = LSTM(lstm_units)(x)
    x = MCDropout(dropout_rate)(x)  # Use the custom MCDropout layer
    
    # Output layer
    outputs = Dense(outputs)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Custom loss function that predicts mean and learns variance
    def negative_log_likelihood(y_true, y_pred):
        # MSE loss (Gaussian negative log likelihood with constant variance)
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    
    model.compile(loss=negative_log_likelihood, optimizer=Adam(learning_rate=0.001))
    
    return model

# Create and train the Bayesian LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])
outputs = y_train.shape[1]

bayesian_lstm = create_bayesian_lstm_model(input_shape, outputs)
bayesian_lstm.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Train the model
print("Training the Bayesian RNN model...")
history = bayesian_lstm.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('bayesian_rnn_training.png')
plt.close()

# Function to perform MC Dropout predictions
def mc_dropout_predict(model, X, n_samples=100):
    """
    Perform Monte Carlo Dropout predictions
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Bayesian model with dropout
    X : ndarray
        Input data
    n_samples : int
        Number of MC samples
        
    Returns:
    --------
    y_pred_mean : ndarray
        Mean predictions
    y_pred_std : ndarray
        Standard deviation of predictions (uncertainty)
    predictions : ndarray
        All MC samples
    """
    predictions = np.zeros((n_samples, X.shape[0], model.output_shape[1]))
    
    # Ensure the function runs even for a single input
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=0)
    
    for i in range(n_samples):
        predictions[i] = model.predict(X, verbose=0)
    
    y_pred_mean = np.mean(predictions, axis=0)
    y_pred_std = np.std(predictions, axis=0)
    
    return y_pred_mean, y_pred_std, predictions

# Evaluate the model on validation data
print("Evaluating the model with MC Dropout...")
y_val_pred_mean, y_val_pred_std, val_predictions = mc_dropout_predict(bayesian_lstm, X_val)

# Calculate RMSE on the original scale
# First transform predictions back to original scale
y_val_pred_mean_orig = np.array([scaler.inverse_transform(pred.reshape(1, -1))[0] for pred in y_val_pred_mean])
y_val_orig = np.array([scaler.inverse_transform(actual.reshape(1, -1))[0] for actual in y_val])

# Calculate RMSE
rmse = np.sqrt(np.mean((y_val_orig - y_val_pred_mean_orig) ** 2, axis=0))
print(f"RMSE for SP500_log_return: {rmse[0]:.4f}")
print(f"RMSE for FedFunds_diff: {rmse[1]:.4f}")

# Calculate prediction intervals in the original scale
# For each prediction, generate samples, transform them to original scale, then compute quantiles
lower_bounds = np.zeros_like(y_val_pred_mean_orig)
upper_bounds = np.zeros_like(y_val_pred_mean_orig)

for i in range(len(y_val)):
    # Get all MC samples for this prediction
    samples = val_predictions[:, i, :]
    
    # Transform each sample back to original scale
    samples_orig = np.array([scaler.inverse_transform(sample.reshape(1, -1))[0] for sample in samples])
    
    # Compute 2.5% and 97.5% quantiles
    lower_bounds[i] = np.percentile(samples_orig, 2.5, axis=0)
    upper_bounds[i] = np.percentile(samples_orig, 97.5, axis=0)

# Calculate coverage
in_interval = np.logical_and(
    y_val_orig >= lower_bounds,
    y_val_orig <= upper_bounds
)

coverage = np.mean(in_interval, axis=0)
print(f"95% Prediction interval coverage for SP500_log_return: {coverage[0]:.4f}")
print(f"95% Prediction interval coverage for FedFunds_diff: {coverage[1]:.4f}")

# Plot the predictions vs actual for validation set
plt.figure(figsize=(14, 10))

# Plot S&P 500 log returns
plt.subplot(2, 1, 1)
plt.plot(y_val_orig[:, 0], 'b-', label='Actual', alpha=0.7)
plt.plot(y_val_pred_mean_orig[:, 0], 'r-', label='Predicted', alpha=0.7)
plt.fill_between(
    np.arange(len(y_val_orig)),
    lower_bounds[:, 0],
    upper_bounds[:, 0],
    color='r', alpha=0.2, label='95% Prediction Interval'
)
plt.title('S&P 500 Log Returns')
plt.legend()

# Plot Fed Funds Rate differences
plt.subplot(2, 1, 2)
plt.plot(y_val_orig[:, 1], 'g-', label='Actual', alpha=0.7)
plt.plot(y_val_pred_mean_orig[:, 1], 'r-', label='Predicted', alpha=0.7)
plt.fill_between(
    np.arange(len(y_val_orig)),
    lower_bounds[:, 1],
    upper_bounds[:, 1],
    color='r', alpha=0.2, label='95% Prediction Interval'
)
plt.title('Federal Funds Rate Differences')
plt.legend()

plt.tight_layout()
plt.savefig('bayesian_rnn_validation.png')
plt.close()

# Function to generate forecasts with fixed MC Dropout
def generate_forecasts(model, last_sequence, n_steps, scaler, n_samples=100):
    """
    Generate multi-step forecasts from the Bayesian RNN
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained Bayesian RNN model
    last_sequence : ndarray
        Last observed sequence for initial prediction
    n_steps : int
        Number of steps to forecast
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used for data preprocessing
    n_samples : int
        Number of MC samples for each forecast
        
    Returns:
    --------
    forecast_means_orig : ndarray
        Mean forecasts for each step (original scale)
    forecast_lower_orig : ndarray
        Lower bounds of forecasts (original scale)
    forecast_upper_orig : ndarray
        Upper bounds of forecasts (original scale)
    forecast_samples : ndarray
        All MC samples for each forecast step (scaled)
    """
    # Generate multiple forecast paths (one complete path per MC sample)
    forecast_samples = np.zeros((n_samples, n_steps, model.output_shape[1]))
    
    for s in range(n_samples):
        # Start with the last observed sequence for this sample
        current_sequence = last_sequence.copy()
        
        # Generate a complete path
        for step in range(n_steps):
            # Make a single prediction with the current sequence
            pred = model.predict(np.expand_dims(current_sequence, axis=0), verbose=0)[0]
            
            # Store the prediction
            forecast_samples[s, step] = pred
            
            # Update sequence for next step prediction
            current_sequence = np.vstack([current_sequence[1:], pred])
    
    # Calculate statistics in the original scale
    forecast_means_orig = np.zeros((n_steps, model.output_shape[1]))
    forecast_lower_orig = np.zeros((n_steps, model.output_shape[1]))
    forecast_upper_orig = np.zeros((n_steps, model.output_shape[1]))
    
    for step in range(n_steps):
        # Get all samples for this step across all paths
        step_samples = forecast_samples[:, step, :]
        
        # Transform each sample back to original scale
        step_samples_orig = np.array([scaler.inverse_transform(sample.reshape(1, -1))[0] 
                                      for sample in step_samples])
        
        # Calculate statistics
        forecast_means_orig[step] = np.mean(step_samples_orig, axis=0)
        forecast_lower_orig[step] = np.percentile(step_samples_orig, 2.5, axis=0)
        forecast_upper_orig[step] = np.percentile(step_samples_orig, 97.5, axis=0)
    
    return forecast_means_orig, forecast_lower_orig, forecast_upper_orig, forecast_samples

# Get the last sequence from the data
last_sequence = scaled_data[-seq_length:]

# Generate forecasts for next 24 months
forecast_horizon = 24
print(f"Generating forecasts for the next {forecast_horizon} months...")
forecast_means, forecast_lower, forecast_upper, forecast_samples = generate_forecasts(
    bayesian_lstm, last_sequence, forecast_horizon, scaler
)

# Setup dates for forecast plotting
last_date = data.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='ME')[1:]

# Plot the forecasts
plt.figure(figsize=(14, 10))

# Plot S&P 500 log returns forecast
plt.subplot(2, 1, 1)
# Plot historical data (last 36 months)
hist_dates = data.index[-36:]
hist_values = data['SP500_log_return'].values[-36:]
plt.plot(hist_dates, hist_values, 'b-', label='Historical', alpha=0.7)

# Plot forecast
plt.plot(forecast_dates, forecast_means[:, 0], 'r-', label='Forecast Mean', alpha=0.7)
plt.fill_between(
    forecast_dates,
    forecast_lower[:, 0],
    forecast_upper[:, 0],
    color='r', alpha=0.2, label='95% Prediction Interval'
)
plt.axvline(x=last_date, color='k', linestyle='--', alpha=0.5)
plt.title('S&P 500 Log Returns Forecast')
plt.legend()

# Plot Fed Funds Rate differences forecast
plt.subplot(2, 1, 2)
# Plot historical data (last 36 months)
hist_values = data['FedFunds_diff'].values[-36:]
plt.plot(hist_dates, hist_values, 'g-', label='Historical', alpha=0.7)

# Plot forecast
plt.plot(forecast_dates, forecast_means[:, 1], 'r-', label='Forecast Mean', alpha=0.7)
plt.fill_between(
    forecast_dates,
    forecast_lower[:, 1],
    forecast_upper[:, 1],
    color='r', alpha=0.2, label='95% Prediction Interval'
)
plt.axvline(x=last_date, color='k', linestyle='--', alpha=0.5)
plt.title('Federal Funds Rate Differences Forecast')
plt.legend()

plt.tight_layout()
plt.savefig('bayesian_rnn_forecasts.png')
plt.close()

# Function to simulate interventions with fixed intervention handling
def simulate_intervention(model, last_sequence, intervention_series, intervention_value,
                         n_steps, scaler, n_samples=100):
    """
    Simulate the effect of an intervention on one series
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained Bayesian RNN model
    last_sequence : ndarray
        Last observed sequence for initial prediction
    intervention_series : int
        Index of the series to intervene on (0 for SP500_log_return, 1 for FedFunds_diff)
    intervention_value : float
        The value to set for the intervention (in original scale)
    n_steps : int
        Number of steps to forecast
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used for data preprocessing
    n_samples : int
        Number of MC samples for each forecast
        
    Returns:
    --------
    baseline_means : ndarray
        Mean forecasts without intervention
    baseline_lower : ndarray
        Lower bounds of forecasts without intervention
    baseline_upper : ndarray
        Upper bounds of forecasts without intervention
    intervention_means : ndarray
        Mean forecasts with intervention
    intervention_lower : ndarray
        Lower bounds of forecasts with intervention
    intervention_upper : ndarray
        Upper bounds of forecasts with intervention
    """
    # First generate baseline forecasts (no intervention)
    baseline_means, baseline_lower, baseline_upper, baseline_samples = generate_forecasts(
        model, last_sequence, n_steps, scaler, n_samples
    )
    
    # Now generate intervention forecasts using a similar approach to the baseline
    # but with the intervention applied at the first step
    
    # Generate multiple forecast paths with intervention
    intervention_samples = np.zeros((n_samples, n_steps, model.output_shape[1]))
    
    for s in range(n_samples):
        # Start with the last observed sequence for this sample
        current_sequence = last_sequence.copy()
        
        # Generate a complete path
        for step in range(n_steps):
            # Make a single prediction with the current sequence
            pred = model.predict(np.expand_dims(current_sequence, axis=0), verbose=0)[0]
            
            # Apply intervention at the first step
            if step == 0:
                # Need to transform predictions, apply intervention, then transform back
                # First get the prediction in original scale
                pred_orig = scaler.inverse_transform(pred.reshape(1, -1))[0]
                
                # Apply intervention - add the specified change to the target series
                pred_orig[intervention_series] = pred_orig[intervention_series] + intervention_value
                
                # Transform back to scaled form
                pred = scaler.transform(pred_orig.reshape(1, -1))[0]
            
            # Store the prediction
            intervention_samples[s, step] = pred
            
            # Update sequence for next step prediction
            current_sequence = np.vstack([current_sequence[1:], pred])
    
    # Calculate statistics in the original scale
    intervention_means_orig = np.zeros((n_steps, model.output_shape[1]))
    intervention_lower_orig = np.zeros((n_steps, model.output_shape[1]))
    intervention_upper_orig = np.zeros((n_steps, model.output_shape[1]))
    
    for step in range(n_steps):
        # Get all samples for this step across all paths
        step_samples = intervention_samples[:, step, :]
        
        # Transform each sample back to original scale
        step_samples_orig = np.array([scaler.inverse_transform(sample.reshape(1, -1))[0] 
                                      for sample in step_samples])
        
        # Calculate statistics
        intervention_means_orig[step] = np.mean(step_samples_orig, axis=0)
        intervention_lower_orig[step] = np.percentile(step_samples_orig, 2.5, axis=0)
        intervention_upper_orig[step] = np.percentile(step_samples_orig, 97.5, axis=0)
    
    return (baseline_means, baseline_lower, baseline_upper,
            intervention_means_orig, intervention_lower_orig, intervention_upper_orig)

# Define an intervention scenario: Fed Funds Rate hike of 0.5%
intervention_series = 1  # Index for FedFunds_diff
intervention_value = 0.5  # 0.5 percentage point increase

# Simulate the intervention
print("Simulating intervention (Fed Funds Rate hike of 0.5%)...")
(baseline_means, baseline_lower, baseline_upper,
 intervention_means, intervention_lower, intervention_upper) = simulate_intervention(
    bayesian_lstm, last_sequence, intervention_series, intervention_value, 
    forecast_horizon, scaler
)

# Calculate causal effect (difference between intervention and baseline)
causal_effect = intervention_means - baseline_means

# Calculate average and cumulative effects
effect_6m_sp500 = causal_effect[:6, 0].mean()
effect_12m_sp500 = causal_effect[:12, 0].mean()
cumulative_effect_12m_sp500 = np.sum(causal_effect[:12, 0])

print("\nIntervention Analysis Results:")
print(f"- Interest Rate Intervention: +{intervention_value} percentage points")
print(f"- Average effect on S&P 500 monthly returns over 6 months: {effect_6m_sp500:.4f}")
print(f"- Average effect on S&P 500 monthly returns over 12 months: {effect_12m_sp500:.4f}")
print(f"- Cumulative effect on S&P 500 returns over 12 months: {cumulative_effect_12m_sp500:.4f}")

# Plot intervention analysis
plt.figure(figsize=(16, 12))

# Plot both series with and without intervention
for i, series_name in enumerate(['SP500_log_return', 'FedFunds_diff']):
    plt.subplot(3, 2, i + 1)
    
    # Plot baseline forecast
    plt.plot(forecast_dates, baseline_means[:, i], 'b-', label='Baseline')
    plt.fill_between(
        forecast_dates,
        baseline_lower[:, i],
        baseline_upper[:, i],
        color='b', alpha=0.1
    )
    
    # Plot intervention forecast
    plt.plot(forecast_dates, intervention_means[:, i], 'r-', label='Intervention')
    plt.fill_between(
        forecast_dates,
        intervention_lower[:, i],
        intervention_upper[:, i],
        color='r', alpha=0.1
    )
    
    plt.title(f'{series_name}: Baseline vs. Intervention')
    plt.legend()
    
    # If this is the intervention variable, add a note
    if i == intervention_series:
        plt.annotate(f'Intervention: +{intervention_value}', xy=(0.05, 0.9), 
                    xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

# Plot the causal effect (difference)
for i, series_name in enumerate(['SP500_log_return', 'FedFunds_diff']):
    plt.subplot(3, 2, i + 3)
    
    plt.plot(forecast_dates, causal_effect[:, i], 'g-', label='Causal Effect')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Causal Effect on {series_name}')
    
    # Compute cumulative effect over time
    cum_effect = np.cumsum(causal_effect[:, i])
    
    plt.subplot(3, 2, i + 5)
    plt.plot(forecast_dates, cum_effect, 'm-', label='Cumulative Effect')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Cumulative Effect on {series_name}')

plt.tight_layout()
plt.savefig('bayesian_rnn_intervention.png')
plt.close()

# Compare RNN with VAR model
print("\nComparison with Bayesian VAR model:")
print("Bayesian VAR Model:")
print("- Average effect on S&P 500 monthly returns over 6 months: -0.0020")
print("- Average effect on S&P 500 monthly returns over 12 months: -0.0009")
print("- Cumulative effect on S&P 500 returns over 12 months: -0.0107")

print("\nBayesian RNN Model:")
print(f"- Average effect on S&P 500 monthly returns over 6 months: {effect_6m_sp500:.4f}")
print(f"- Average effect on S&P 500 monthly returns over 12 months: {effect_12m_sp500:.4f}")
print(f"- Cumulative effect on S&P 500 returns over 12 months: {cumulative_effect_12m_sp500:.4f}")

print("\nBayesian RNN model analysis complete!")