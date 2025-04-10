import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting style
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.2)

# Load the preprocessed data
data = pd.read_csv("financial_timeseries_preprocessed.csv", index_col=0, parse_dates=True)

# We'll focus on the stationary transformed series for VAR modeling
# Using the log returns of S&P 500 and the differenced Federal Funds Rate
var_data = data[['SP500_log_return', 'FedFunds_diff']].copy()

# Convert date index for better plotting
var_data.index = pd.to_datetime(var_data.index)

# Let's look at our data
print("Data shape:", var_data.shape)
print("\nFirst few rows:")
print(var_data.head())
print("\nStatistics:")
print(var_data.describe())

# Visualize the time series
plt.figure(figsize=(14, 8))

# Plot S&P 500 log returns
plt.subplot(2, 1, 1)
plt.plot(var_data.index, var_data['SP500_log_return'], 'b-', alpha=0.7)
plt.title('S&P 500 Monthly Log Returns')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.ylabel('Log Return')

# Plot Fed Funds Rate differences
plt.subplot(2, 1, 2)
plt.plot(var_data.index, var_data['FedFunds_diff'], 'g-', alpha=0.7)
plt.title('Federal Funds Rate Monthly Differences')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.ylabel('Rate Difference')

plt.tight_layout()
plt.savefig('time_series_visualization.png')
plt.close()

# INSERT THE DAG VISUALIZATION CODE HERE
# First, add the required import at the top of your file with other imports:
import networkx as nx

# Then add this function and function call:
# Create and visualize the causal DAG
def plot_causal_dag():
    """
    Create and visualize the explicit causal DAG for the interest rate -> asset price relationship
    """
    plt.figure(figsize=(8, 6))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for our variables
    G.add_node("Interest Rate\n(Fed Funds)", pos=(0, 0))
    G.add_node("Asset Price\n(S&P 500)", pos=(1, 0))
    
    # Add the causal edge (Interest Rate -> Asset Price)
    G.add_edge("Interest Rate\n(Fed Funds)", "Asset Price\n(S&P 500)")
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the DAG
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
            font_size=12, font_weight='bold', arrowsize=20, 
            width=2, edge_color='darkblue')
    
    # Add title and annotations
    plt.title('Causal Directed Acyclic Graph (DAG)', fontsize=14, fontweight='bold')
    plt.annotate('Causal assumption: Interest rates directly affect asset prices', 
                xy=(0.5, -0.1), xycoords='axes fraction', 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                                                  fc="lightyellow", ec="orange", alpha=0.8))
    
    # Explain the DAG interpretation
    plt.figtext(0.5, -0.2, 
                "This DAG encodes our economic assumption that interest rate changes\n"
                "causally influence asset prices, but not vice versa (in the short term).\n"
                "This structure guides our intervention analysis when applying Pearl's do-operator.",
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", 
                                                   fc="whitesmoke", ec="lightgray", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('causal_dag.png', bbox_inches='tight')
    plt.close()

# Call the function to create and visualize the DAG
print("Creating causal DAG visualization...")
plot_causal_dag()


# Define the lag order for our VAR model (p=2 as you created lag-2 features)
p = 2

# Prepare data for the VAR model
# We'll use the original series and create lag matrices ourselves
# This gives us more flexibility in the modeling
series_names = ['SP500_log_return', 'FedFunds_diff']
n_series = len(series_names)

# Trim the beginning of the dataset to account for lags
y = var_data.iloc[p:].values  # Response variables (t)
n_obs = len(y)

# Create design matrices (lagged predictors)
X = np.zeros((n_obs, n_series * p))

for i in range(p):
    X[:, i * n_series:(i + 1) * n_series] = var_data.iloc[p - i - 1:-i - 1].values

print(f"Response shape: {y.shape}, Design matrix shape: {X.shape}")

# Define and fit the Bayesian VAR model without explicitly modeling the covariance structure
def build_and_sample_bvar(y, X, n_series, p, tune=2000, draws=2000, chains=4):
    # Calculate some dimensions
    n_obs = y.shape[0]
    n_predictors = X.shape[1]  # Number of predictors (lags * series)
    
    # Construct the model
    with pm.Model() as bvar_model:
        # Priors for VAR coefficients (Minnesota-style prior)
        # We'll use normal priors with stronger priors on higher lags
        
        # Prior scale for own lags (diagonal elements) - higher for first lag
        own_lag_scales = np.ones(p) * 1.0
        own_lag_scales[0] = 2.0  # Less shrinkage
        cross_lag_scales = np.ones(p) * 0.5
        
        # Initialize coefficient array with appropriate priors
        coefs = []
        
        for i in range(n_series):  # For each equation
            eq_coefs = []
            
            for j in range(p):  # For each lag
                for k in range(n_series):  # For each series
                    if i == k:  # Own lag
                        # Center near 0.5 for first lag (persistence) with some shrinkage
                        mu = 0.5 if j == 0 else 0.0
                        scale = own_lag_scales[j]
                    else:  # Cross lag
                        # Center around 0 with stronger shrinkage
                        mu = 0.0
                        scale = cross_lag_scales[j]
                    
                    # Define the coefficient with normal prior
                    coef = pm.Normal(f'coef_{i}_{j}_{k}', mu=mu, sigma=scale)
                    eq_coefs.append(coef)
            
            coefs.append(eq_coefs)
        
        # Reshape coefficients for matrix operations
        beta = pm.math.stack([pm.math.stack(eq) for eq in coefs])
        
        # Prior for intercepts
        intercept = pm.Normal('intercept', mu=0, sigma=0.5, shape=n_series)
        
        # Prior for error standard deviations (equation-specific)
        sigma = pm.HalfNormal('sigma', sigma=1.0, shape=n_series)
        
        # Set up the likelihood for each equation separately
        for i in range(n_series):
            # Compute mean for this equation
            mu_i = intercept[i] + pm.math.sum(X * beta[i], axis=1)
            
            # Set up likelihood
            pm.Normal(f'y_{i}', mu=mu_i, sigma=sigma[i], observed=y[:, i])
        
        # Sample from the posterior
        trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9)
        
        # Get model information for later use
        model_info = {
            'n_series': n_series,
            'series_names': series_names,
            'p': p,
            'n_obs': n_obs
        }
        
    return bvar_model, trace, model_info

# Build and sample the Bayesian VAR model
print("Sampling from the Bayesian VAR model (this may take a while)...")
bvar_model, trace, model_info = build_and_sample_bvar(y, X, n_series, p)

# Print summary of posterior
print("\nPosterior Summary:")
summary = az.summary(trace)
print(summary)

# Save the trace for later use
az.to_netcdf(trace, 'bvar_trace.nc')

# Helper function to extract coefficients from trace
def get_var_coefficients(trace, n_series, p):
    # Initialize coefficient array
    coefs = np.zeros((n_series, n_series * p))
    
    # Fill in coefficients
    for i in range(n_series):
        for j in range(p):
            for k in range(n_series):
                coef_name = f'coef_{i}_{j}_{k}'
                coefs[i, j * n_series + k] = np.mean(trace.posterior[coef_name].values, axis=(0, 1))
    
    return coefs

# Get the mean VAR coefficients
coefs = get_var_coefficients(trace, n_series, p)

# Display coefficients in a more readable format
coef_df = pd.DataFrame(index=series_names)

for i, resp in enumerate(series_names):
    for j in range(p):
        for k, pred in enumerate(series_names):
            coef_name = f"{pred}_lag{j+1}"
            coef_df.loc[resp, coef_name] = coefs[i, j * n_series + k]

print("\nVAR Coefficients (posterior means):")
print(coef_df)

def plot_coefficient_posteriors(trace, n_series, p, series_names):
    plt.figure(figsize=(20, 16))
    
    # Create separate subplots for each row to ensure proper scaling
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    
    for i, resp in enumerate(series_names):
        for j in range(p):
            for k, pred in enumerate(series_names):
                ax = axes[i, j*n_series + k]
                
                coef_name = f'coef_{i}_{j}_{k}'
                coef_samples = trace.posterior[coef_name].values.flatten()
                
                # Calculate appropriate x-axis limits based on the data
                mean = np.mean(coef_samples)
                std = np.std(coef_samples)
                x_min = mean - 4*std
                x_max = mean + 4*std
                
                # Plot the distribution with controlled x-limits
                sns.kdeplot(coef_samples, fill=True, color='indianred', alpha=0.7, ax=ax)
                ax.axvline(x=0, color='darkred', linestyle='--', alpha=0.7)
                ax.set_xlim(x_min, x_max)
                
                # Add mean and 95% credible interval
                mean_val = np.mean(coef_samples)
                ci_lower = np.percentile(coef_samples, 2.5)
                ci_upper = np.percentile(coef_samples, 97.5)
                
                # Improved title with better formatting
                ax.set_title(f"{resp} ← {pred} (lag {j+1})", fontweight='bold', fontsize=14)
                
                # Add statistics in a text box
                textstr = f"Mean: {mean_val:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
                props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray")
                ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
    
    plt.suptitle("Posterior Distributions of VAR Coefficients", fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('coefficient_posteriors.png', dpi=300, bbox_inches='tight')
    plt.close()
# Plot coefficient posteriors
plot_coefficient_posteriors(trace, n_series, p, series_names)

# Create simulated residuals for impulse response analysis
def compute_residuals(y, X, coefs):
    # Fitted values
    y_hat = X @ coefs.T
    
    # Residuals
    return y - y_hat

# Get residuals
residuals = compute_residuals(y, X, coefs)

# Compute residual covariance matrix
sigma_resid = np.cov(residuals.T)

# Function to compute impulse response functions
def compute_impulse_responses(coefs, n_series, p, sigma_resid, steps=24):
    # Initialize IRFs
    irfs = np.zeros((steps, n_series, n_series))
    
    # Initialize companion matrix
    companion = np.zeros((n_series * p, n_series * p))
    
    # Fill in companion matrix with coefficients
    companion[:n_series, :] = coefs
    companion[n_series:, :-n_series] = np.eye(n_series * (p - 1))
    
    # Compute structural errors
    # Here we're using a simple Cholesky decomposition for identification
    # (Interest rates affect stock prices contemporaneously, but not vice versa)
    try:
        chol = np.linalg.cholesky(sigma_resid)
    except np.linalg.LinAlgError:
        # If Cholesky decomposition fails, add a small constant to diagonal
        print("Warning: Adding small value to covariance diagonal for numerical stability")
        sigma_resid_adjusted = sigma_resid + np.eye(n_series) * 1e-6
        chol = np.linalg.cholesky(sigma_resid_adjusted)
    
    # First period is just the Cholesky factor
    irfs[0, :, :] = chol
    
    # Compute remaining IRFs recursively
    eye = np.eye(n_series)
    for i in range(1, steps):
        prev_irf = np.zeros((n_series, n_series))
        for j in range(min(i, p)):
            prev_companion = np.linalg.matrix_power(companion, i - j - 1)
            prev_companion = prev_companion[:n_series, :n_series]
            prev_irf += np.dot(prev_companion, coefs[:, j * n_series:(j + 1) * n_series])
        
        irfs[i, :, :] = np.dot(prev_irf, chol)
    
    return irfs

# Compute impulse responses
irfs = compute_impulse_responses(coefs, n_series, p, sigma_resid)

# Plot impulse response functions
def plot_impulse_responses(irfs, series_names, steps=24):
    plt.figure(figsize=(12, 8))
    
    for i, resp in enumerate(series_names):
        for j, shock in enumerate(series_names):
            plt.subplot(n_series, n_series, i * n_series + j + 1)
            
            plt.plot(range(steps), irfs[:, i, j], 'b-', label='Point IRF')
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            plt.title(f"Response of {resp} to {shock} shock")
            if i == n_series - 1:
                plt.xlabel('Months')
            if j == 0:
                plt.ylabel('Response')
    
    plt.tight_layout()
    plt.savefig('impulse_responses.png')
    plt.close()

# Plot impulse responses
plot_impulse_responses(irfs, series_names)

# Forecast function with simplified VAR structure
def forecast_bvar(trace, X_last, n_series, p, steps=12, n_samples=500):
    """
    Generate forecasts from the Bayesian VAR model
    
    Parameters:
    -----------
    trace : arviz trace
        The MCMC trace from the VAR model
    X_last : ndarray
        The last observed values (used as initial condition)
    n_series : int
        Number of series in the model
    p : int
        Lag order of the VAR
    steps : int
        Number of steps to forecast
    n_samples : int
        Number of posterior samples to use
        
    Returns:
    --------
    all_forecasts : ndarray (n_samples, steps, n_series)
        Array of forecasts for each posterior sample
    """
    # Get a sample of coefficients from the posterior
    all_forecasts = np.zeros((n_samples, steps, n_series))
    
    # Get random sample indices
    sample_idxs = np.random.choice(trace.posterior.draw.size * trace.posterior.chain.size, 
                                   size=n_samples, replace=True)
    chain_idxs = sample_idxs // trace.posterior.draw.size
    draw_idxs = sample_idxs % trace.posterior.draw.size
    
    # Extract intercepts and sigmas
    intercepts = np.zeros((n_samples, n_series))
    sigmas = np.zeros((n_samples, n_series))
    
    # Extract coefficients
    coefficients = np.zeros((n_samples, n_series, n_series * p))
    
    # Fill in parameters from trace
    for i in range(n_series):
        intercepts[:, i] = trace.posterior.intercept.values[chain_idxs, draw_idxs, i]
        sigmas[:, i] = trace.posterior.sigma.values[chain_idxs, draw_idxs, i]
        
        for j in range(p):
            for k in range(n_series):
                coef_name = f'coef_{i}_{j}_{k}'
                coefficients[:, i, j * n_series + k] = trace.posterior[coef_name].values[chain_idxs, draw_idxs]
    
    # Initialize forecasts with the last observed values
    X_forecast = X_last.copy()  # Shape is (1, n_series * p)
    
    # Generate forecasts step by step
    for t in range(steps):
        # Generate forecast for each sample
        for s in range(n_samples):
            # Compute mean forecast - explicitly handle the computation to ensure scalar results
            mean = np.zeros(n_series)
            for i in range(n_series):
                # Explicitly calculate the mean for each series (equation)
                mean_i = intercepts[s, i]
                for j in range(n_series * p):
                    mean_i += X_forecast[0, j] * coefficients[s, i, j]
                mean[i] = mean_i
            
            # Add random noise
            forecast = np.zeros(n_series)
            for i in range(n_series):
                forecast[i] = mean[i] + np.random.normal(0, sigmas[s, i])
            
            # Store forecast
            all_forecasts[s, t] = forecast
            
            # Update X_forecast for next step
            if t < steps - 1:
                # Update X_forecast properly - shift values and add new ones
                # For a VAR(p) model, the new X becomes:
                # [y_t, y_{t-1}, ..., y_{t-p+1}] where y_t is our newly generated forecast
                
                # First, shift the existing values to the right
                for j in range(p-1, 0, -1):
                    start_idx = (j-1) * n_series
                    end_idx = j * n_series
                    target_start = j * n_series
                    target_end = (j+1) * n_series
                    X_forecast[0, target_start:target_end] = X_forecast[0, start_idx:end_idx]
                
                # Then, add the new forecast to the beginning
                X_forecast[0, 0:n_series] = forecast
    
    return all_forecasts

# Prepare last observed values for forecasting
X_last = X[-1:].copy()

# Generate forecasts
forecast_horizon = 24  # 24 months (2 years)
forecasts = forecast_bvar(trace, X_last, n_series, p, steps=forecast_horizon)

# Calculate forecast statistics
forecast_mean = np.mean(forecasts, axis=0)
forecast_lower = np.percentile(forecasts, 2.5, axis=0)
forecast_upper = np.percentile(forecasts, 97.5, axis=0)

# Prepare dates for forecasting
last_date = var_data.index[-1]
forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='ME')[1:]

# Plot forecasts
plt.figure(figsize=(14, 8))

for i, series in enumerate(series_names):
    plt.subplot(n_series, 1, i + 1)
    
    # Plot historical data
    hist_dates = var_data.index[-36:]  # Last 3 years
    hist_values = var_data[series].values[-36:]
    plt.plot(hist_dates, hist_values, 'b-', label='Historical')
    
    # Plot forecast mean
    plt.plot(forecast_dates, forecast_mean[:, i], 'r-', label='Forecast Mean')
    
    # Plot 95% credible interval
    plt.fill_between(forecast_dates, forecast_lower[:, i], forecast_upper[:, i], 
                    color='r', alpha=0.2, label='95% Credible Interval')
    
    plt.axvline(x=last_date, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Forecast for {series}')
    plt.legend()

plt.tight_layout()
plt.savefig('bvar_forecasts.png')
plt.close()

# Function to simulate causal effects (intervention analysis)
def simulate_intervention(trace, X_baseline, intervention_series, intervention_value, 
                         n_series, p, steps=24, n_samples=500):
    """
    Simulate the effect of an intervention on one series.
    
    Parameters:
    -----------
    trace : arviz trace
        The MCMC trace from the VAR model
    X_baseline : ndarray
        The baseline (last observed) values
    intervention_series : int
        Index of the series to intervene on (0 for SP500_log_return, 1 for FedFunds_diff)
    intervention_value : float
        The value to set for the intervention
    n_series : int
        Number of series in the model
    p : int
        Lag order of the VAR
    steps : int
        Number of steps to forecast
    n_samples : int
        Number of posterior samples to use
        
    Returns:
    --------
    baseline_forecasts : ndarray
        Forecasts without intervention
    intervention_forecasts : ndarray
        Forecasts with intervention
    """
    # Generate baseline forecasts
    baseline_forecasts = forecast_bvar(trace, X_baseline, n_series, p, steps, n_samples)
    
    # Generate intervention forecasts (similar to baseline but with intervention)
    intervention_forecasts = np.zeros_like(baseline_forecasts)
    
    # Get random sample indices
    sample_idxs = np.random.choice(trace.posterior.draw.size * trace.posterior.chain.size, 
                                   size=n_samples, replace=True)
    chain_idxs = sample_idxs // trace.posterior.draw.size
    draw_idxs = sample_idxs % trace.posterior.draw.size
    
    # Extract parameters from trace
    intercepts = np.zeros((n_samples, n_series))
    sigmas = np.zeros((n_samples, n_series))
    coefficients = np.zeros((n_samples, n_series, n_series * p))
    
    # Fill in parameters
    for i in range(n_series):
        intercepts[:, i] = trace.posterior.intercept.values[chain_idxs, draw_idxs, i]
        sigmas[:, i] = trace.posterior.sigma.values[chain_idxs, draw_idxs, i]
        
        for j in range(p):
            for k in range(n_series):
                coef_name = f'coef_{i}_{j}_{k}'
                coefficients[:, i, j * n_series + k] = trace.posterior[coef_name].values[chain_idxs, draw_idxs]
    
    # Initialize forecasts with the last observed values
    X_forecast = X_baseline.copy()
    
    # Apply intervention
    for t in range(steps):
        # Generate forecast for each sample
        for s in range(n_samples):
            # Compute mean forecast - explicitly handle the computation to ensure scalar results
            mean = np.zeros(n_series)
            for i in range(n_series):
                # Explicitly calculate the mean for each series (equation)
                mean_i = intercepts[s, i]
                for j in range(n_series * p):
                    mean_i += X_forecast[0, j] * coefficients[s, i, j]
                mean[i] = mean_i
            
            # Add random noise
            forecast = np.zeros(n_series)
            for i in range(n_series):
                forecast[i] = mean[i] + np.random.normal(0, sigmas[s, i])
            
            # Apply intervention at each step (do-operator)
            # This overrides the forecast for the intervention series
            if t == 0:  # Only intervene at the first step
                forecast[intervention_series] = intervention_value
            
            # Store forecast
            intervention_forecasts[s, t] = forecast
            
            # Update X_forecast for next step
            if t < steps - 1:
                # Shift data and add new forecast
                X_forecast = np.roll(X_forecast, -n_series)
                X_forecast[-n_series:] = forecast
    
    return baseline_forecasts, intervention_forecasts

# Define an intervention scenario: Fed Funds Rate hike of 0.5%
intervention_series = 1  # Index for FedFunds_diff
intervention_value = 0.5  # 0.5 percentage point increase

# Generate baseline forecasts first (no intervention)
baseline_forecasts = forecast_bvar(trace, X_last, n_series, p, steps=forecast_horizon)

# Now generate intervention forecasts separately
# This approach avoids issues with the original simulate_intervention function
# Get random sample indices
n_samples = 500
sample_idxs = np.random.choice(trace.posterior.draw.size * trace.posterior.chain.size, 
                               size=n_samples, replace=True)
chain_idxs = sample_idxs // trace.posterior.draw.size
draw_idxs = sample_idxs % trace.posterior.draw.size

# Extract parameters from trace
intercepts = np.zeros((n_samples, n_series))
sigmas = np.zeros((n_samples, n_series))
coefficients = np.zeros((n_samples, n_series, n_series * p))

# Fill in parameters
for i in range(n_series):
    intercepts[:, i] = trace.posterior.intercept.values[chain_idxs, draw_idxs, i]
    sigmas[:, i] = trace.posterior.sigma.values[chain_idxs, draw_idxs, i]
    
    for j in range(p):
        for k in range(n_series):
            coef_name = f'coef_{i}_{j}_{k}'
            coefficients[:, i, j * n_series + k] = trace.posterior[coef_name].values[chain_idxs, draw_idxs]

# Initialize intervention forecasts array
intervention_forecasts = np.zeros((n_samples, forecast_horizon, n_series))

# For each sample, generate a forecast with intervention
for s in range(n_samples):
    # Copy X_last for this sample
    X_forecast = X_last.copy()
    
    # Generate forecasts step by step
    for t in range(forecast_horizon):
        # Compute mean forecast - explicitly handle the computation to ensure scalar results
        mean = np.zeros(n_series)
        for i in range(n_series):
            # Explicitly calculate the mean for each series (equation)
            mean_i = intercepts[s, i]
            for j in range(n_series * p):
                mean_i += X_forecast[0, j] * coefficients[s, i, j]
            mean[i] = mean_i
        
        # Add random noise
        forecast = np.zeros(n_series)
        for i in range(n_series):
            forecast[i] = mean[i] + np.random.normal(0, sigmas[s, i])
        
        # Apply intervention at time 0
        if t == 0:
            forecast[intervention_series] = intervention_value
        
        # Store forecast
        intervention_forecasts[s, t] = forecast
        
        # Update X_forecast for next step
        if t < forecast_horizon - 1:
            # Update X_forecast properly - shift values and add new ones
            # First, shift the existing values to the right
            for j in range(p-1, 0, -1):
                start_idx = (j-1) * n_series
                end_idx = j * n_series
                target_start = j * n_series
                target_end = (j+1) * n_series
                X_forecast[0, target_start:target_end] = X_forecast[0, start_idx:end_idx]
            
            # Then, add the new forecast to the beginning
            X_forecast[0, 0:n_series] = forecast

# Calculate statistics for intervention simulation
baseline_mean = np.mean(baseline_forecasts, axis=0)
baseline_lower = np.percentile(baseline_forecasts, 2.5, axis=0)
baseline_upper = np.percentile(baseline_forecasts, 97.5, axis=0)

intervention_mean = np.mean(intervention_forecasts, axis=0)
intervention_lower = np.percentile(intervention_forecasts, 2.5, axis=0)
intervention_upper = np.percentile(intervention_forecasts, 97.5, axis=0)

# Calculate causal effect (mean difference)
causal_effect = intervention_mean - baseline_mean
causal_effect_lower = np.percentile(intervention_forecasts - baseline_forecasts, 2.5, axis=0)
causal_effect_upper = np.percentile(intervention_forecasts - baseline_forecasts, 97.5, axis=0)

# Plot intervention analysis
plt.figure(figsize=(16, 12))

# Plot both series with and without intervention
for i, series in enumerate(series_names):
    plt.subplot(3, n_series, i + 1)
    
    # Plot baseline forecast
    plt.plot(forecast_dates, baseline_mean[:, i], 'b-', label='Baseline')
    plt.fill_between(forecast_dates, baseline_lower[:, i], baseline_upper[:, i], 
                    color='b', alpha=0.1)
    
    # Plot intervention forecast
    plt.plot(forecast_dates, intervention_mean[:, i], 'r-', label='Intervention')
    plt.fill_between(forecast_dates, intervention_lower[:, i], intervention_upper[:, i], 
                    color='r', alpha=0.1)
    
    plt.title(f'{series}: Baseline vs. Intervention')
    plt.legend()
    
    # If this is the intervention variable, add a note
    if i == intervention_series:
        plt.annotate(f'Intervention: +{intervention_value}', xy=(0.05, 0.9), 
                    xycoords='axes fraction', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

# Plot the causal effect (difference)
for i, series in enumerate(series_names):
    plt.subplot(3, n_series, n_series + i + 1)
    
    plt.plot(forecast_dates, causal_effect[:, i], 'g-', label='Causal Effect')
    plt.fill_between(forecast_dates, causal_effect_lower[:, i], causal_effect_upper[:, i], 
                    color='g', alpha=0.2)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Causal Effect on {series}')
    
    # Compute cumulative effect over time
    cum_effect = np.cumsum(causal_effect[:, i])
    
    plt.subplot(3, n_series, 2*n_series + i + 1)
    plt.plot(forecast_dates, cum_effect, 'm-', label='Cumulative Effect')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Cumulative Effect on {series}')

plt.tight_layout()
plt.savefig('intervention_analysis.png')
plt.close()

print("\nIntervention Analysis Results:")
print(f"- Interest Rate Intervention: +{intervention_value} percentage points")

# Calculate the average effect on S&P 500 returns over the first 6 months
effect_6m = causal_effect[:6, 0].mean()
effect_12m = causal_effect[:12, 0].mean()
cumulative_effect_12m = np.cumsum(causal_effect[:12, 0])[-1]

print(f"- Average effect on S&P 500 monthly returns over 6 months: {effect_6m:.4f}")
print(f"- Average effect on S&P 500 monthly returns over 12 months: {effect_12m:.4f}")
print(f"- Cumulative effect on S&P 500 returns over 12 months: {cumulative_effect_12m:.4f}")

print("\nBayesian VAR model analysis complete!")